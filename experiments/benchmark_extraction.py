#!/usr/bin/env python3
"""
Benchmark OSM extraction and InaRISK risk scoring in isolation.
Skips graph build, routing, and assignment entirely.

Parallelization scenarios
-------------------------
  --osm-modes sequential parallel
      sequential  — road network, villages, shelters extracted one by one
      parallel    — all three extracted concurrently via ThreadPoolExecutor(3)
                    (same as parallel_runner does internally)

  --inarisk-threads 1 2 4 8
      Run InaRISK POI enrichment with each thread count as a separate
      result entry, so you can plot the speedup curve.
      With cached data this mostly measures ThreadPoolExecutor overhead;
      use --no-cache-inarisk to measure actual API throughput.

Usage
-----
  # Cached run, single scenario (default: sequential OSM, 4 InaRISK threads)
  python -m experiments.benchmark_extraction --scenarios banjarnegara_landslide_2021

  # Compare sequential vs parallel OSM extraction
  python -m experiments.benchmark_extraction --osm-modes sequential parallel

  # Compare InaRISK thread counts
  python -m experiments.benchmark_extraction --inarisk-threads 1 2 4 8

  # Full matrix: both OSM modes × all thread counts (fresh InaRISK for real timings)
  python -m experiments.benchmark_extraction \\
      --osm-modes sequential parallel \\
      --inarisk-threads 1 2 4 8 \\
      --no-cache-inarisk

  # Force fresh download of everything
  python -m experiments.benchmark_extraction --no-cache

  # POI risk only (skip InaRISK edge enrichment — no graph build needed)
  python -m experiments.benchmark_extraction --skip-inarisk-edges

  # All 10 scenarios
  python -m experiments.benchmark_extraction

Output
------
  output/benchmark_extraction.json  — all result entries
  Summary table printed to stdout
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_extraction")

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR  = PROJECT_ROOT / "configs"
OUTPUT_DIR   = PROJECT_ROOT / "output"


# ------------------------------------------------------------------ #
# OSM extraction helpers
# ------------------------------------------------------------------ #

def _extract_osm_sequential(extractor, region, cfg, use_cache: bool) -> tuple:
    """Extract road network, villages, shelters one by one. Returns (nodes, edges, villages, shelters, sub_timings)."""
    from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
    from src.hpc.runner_utils import MemoryTracker

    mem = MemoryTracker()
    sub_timings = {}
    memory_mb = {}

    m0 = mem.rss_mb()
    t0 = time.perf_counter()
    if cfg.preloaded_network_pycgr:
        nodes, edges = extractor.load_network_from_pycgr(cfg.preloaded_network_pycgr)
    elif cfg.preloaded_network_json:
        nodes, edges = extractor.load_network_from_json(cfg.preloaded_network_json)
    else:
        nodes, edges = extractor.extract_road_network(
            region,
            network_type=cfg.extraction.network_type,
            road_types=cfg.extraction.road_types,
            use_cache=use_cache,
        )
    sub_timings["osm_network"] = time.perf_counter() - t0
    m0 = mem.snapshot("osm_network", m0, memory_mb)

    t0 = time.perf_counter()
    if cfg.preloaded_villages_geojson:
        villages = extractor.load_villages_from_geojson(cfg.preloaded_villages_geojson)
    else:
        villages = extractor.extract_villages(
            region,
            admin_levels=cfg.extraction.village_admin_levels,
            population_density_per_km2=cfg.extraction.village_pop_density,
            max_population_per_village=cfg.extraction.village_max_pop,
            use_cache=use_cache,
            sources=cfg.extraction.village_sources,
            place_tags=cfg.extraction.village_place_tags,
            place_settings=cfg.extraction.village_place_settings,
            place_radius_m=cfg.extraction.village_place_radius_m,
            cluster_eps_m=cfg.extraction.village_cluster_eps_m,
            cluster_min_buildings=cfg.extraction.village_cluster_min_buildings,
            cluster_max_area_km2=cfg.extraction.village_cluster_max_area_km2,
            persons_per_dwelling=cfg.extraction.village_persons_per_dwelling,
            building_persons=cfg.extraction.village_building_persons,
            fill_uncovered_l9=cfg.extraction.village_fill_uncovered_l9,
        )
    PopulationLoader().apply_population(
        villages,
        population_csv=cfg.extraction.population_csv,
        density_per_km2=cfg.extraction.village_pop_density,
    )
    sub_timings["osm_villages"] = time.perf_counter() - t0
    m0 = mem.snapshot("osm_villages", m0, memory_mb)

    t0 = time.perf_counter()
    if cfg.preloaded_shelters_geojson:
        shelters = extractor.load_shelters_from_geojson(cfg.preloaded_shelters_geojson)
    else:
        shelters = extractor.extract_shelters(
            region,
            shelter_tags=cfg.extraction.shelter_tags,
            min_area_m2=cfg.extraction.shelter_min_area_m2,
            m2_per_person=cfg.extraction.shelter_m2_per_person,
            use_cache=use_cache,
            cluster_eps_m=cfg.extraction.shelter_cluster_eps_m,
            cluster_min_shelters=cfg.extraction.shelter_cluster_min_shelters,
        )
    ShelterCapacityLoader().apply_capacity(
        shelters,
        capacity_csv=cfg.extraction.shelter_capacity_csv,
        m2_per_person=cfg.extraction.shelter_m2_per_person,
    )
    sub_timings["osm_shelters"] = time.perf_counter() - t0
    mem.snapshot("osm_shelters", m0, memory_mb)

    return nodes, edges, villages, shelters, sub_timings, memory_mb


def _extract_osm_parallel(extractor, region, cfg, use_cache: bool) -> tuple:
    """
    Extract road network, villages, shelters concurrently (ThreadPoolExecutor(3)).
    Same approach as ParallelRunner._extract_parallel().
    Returns (nodes, edges, villages, shelters, sub_timings, memory_mb).

    Wall time = max(network_t, villages_t, shelters_t) — the slowest task.
    Individual task times are still recorded for transparency.
    """
    from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
    from src.hpc.runner_utils import MemoryTracker

    mem = MemoryTracker()
    m0 = mem.rss_mb()
    task_times = {}

    def _extract_network():
        t = time.perf_counter()
        if cfg.preloaded_network_pycgr:
            result = extractor.load_network_from_pycgr(cfg.preloaded_network_pycgr)
        elif cfg.preloaded_network_json:
            result = extractor.load_network_from_json(cfg.preloaded_network_json)
        else:
            result = extractor.extract_road_network(
                region,
                network_type=cfg.extraction.network_type,
                road_types=cfg.extraction.road_types,
                use_cache=use_cache,
            )
        task_times["osm_network"] = time.perf_counter() - t
        return result

    def _extract_villages():
        t = time.perf_counter()
        if cfg.preloaded_villages_geojson:
            v = extractor.load_villages_from_geojson(cfg.preloaded_villages_geojson)
        else:
            v = extractor.extract_villages(
                region,
                admin_levels=cfg.extraction.village_admin_levels,
                population_density_per_km2=cfg.extraction.village_pop_density,
                max_population_per_village=cfg.extraction.village_max_pop,
                use_cache=use_cache,
                sources=cfg.extraction.village_sources,
                place_tags=cfg.extraction.village_place_tags,
                place_settings=cfg.extraction.village_place_settings,
                place_radius_m=cfg.extraction.village_place_radius_m,
                cluster_eps_m=cfg.extraction.village_cluster_eps_m,
                cluster_min_buildings=cfg.extraction.village_cluster_min_buildings,
                cluster_max_area_km2=cfg.extraction.village_cluster_max_area_km2,
                persons_per_dwelling=cfg.extraction.village_persons_per_dwelling,
                building_persons=cfg.extraction.village_building_persons,
                fill_uncovered_l9=cfg.extraction.village_fill_uncovered_l9,
            )
        PopulationLoader().apply_population(
            v,
            population_csv=cfg.extraction.population_csv,
            density_per_km2=cfg.extraction.village_pop_density,
        )
        task_times["osm_villages"] = time.perf_counter() - t
        return v

    def _extract_shelters():
        t = time.perf_counter()
        if cfg.preloaded_shelters_geojson:
            s = extractor.load_shelters_from_geojson(cfg.preloaded_shelters_geojson)
        else:
            s = extractor.extract_shelters(
                region,
                shelter_tags=cfg.extraction.shelter_tags,
                min_area_m2=cfg.extraction.shelter_min_area_m2,
                m2_per_person=cfg.extraction.shelter_m2_per_person,
                use_cache=use_cache,
                cluster_eps_m=cfg.extraction.shelter_cluster_eps_m,
                cluster_min_shelters=cfg.extraction.shelter_cluster_min_shelters,
            )
        ShelterCapacityLoader().apply_capacity(
            s,
            capacity_csv=cfg.extraction.shelter_capacity_csv,
            m2_per_person=cfg.extraction.shelter_m2_per_person,
        )
        task_times["osm_shelters"] = time.perf_counter() - t
        return s

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as tpe:
        f_net  = tpe.submit(_extract_network)
        f_vill = tpe.submit(_extract_villages)
        f_shlt = tpe.submit(_extract_shelters)
        nodes, edges = f_net.result()
        villages     = f_vill.result()
        shelters     = f_shlt.result()
    wall_time = time.perf_counter() - wall_start

    # sub_timings: individual thread times + wall time
    sub_timings = dict(task_times)
    sub_timings["osm_parallel_wall"] = wall_time   # what the user actually waits

    # Memory after all three complete (approximate — threads ran concurrently)
    memory_mb = {}
    mem.snapshot("osm_parallel", m0, memory_mb)

    return nodes, edges, villages, shelters, sub_timings, memory_mb


# ------------------------------------------------------------------ #
# Core benchmark function
# ------------------------------------------------------------------ #

def _bench_scenario(
    cfg,
    use_cache_osm: bool,
    use_cache_inarisk: bool,
    osm_mode: str = "sequential",      # "sequential" | "parallel"
    inarisk_threads: int = 4,
    skip_inarisk_edges: bool = False,
) -> dict:
    """
    Run OSM extraction + InaRISK for one (scenario, osm_mode, inarisk_threads) combination.
    Returns a dict with per-phase timings, memory, and metadata.
    """
    from src.data.models import DisasterInput, RegionOfInterest, RegionType, DisasterType
    from src.data.osm_extractor import OSMExtractor
    from src.hpc.runner_utils import resolve_hazard_layers, MemoryTracker, apply_risk_parallel

    mem = MemoryTracker()
    timings: dict   = {}
    memory_mb: dict = {}
    meta: dict      = {}

    disaster = DisasterInput(
        location=(cfg.disaster.lat, cfg.disaster.lon),
        disaster_type=DisasterType(cfg.disaster.disaster_type),
        name=cfg.disaster.name,
        severity=cfg.disaster.severity,
    )
    region = RegionOfInterest(
        region_type=RegionType(cfg.region.region_type),
        bbox=tuple(cfg.region.bbox) if cfg.region.bbox else None,
        center=tuple(cfg.region.center) if cfg.region.center else None,
        radius_km=cfg.region.radius_km,
    )

    extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)

    # ── Phase: OSM extraction (sequential or parallel) ──────────────────
    if osm_mode == "parallel":
        nodes, edges, villages, shelters, sub_t, sub_mem = _extract_osm_parallel(
            extractor, region, cfg, use_cache_osm
        )
        log.info(
            f"  OSM parallel wall={sub_t.get('osm_parallel_wall', 0):.2f}s "
            f"(net={sub_t.get('osm_network', 0):.2f}s "
            f"vill={sub_t.get('osm_villages', 0):.2f}s "
            f"shlt={sub_t.get('osm_shelters', 0):.2f}s)"
        )
    else:
        nodes, edges, villages, shelters, sub_t, sub_mem = _extract_osm_sequential(
            extractor, region, cfg, use_cache_osm
        )
        log.info(
            f"  OSM sequential: net={sub_t.get('osm_network', 0):.2f}s "
            f"vill={sub_t.get('osm_villages', 0):.2f}s "
            f"shlt={sub_t.get('osm_shelters', 0):.2f}s"
        )

    timings.update(sub_t)
    memory_mb.update(sub_mem)
    meta.update({
        "n_nodes": len(nodes), "n_edges": len(edges),
        "n_villages": len(villages), "n_shelters": len(shelters),
    })

    # ── Phase: InaRISK POI risk scoring ─────────────────────────────────
    if cfg.skip_inarisk:
        log.info("  InaRISK POI: skipped (skip_inarisk=true in config)")
        timings["inarisk_poi"] = 0.0
    else:
        m0 = mem.rss_mb()
        t0 = time.perf_counter()
        apply_risk_parallel(cfg, villages, shelters, disaster, n_threads=inarisk_threads)
        timings["inarisk_poi"] = time.perf_counter() - t0
        mem.snapshot("inarisk_poi", m0, memory_mb)
        log.info(
            f"  InaRISK POI: {timings['inarisk_poi']:.2f}s "
            f"({inarisk_threads} thread{'s' if inarisk_threads != 1 else ''})"
        )

    # ── Phase: InaRISK road edges ────────────────────────────────────────
    if not cfg.skip_inarisk and not skip_inarisk_edges:
        from src.graph.graph_builder import EvacuationGraphBuilder
        from src.data.inarisk_client import InaRISKClient

        m0 = mem.rss_mb()
        t0 = time.perf_counter()
        builder = EvacuationGraphBuilder()
        G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
        builder.attach_pois_to_graph(villages, shelters)
        hazard_layers = resolve_hazard_layers(cfg, disaster)
        inarisk = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )
        builder.apply_inarisk_to_edges(
            inarisk=inarisk,
            hazard_layers=hazard_layers,
            aggregation=cfg.routing.hazard_aggregation,
            cache_path=Path(cfg.extraction.inarisk_cache_dir) / "road_risk_cache.json",
            use_cache=use_cache_inarisk,
        )
        timings["inarisk_edges"] = time.perf_counter() - t0
        mem.snapshot("inarisk_edges", m0, memory_mb)
        meta["n_graph_nodes"] = G.number_of_nodes()
        meta["n_graph_edges"] = G.number_of_edges()
        log.info(f"  InaRISK edges: {timings['inarisk_edges']:.2f}s")

    # ── Totals ───────────────────────────────────────────────────────────
    osm_keys     = {"osm_network", "osm_villages", "osm_shelters", "osm_parallel_wall"}
    inarisk_keys = {"inarisk_poi", "inarisk_edges"}

    if osm_mode == "parallel":
        timings["total_osm"] = timings.get("osm_parallel_wall", 0.0)
    else:
        timings["total_osm"] = sum(v for k, v in timings.items() if k in osm_keys)

    timings["total_inarisk"] = sum(v for k, v in timings.items() if k in inarisk_keys)
    timings["total"] = timings["total_osm"] + timings["total_inarisk"]

    return {
        "timings":    timings,
        "memory_mb":  memory_mb,
        "peak_rss_mb": round(mem.peak_rss_mb(), 1),
        "meta":       meta,
    }


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OSM extraction and InaRISK risk scoring with parallelization variants"
    )
    parser.add_argument(
        "--scenarios", nargs="*",
        help="Scenario IDs (default: all configs/*.yaml)",
    )
    parser.add_argument(
        "--osm-modes", nargs="+", choices=["sequential", "parallel"],
        default=["sequential"],
        metavar="MODE",
        help="OSM extraction modes to benchmark (default: sequential). "
             "parallel runs network+villages+shelters concurrently via ThreadPoolExecutor(3).",
    )
    parser.add_argument(
        "--inarisk-threads", nargs="+", type=int,
        default=[4],
        metavar="N",
        help="InaRISK POI thread counts to benchmark (default: 4). "
             "Each value produces a separate result entry. "
             "Use --no-cache-inarisk for meaningful API throughput numbers.",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Force fresh download of both OSM and InaRISK data",
    )
    parser.add_argument(
        "--no-cache-osm", action="store_true",
        help="Force fresh OSM download only",
    )
    parser.add_argument(
        "--no-cache-inarisk", action="store_true",
        help="Force fresh InaRISK queries only (useful with --inarisk-threads to measure real API speedup)",
    )
    parser.add_argument(
        "--skip-inarisk-edges", action="store_true",
        help="Skip InaRISK edge enrichment (no graph build needed — POI risk only)",
    )
    args = parser.parse_args()

    use_cache_osm     = not (args.no_cache or args.no_cache_osm)
    use_cache_inarisk = not (args.no_cache or args.no_cache_inarisk)

    if args.scenarios:
        configs_paths = [CONFIGS_DIR / f"{s}.yaml" for s in args.scenarios]
        missing = [p for p in configs_paths if not p.exists()]
        if missing:
            parser.error(f"Config(s) not found: {[str(p) for p in missing]}")
    else:
        configs_paths = sorted(CONFIGS_DIR.glob("*.yaml"))

    from src.config.config_loader import load_config

    # Build the full variant matrix: (osm_mode, inarisk_threads) combinations
    variants = [
        {"osm_mode": osm_mode, "inarisk_threads": n_threads}
        for osm_mode in args.osm_modes
        for n_threads in args.inarisk_threads
    ]

    n_total = len(configs_paths) * len(variants)
    log.info(
        f"Benchmark plan: {len(configs_paths)} scenario(s) × "
        f"{len(variants)} variant(s) = {n_total} run(s)"
    )
    log.info(
        f"  OSM modes: {args.osm_modes}  |  "
        f"InaRISK threads: {args.inarisk_threads}  |  "
        f"cache: OSM={'warm' if use_cache_osm else 'COLD'} "
        f"InaRISK={'warm' if use_cache_inarisk else 'COLD'}"
    )

    results = []
    n_done  = 0

    for cfg_path in configs_paths:
        scenario_id = cfg_path.stem
        log.info(f"\n{'='*60}")
        log.info(f"Scenario: {scenario_id}")
        log.info(f"{'='*60}")

        try:
            cfg = load_config(str(cfg_path))
        except Exception as e:
            log.error(f"  Failed to load config: {e}")
            continue

        for variant in variants:
            osm_mode       = variant["osm_mode"]
            inarisk_threads = variant["inarisk_threads"]
            variant_id = f"osm_{osm_mode}__inarisk_{inarisk_threads}t"
            log.info(f"\n  Variant: {variant_id}")

            try:
                result = _bench_scenario(
                    cfg,
                    use_cache_osm=use_cache_osm,
                    use_cache_inarisk=use_cache_inarisk,
                    osm_mode=osm_mode,
                    inarisk_threads=inarisk_threads,
                    skip_inarisk_edges=args.skip_inarisk_edges,
                )
                result["success"] = True
            except Exception as e:
                log.error(f"  FAILED: {e}")
                result = {"success": False, "error": str(e)}

            result.update({
                "scenario":         scenario_id,
                "variant_id":       variant_id,
                "osm_mode":         osm_mode,
                "inarisk_threads":  inarisk_threads,
                "use_cache_osm":    use_cache_osm,
                "use_cache_inarisk": use_cache_inarisk,
                "timestamp":        datetime.now().isoformat(),
            })
            results.append(result)
            n_done += 1

            if result["success"]:
                t = result["timings"]
                log.info(
                    f"  → OSM={t.get('total_osm', 0):.2f}s  "
                    f"InaRISK_poi={t.get('inarisk_poi', 0):.2f}s  "
                    f"InaRISK_edges={t.get('inarisk_edges', 0):.2f}s  "
                    f"total={t.get('total', 0):.2f}s  "
                    f"peak={result.get('peak_rss_mb', 0):.0f} MiB"
                )
            log.info(f"  Progress: {n_done}/{n_total}")

    # Save JSON
    out_path = OUTPUT_DIR / "benchmark_extraction.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results, "updated": datetime.now().isoformat()}, f, indent=2)
    log.info(f"\nResults → {out_path}")

    _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    successful = [r for r in results if r.get("success")]
    if not successful:
        return

    log.info("\n── Extraction benchmark summary ──")

    scenarios = sorted({r["scenario"] for r in successful})
    variants  = list({r["variant_id"]: r for r in successful}.keys())  # preserve order

    # Short labels: "seq/1t", "seq/2t", "par/4t", etc.
    def _short(v: str) -> str:
        osm = "par" if "parallel" in v else "seq"
        t = v.split("inarisk_")[1].replace("t", "")
        return f"{osm}/{t}t"

    short_labels = [_short(v) for v in variants]
    col_w = max(8, max(len(s) for s in short_labels) + 2)

    # Per-scenario total time table
    hdr = f"{'Scenario':<36}" + "".join(f"{s:>{col_w}}" for s in short_labels)
    log.info(hdr)
    log.info("-" * len(hdr))
    for sc in scenarios:
        sc_results = {r["variant_id"]: r for r in successful if r["scenario"] == sc}
        row = f"{sc:<36}"
        for v in variants:
            if v in sc_results:
                total = sc_results[v]["timings"].get("total", 0)
                row += f"{total:>{col_w}.2f}s"
            else:
                row += f"{'—':>{col_w}}"
        log.info(row)

    # Detailed phase + memory breakdown for the first scenario
    if len(variants) > 1:
        first_sc = scenarios[0]
        sc_results = {r["variant_id"]: r for r in successful if r["scenario"] == first_sc}

        log.info(f"\n── Phase timing (s) — {first_sc} ──")
        phases = ["osm_network", "osm_villages", "osm_shelters",
                  "osm_parallel_wall", "inarisk_poi", "inarisk_edges",
                  "total_osm", "total_inarisk"]
        hdr2 = f"{'Phase':<22}" + "".join(f"{s:>{col_w}}" for s in short_labels)
        log.info(hdr2)
        log.info("-" * len(hdr2))
        for ph in phases:
            row = f"{ph:<22}"
            any_val = False
            for v in variants:
                val = sc_results.get(v, {}).get("timings", {}).get(ph)
                if val is not None:
                    row += f"{val:>{col_w}.3f}s"
                    any_val = True
                else:
                    row += f"{'—':>{col_w}}"
            if any_val:
                log.info(row)

        log.info(f"\n── Memory delta (MiB) — {first_sc} ──")
        mem_phases = ["osm_network", "osm_villages", "osm_shelters",
                      "osm_parallel", "inarisk_poi", "inarisk_edges"]
        hdr3 = f"{'Phase':<22}" + "".join(f"{s:>{col_w}}" for s in short_labels)
        log.info(hdr3)
        log.info("-" * len(hdr3))
        for ph in mem_phases:
            row = f"{ph:<22}"
            any_val = False
            for v in variants:
                delta = sc_results.get(v, {}).get("memory_mb", {}).get(ph, {}).get("delta_mb")
                if delta is not None:
                    row += f"{delta:>{col_w}.1f}"
                    any_val = True
                else:
                    row += f"{'—':>{col_w}}"
            if any_val:
                log.info(row)


if __name__ == "__main__":
    main()
