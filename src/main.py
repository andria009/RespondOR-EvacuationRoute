"""
RespondOR-EvacuationRoute — Main Entrypoint
Disaster evacuation route optimization system.

Usage:
  # Run full pipeline (naive mode):
  python -m src.main --config configs/demak_flood_2024.yaml

  # Parallel mode (multiprocessing, single node):
  python -m src.main --config configs/demak_flood_2024.yaml --mode parallel --workers 8

  # HPC mode (MPI, launch via srun/mpirun):
  srun --mpi=pmix -n 32 python -m src.main --config configs/demak_flood_2024.yaml --mode hpc

  # Generate visualization only:
  python -m src.main --config configs/demak_flood_2024.yaml --visualize-only

  # Prepare GAMA simulation inputs (after optimization):
  python -m experiments.prepare_gama_inputs --config configs/demak_flood_2024.yaml
"""

import argparse
import json
import logging
from pathlib import Path

from src.utils.logging_setup import setup_logging as _setup_logging


def setup_logging(level: str = "INFO"):
    _setup_logging("main", level=level)


def parse_args():
    p = argparse.ArgumentParser(
        description="RespondOR-EvacuationRoute: Disaster evacuation optimization system"
    )
    p.add_argument("--config", "-c", required=True,
                   help="Path to YAML/JSON config file")
    p.add_argument("--mode", choices=["naive", "parallel", "hpc"],
                   help="Execution mode (overrides config)")
    p.add_argument("--workers", "-w", type=int,
                   help="Number of parallel workers (overrides config; ignored in HPC/MPI mode)")
    p.add_argument("--visualize-only", action="store_true",
                   help="Regenerate visualizations from existing results")
    p.add_argument("--output-dir", "-o",
                   help="Output directory (overrides config)")
    p.add_argument("--log-level", default=None,
                   help="Logging level (DEBUG/INFO/WARNING/ERROR)")
    p.add_argument("--assignment-method", choices=["greedy", "lp"],
                   help="Assignment algorithm (overrides config): greedy (fast) | lp (optimal)")
    p.add_argument("--village-limit", type=int, default=None,
                   help="Cap number of villages processed (overrides config; useful for benchmarking)")
    return p.parse_args()


def run_optimization(config, mode_override=None, workers_override=None,
                     assignment_method_override=None, village_limit_override=None):
    """Run the optimization pipeline in the specified mode."""
    from src.data.models import ExecutionMode

    if mode_override:
        config.execution.mode = mode_override
    if workers_override:
        config.execution.n_workers = workers_override
    if assignment_method_override:
        config.routing.assignment_method = assignment_method_override
    if village_limit_override is not None:
        config.benchmark_village_limit = village_limit_override

    mode = ExecutionMode(config.execution.mode)
    logging.info(f"Starting optimization in {mode.value.upper()} mode")

    if mode == ExecutionMode.NAIVE:
        from src.hpc.naive_runner import NaiveRunner
        runner = NaiveRunner(config)
    elif mode == ExecutionMode.PARALLEL:
        from src.hpc.parallel_runner import ParallelRunner
        runner = ParallelRunner(config)
    elif mode == ExecutionMode.HPC:
        from src.hpc.distributed_runner import MPIRunner
        runner = MPIRunner(config)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return runner.run()


def save_optimization_result(result, villages, shelters, routes_by_village, output_dir: str):
    """Save optimization results to JSON for downstream use."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = {
        "scenario_id": result.scenario_id,
        "mode": result.mode.value,
        "total_population": result.total_population,
        "total_evacuated": result.total_evacuated,
        "total_unmet": result.total_unmet,
        "evacuation_ratio": round(result.evacuation_ratio, 4),
        "avg_route_risk": round(result.avg_route_risk, 4),
        "avg_route_distance_km": round(result.avg_route_distance_km, 2),
        "avg_route_time_min": round(result.avg_route_time_min, 1),
        "runtime_s": round(result.runtime_s, 3),
        "shelter_utilization": {k: round(v, 3) for k, v in result.shelter_utilization.items()},
    }

    with open(out / "optimization_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info(f"Optimization summary saved to {out}/optimization_summary.json")
    return summary


def _load_viz_extras(config, villages, shelters, G=None) -> tuple:
    """
    Load geometries, node_coords, admin context, and hazard scores for visualization.
    Returns (village_geoms, shelter_geoms, node_coords, village_admin_ctx,
             shelter_admin_ctx, hazard_scores).
    """
    from shapely.wkt import loads as _wkt_loads
    from src.data.models import RegionOfInterest, RegionType

    region = RegionOfInterest(
        region_type=RegionType(config.region.region_type),
        bbox=tuple(config.region.bbox) if config.region.bbox else None,
        center=tuple(config.region.center) if config.region.center else None,
        radius_km=config.region.radius_km,
    )

    # Geometries straight from model objects
    village_geoms = {}
    for v in villages:
        if v.geometry_wkt:
            try:
                village_geoms[v.village_id] = _wkt_loads(v.geometry_wkt)
            except Exception:
                pass

    shelter_geoms = {}
    for s in shelters:
        if s.geometry_wkt:
            try:
                shelter_geoms[s.shelter_id] = _wkt_loads(s.geometry_wkt)
            except Exception:
                pass

    # Node coords: read directly from graph G
    node_coords = {}
    if G is not None:
        for n, d in G.nodes(data=True):
            if "lat" in d and "lon" in d:
                node_coords[n] = (d["lat"], d["lon"])

    if not node_coords:
        cache_dir = Path(config.extraction.osm_cache_dir)
        npath = (Path(config.preloaded_network_json)
                 if config.preloaded_network_json else None)
        if npath is None:
            candidates = sorted(cache_dir.glob("network_*.json"), key=lambda p: p.stat().st_mtime)
            npath = candidates[-1] if candidates else None
        if npath and npath.exists():
            try:
                with open(npath) as f:
                    nd = json.load(f)
                for n in nd.get("nodes", []):
                    node_coords[n["id"]] = (n["lat"], n["lon"])
            except Exception:
                pass

    # Admin context: spatial join clusters/shelters → L9 kelurahan → L8 kecamatan
    village_admin_ctx = {}
    shelter_admin_ctx = {}
    try:
        from src.data.wilayah_loader import WilayahLoader
        from experiments.preview_region import build_cluster_context, _shelter_admin_context
        region_bbox = region.to_bbox()
        with WilayahLoader() as wloader:
            l8_villages = wloader.load_villages(bbox=region_bbox, admin_levels=[8])
            l9_villages = wloader.load_villages(bbox=region_bbox, admin_levels=[9])
        if l9_villages:
            village_admin_ctx = build_cluster_context(villages, l9_villages, l8_villages)
            shelter_admin_ctx = _shelter_admin_context(shelters, l9_villages, l8_villages)
    except Exception as e:
        logging.getLogger(__name__).debug(f"Admin context unavailable: {e}")

    # Hazard scores: load from hazard_grid_cache.json, clip to scenario bbox
    hazard_scores = {}
    try:
        cache_path = Path(config.extraction.inarisk_cache_dir) / "hazard_grid_cache.json"
        if cache_path.exists():
            with open(cache_path) as f:
                full_cache = json.load(f)
            south, west, north, east = region.to_bbox()
            # Determine which hazard layers to show
            if config.routing.hazard_layers:
                layers = list(config.routing.hazard_layers.keys())
            else:
                layers = [config.disaster.disaster_type]
            for htype in layers:
                if htype in full_cache:
                    hazard_scores[htype] = {
                        k: v for k, v in full_cache[htype].items()
                        if south <= float(k.split(",")[0]) <= north
                        and west  <= float(k.split(",")[1]) <= east
                    }
    except Exception as e:
        logging.getLogger(__name__).debug(f"Hazard scores unavailable: {e}")

    return (village_geoms, shelter_geoms, node_coords,
            village_admin_ctx, shelter_admin_ctx, hazard_scores)


def save_graph_stats(G, output_dir: str, disaster_type: str):
    """Save graph summary and edge risk distribution to graph_stats.json."""
    import json
    import statistics
    from pathlib import Path

    risks   = [d.get("risk", 0.0) for _, _, d in G.edges(data=True)]
    weights = [d.get("weight", 0.0) for _, _, d in G.edges(data=True)]

    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "disaster_type": disaster_type,
        "edge_risk": {
            "min":    round(min(risks), 4) if risks else 0,
            "max":    round(max(risks), 4) if risks else 0,
            "mean":   round(statistics.mean(risks), 4) if risks else 0,
            "median": round(statistics.median(risks), 4) if risks else 0,
        },
        "edge_weight": {
            "min":  round(min(weights), 4) if weights else 0,
            "max":  round(max(weights), 4) if weights else 0,
            "mean": round(statistics.mean(weights), 4) if weights else 0,
        },
        "risk_distribution": {
            "zero (0.0)":          sum(1 for r in risks if r == 0.0),
            "very_low (0–0.2)":    sum(1 for r in risks if 0.0 < r <= 0.2),
            "low (0.2–0.4)":       sum(1 for r in risks if 0.2 < r <= 0.4),
            "medium (0.4–0.6)":    sum(1 for r in risks if 0.4 < r <= 0.6),
            "high (0.6–0.8)":      sum(1 for r in risks if 0.6 < r <= 0.8),
            "very_high (0.8–1.0)": sum(1 for r in risks if 0.8 < r <= 1.0),
        },
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logging.info(f"Graph stats → {out}/graph_stats.json")
    return stats


def save_routes(routes_by_village, villages, shelters, output_dir: str):
    """Save all candidate routes to routes.csv and routes_summary.json."""
    import csv

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    v_map = {v.village_id: v for v in villages}
    s_map = {s.shelter_id: s for s in shelters}

    rows = []
    for vid, routes in routes_by_village.items():
        v = v_map.get(vid)
        for rank, route in enumerate(routes):
            s = s_map.get(route.shelter_id)
            rows.append({
                "village_id":       vid,
                "village_name":     v.name if v else "",
                "population":       v.population if v else 0,
                "rank":             rank + 1,
                "shelter_id":       route.shelter_id,
                "shelter_name":     s.name if s else "",
                "shelter_capacity": s.capacity if s else 0,
                "distance_km":      round(route.total_distance_km, 3),
                "travel_time_min":  round(route.total_time_min, 1),
                "avg_risk":         round(route.avg_risk_score, 4),
                "composite_score":  round(route.composite_score, 4),
                "n_nodes":          len(route.node_path) if route.node_path else 0,
            })

    csv_path = out / "routes.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    logging.info(f"Routes CSV → {csv_path}  ({len(rows)} routes)")

    json_path = out / "routes_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "total_villages": len(routes_by_village),
            "total_routes":   len(rows),
            "routes":         rows,
        }, f, indent=2)
    logging.info(f"Routes JSON → {json_path}")
    return rows


def main():
    args = parse_args()

    # Load config
    from src.config.config_loader import load_config
    config = load_config(args.config)

    # Override output dir
    if args.output_dir:
        config.output_dir = args.output_dir

    setup_logging(args.log_level or config.log_level)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(" RespondOR-EvacuationRoute v1.0")
    logger.info(f" Scenario: {config.scenario_id}")
    logger.info(f" Disaster: {config.disaster.disaster_type} @ "
                f"({config.disaster.lat}, {config.disaster.lon})")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # OPTIMIZATION
    # ------------------------------------------------------------------ #
    result, villages, shelters, routes_by_village, timings, G = run_optimization(
        config,
        mode_override=args.mode,
        workers_override=args.workers,
        assignment_method_override=args.assignment_method,
        village_limit_override=args.village_limit,
    )

    # ------------------------------------------------------------------ #
    # STAGE OUTPUTS
    # ------------------------------------------------------------------ #
    # 1. Graph stats
    save_graph_stats(G, config.output_dir, config.disaster.disaster_type)

    # 2. Routes
    save_routes(routes_by_village, villages, shelters, config.output_dir)

    # 3. Assignment summary
    summary = save_optimization_result(
        result, villages, shelters, routes_by_village, config.output_dir
    )

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVACUATION SUMMARY")
    logger.info(f"  Total population:  {result.total_population:,}")
    logger.info(f"  Evacuated:         {result.total_evacuated:,} ({100*result.evacuation_ratio:.1f}%)")
    logger.info(f"  Unmet demand:      {result.total_unmet:,}")
    logger.info(f"  Avg route risk:    {result.avg_route_risk:.3f}")
    logger.info(f"  Avg distance:      {result.avg_route_distance_km:.1f} km")
    logger.info(f"  Avg travel time:   {result.avg_route_time_min:.0f} min")
    logger.info(f"  Runtime:           {result.runtime_s:.2f}s")
    logger.info("=" * 50)

    # ------------------------------------------------------------------ #
    # VISUALIZATION
    # ------------------------------------------------------------------ #
    from src.visualization.visualizer import EvacuationVisualizer
    viz = EvacuationVisualizer(config.output_dir)

    (village_geoms, shelter_geoms, node_coords,
     village_admin_ctx, shelter_admin_ctx, hazard_scores) = _load_viz_extras(
        config, villages, shelters, G=G
    )

    # Evacuation map — routes, villages, shelters
    viz.create_interactive_map(
        villages=villages,
        shelters=shelters,
        routes_by_village=routes_by_village,
        disaster_location=(config.disaster.lat, config.disaster.lon),
        disaster_type=config.disaster.disaster_type,
        village_geoms=village_geoms,
        shelter_geoms=shelter_geoms,
        node_coords=node_coords,
        disaster_name=config.disaster.name,
        region_radius_km=config.region.radius_km,
        village_admin_ctx=village_admin_ctx,
        shelter_admin_ctx=shelter_admin_ctx,
        hazard_scores=hazard_scores,
    )

    # Summary chart + assignment CSV
    viz.create_evacuation_summary_chart(result)
    viz.export_result_csv(result, villages, shelters)

    logger.info("Done.")
    logger.info(f"  Outputs in: {config.output_dir}/")
    logger.info(f"    graph_stats.json        — graph edge risk distribution")
    logger.info(f"    routes.csv              — all candidate routes ranked")
    logger.info(f"    routes_summary.json     — routes in JSON")
    logger.info(f"    optimization_summary.json — assignment KPIs")
    logger.info(f"    evacuation_results.csv  — per-village assignment")
    logger.info(f"    evacuation_map.html     — interactive routes + assignment map")
    logger.info(f"    evacuation_summary.png  — coverage chart")


if __name__ == "__main__":
    main()
