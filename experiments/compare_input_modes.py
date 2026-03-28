"""
Input Mode Comparison Benchmark
================================
Compares two input modes for the same Merapi disaster scenario:

  Mode A — OSM:    live OSM extraction via osmnx (uses cached data)
  Mode B — PYCGR:  pre-extracted PYCGR + POI CSV (RespondOR v1 format)

For each input mode, runs three execution modes:
  - naive      (sequential, 1 process)
  - parallel-4 (4 worker processes)
  - parallel-8 (8 worker processes)

Total: 6 benchmark runs.

RESOURCE USAGE AND SAFETY CONSTRAINT
--------------------------------------
- benchmark_village_limit = 500  (see configs)  caps routing villages,
  keeping each run under ~10 min on a standard laptop.
- InaRISK API calls are skipped if risk data is already cached.
- OSM network is loaded from local cache — no new HTTP downloads.

Usage:
    python -m experiments.compare_input_modes

Output:
    output/benchmark/comparison_results.json
    output/benchmark/comparison_report.txt
    output/benchmark/comparison_chart.png
"""

import gc
import json
import logging
import sys
import time
import tracemalloc
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

OUT_DIR = Path("output/benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYCGR_EXPORT_DIR = Path("data/processed/merapi_pycgr")


# ------------------------------------------------------------------ #
# Step 1 — Export Merapi OSM data to PYCGR + POI CSV (one-time)
# ------------------------------------------------------------------ #

def export_merapi_to_pycgr() -> bool:
    """
    Export the cached Merapi OSM data (nodes/edges/villages/shelters)
    to PYCGR network + POI CSV for use in the legacy input mode benchmark.

    Returns True if export was performed, False if already exists.
    """
    net_path = PYCGR_EXPORT_DIR / "merapi_network.pycgr"
    poi_path = PYCGR_EXPORT_DIR / "merapi_pois.csv"

    if net_path.exists() and poi_path.exists():
        logger.warning(f"Export files already exist — skipping export")
        return False

    logger.warning("Exporting Merapi OSM cache to PYCGR + POI CSV ...")
    PYCGR_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    from src.config.config_loader import load_config
    from src.data.osm_extractor import OSMExtractor
    from src.data.models import RegionOfInterest, RegionType
    from src.data.population_loader import PopulationLoader, ShelterCapacityLoader

    cfg = load_config("configs/merapi_benchmark_osm.yaml")
    extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)
    region = RegionOfInterest(
        region_type=RegionType(cfg.region.region_type),
        center=tuple(cfg.region.center),
        radius_km=cfg.region.radius_km,
    )

    nodes, edges = extractor.extract_road_network(region, use_cache=True)
    villages = extractor.extract_villages(region, use_cache=True)
    shelters = extractor.extract_shelters(region, use_cache=True)

    PopulationLoader().apply_population(villages, density_per_km2=cfg.extraction.default_pop_density)
    ShelterCapacityLoader().apply_capacity(shelters)

    extractor.export_to_pycgr(nodes, edges, str(net_path))
    extractor.export_pois_to_csv(villages, shelters, str(poi_path))

    logger.warning(
        f"Exported: {len(nodes)} nodes, {len(edges)} edges → {net_path}\n"
        f"          {len(villages)} villages, {len(shelters)} shelters → {poi_path}"
    )
    return True


# ------------------------------------------------------------------ #
# Step 2 — Run a single benchmark configuration
# ------------------------------------------------------------------ #

def run_one(config_path: str, n_workers: int, label: str) -> dict:
    """
    Run the full pipeline for one (input_mode × execution_mode) combination.
    Returns timing dict with stage breakdowns and summary metrics.
    """
    from src.config.config_loader import load_config
    from src.hpc.naive_runner import NaiveRunner
    from src.hpc.parallel_runner import ParallelRunner

    cfg = load_config(config_path)
    cfg.execution.n_workers = n_workers

    gc.collect()
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    logger.warning(f"Starting: {label} ...")

    if n_workers <= 1:
        runner = NaiveRunner(cfg)
    else:
        runner = ParallelRunner(cfg, n_workers=n_workers)

    result, villages, shelters, routes_by_village, timings = runner.run()

    wall_time = time.perf_counter() - wall_start
    cpu_time = time.process_time() - cpu_start
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_routes = sum(len(v) for v in routes_by_village.values())

    return {
        "label": label,
        "config": config_path,
        "n_workers": n_workers,
        "wall_time_s": round(wall_time, 2),
        "cpu_time_s": round(cpu_time, 2),
        "peak_memory_mb": round(peak_mem / 1e6, 1),
        "stage_timings": {k: round(v, 3) for k, v in timings.items()},
        "n_villages": len(villages),
        "n_shelters": len(shelters),
        "n_routes": n_routes,
        "total_population": result.total_population,
        "total_evacuated": result.total_evacuated,
        "evacuation_ratio": round(result.evacuation_ratio, 4),
    }


# ------------------------------------------------------------------ #
# Step 3 — Run all 6 combinations
# ------------------------------------------------------------------ #

RUNS = [
    # (config_path,                         n_workers, label)
    ("configs/merapi_benchmark_osm.yaml",   1,  "OSM    / naive      "),
    ("configs/merapi_benchmark_osm.yaml",   4,  "OSM    / parallel-4 "),
    ("configs/merapi_benchmark_osm.yaml",   8,  "OSM    / parallel-8 "),
    ("configs/merapi_benchmark_pycgr.yaml", 1,  "PYCGR  / naive      "),
    ("configs/merapi_benchmark_pycgr.yaml", 4,  "PYCGR  / parallel-4 "),
    ("configs/merapi_benchmark_pycgr.yaml", 8,  "PYCGR  / parallel-8 "),
]


def run_all_benchmarks() -> list:
    results = []
    for config_path, n_workers, label in RUNS:
        try:
            r = run_one(config_path, n_workers, label)
            results.append(r)
            logger.warning(
                f"  Done {label}: wall={r['wall_time_s']:.1f}s  "
                f"evacuation={100*r['evacuation_ratio']:.1f}%"
            )
        except Exception as e:
            logger.error(f"  FAILED {label}: {e}", exc_info=True)
            results.append({"label": label, "error": str(e)})
        gc.collect()
    return results


# ------------------------------------------------------------------ #
# Step 4 — Report and chart
# ------------------------------------------------------------------ #

def print_report(results: list):
    lines = []
    lines.append("=" * 76)
    lines.append("  RespondOR Input Mode × Execution Mode Benchmark")
    lines.append("  Scenario: Merapi Eruption 2024 (500-village subset)")
    lines.append("=" * 76)
    lines.append(
        f"{'Label':<28} {'Wall(s)':>8} {'CPU(s)':>8} {'Mem(MB)':>8} "
        f"{'Extract(s)':>10} {'Route(s)':>9} {'Evacuated':>10}"
    )
    lines.append("-" * 76)

    for r in results:
        if "error" in r:
            lines.append(f"{r['label']:<28}  ERROR: {r['error']}")
            continue
        st = r.get("stage_timings", {})
        lines.append(
            f"{r['label']:<28} "
            f"{r['wall_time_s']:>8.1f} "
            f"{r['cpu_time_s']:>8.1f} "
            f"{r['peak_memory_mb']:>8.1f} "
            f"{st.get('extraction', 0):>10.2f} "
            f"{st.get('routing', 0):>9.2f} "
            f"{100*r['evacuation_ratio']:>9.1f}%"
        )

    # Speedup vs naive baselines
    lines.append("")
    lines.append("  Speedup (routing stage only, vs each input mode's naive baseline)")
    lines.append("-" * 76)
    baselines = {}
    for r in results:
        if "error" not in r and r["n_workers"] == 1:
            key = "osm" if "osm" in r["config"] else "pycgr"
            baselines[key] = r["stage_timings"].get("routing", 1)

    for r in results:
        if "error" in r:
            continue
        key = "osm" if "osm" in r["config"] else "pycgr"
        base = baselines.get(key, 1)
        route_t = r["stage_timings"].get("routing", 0)
        speedup = base / route_t if route_t > 0 else 0
        efficiency = speedup / r["n_workers"] if r["n_workers"] > 0 else 0
        lines.append(
            f"  {r['label']:<28}  speedup={speedup:.2f}x  "
            f"efficiency={100*efficiency:.0f}%"
        )

    lines.append("=" * 76)
    report = "\n".join(lines)
    print(report)

    out_path = OUT_DIR / "comparison_report.txt"
    out_path.write_text(report)
    logger.warning(f"Report saved to {out_path}")


def save_chart(results: list):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        valid = [r for r in results if "error" not in r]
        if not valid:
            return

        labels = [r["label"].strip() for r in valid]
        wall_times = [r["wall_time_s"] for r in valid]
        route_times = [r["stage_timings"].get("routing", 0) for r in valid]
        extract_times = [r["stage_timings"].get("extraction", 0) for r in valid]

        x = np.arange(len(labels))
        width = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(
            "Input Mode × Execution Mode — Merapi Benchmark\n"
            "(500-village subset, RESOURCE-CONSTRAINED)"
        )

        # Left: wall time breakdown
        ax = axes[0]
        ax.bar(x - width, extract_times, width, label="Extraction", color="#4e79a7")
        ax.bar(x, route_times,   width, label="Routing",    color="#f28e2b")
        other = [r["stage_timings"].get("graph_build", 0)
                 + r["stage_timings"].get("risk_scoring", 0)
                 + r["stage_timings"].get("assignment", 0)
                 for r in valid]
        ax.bar(x + width, other, width, label="Other", color="#76b7b2")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Time (s)")
        ax.set_title("Stage Timings")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # Right: routing speedup
        ax = axes[1]
        colors = ["#4e79a7"] * 3 + ["#f28e2b"] * 3
        ax.bar(x, route_times, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Routing time (s)")
        ax.set_title("Routing Stage Comparison")
        ax.grid(axis="y", alpha=0.3)

        # Annotate bars with speedup
        baselines = {}
        for i, r in enumerate(valid):
            key = "osm" if "osm" in r["config"] else "pycgr"
            if r["n_workers"] == 1:
                baselines[key] = r["stage_timings"].get("routing", 1)
        for i, r in enumerate(valid):
            key = "osm" if "osm" in r["config"] else "pycgr"
            base = baselines.get(key, 1)
            rt = r["stage_timings"].get("routing", 0)
            sp = base / rt if rt > 0 else 0
            ax.text(i, rt + 1, f"{sp:.1f}x", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        chart_path = OUT_DIR / "comparison_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.warning(f"Chart saved to {chart_path}")

    except ImportError:
        logger.warning("matplotlib not available — skipping chart")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    logger.warning("=" * 60)
    logger.warning("RespondOR — Input Mode Comparison Benchmark")
    logger.warning("=" * 60)

    # Step 1: export Merapi OSM to PYCGR (one-time, skipped if done)
    logger.warning("\n[1/3] Exporting Merapi data to PYCGR format ...")
    export_merapi_to_pycgr()

    # Step 2: run all 6 benchmark combinations
    logger.warning("\n[2/3] Running 6 benchmark combinations ...")
    results = run_all_benchmarks()

    # Step 3: save results + report + chart
    logger.warning("\n[3/3] Saving results ...")
    results_path = OUT_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.warning(f"Raw results → {results_path}")

    print_report(results)
    save_chart(results)

    logger.warning("\nDone.")


if __name__ == "__main__":
    main()
