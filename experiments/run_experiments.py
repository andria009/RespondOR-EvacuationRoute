"""
Experimental evaluation script.
Runs all benchmark scenarios and produces comparison tables and charts.

Usage:
  python experiments/run_experiments.py --config configs/benchmark_scenarios.yaml
  python experiments/run_experiments.py --quick  # Small subset for testing
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def run_scenario_benchmark(
    scenario_cfg: dict,
    base_extraction: dict,
    base_routing: dict,
    output_dir: str,
    n_repeats: int = 3,
) -> List[dict]:
    """
    Run a single scenario in all specified modes and collect results.
    Returns list of result dicts.
    """
    from src.config.config_loader import AppConfig, DisasterConfig, RegionConfig
    from src.config.config_loader import ExtractionConfig, RoutingConfig, ExecutionConfig
    from src.data.models import ExecutionMode

    scenario_id = scenario_cfg["scenario_id"]
    disaster_raw = scenario_cfg["disaster"]
    region_raw = scenario_cfg["region"]
    exec_raw = scenario_cfg.get("execution", {})

    results = []

    modes = exec_raw.get("modes", ["naive", "parallel"])
    worker_counts = exec_raw.get("workers", [4])

    for mode in modes:
        workers_for_mode = worker_counts if mode in ("parallel", "hpc") else [1]

        for n_workers in workers_for_mode:
            for repeat in range(n_repeats):
                run_id = f"{scenario_id}_{mode}_w{n_workers}_r{repeat}"
                logger.info(f"--- Running: {run_id} ---")

                config = AppConfig(
                    scenario_id=run_id,
                    output_dir=f"{output_dir}/{run_id}",
                    disaster=DisasterConfig(
                        name=disaster_raw.get("name", "event"),
                        lat=float(disaster_raw["lat"]),
                        lon=float(disaster_raw["lon"]),
                        disaster_type=disaster_raw["type"],
                    ),
                    region=RegionConfig(
                        region_type=region_raw.get("type", "circle"),
                        center=[float(disaster_raw["lat"]), float(disaster_raw["lon"])],
                        radius_km=float(region_raw.get("radius_km", 20.0)),
                    ),
                    extraction=ExtractionConfig(
                        osm_cache_dir=base_extraction.get("osm_cache_dir", "data/raw/osm_cache"),
                        use_cached_osm=True,
                        inarisk_batch_size=base_extraction.get("inarisk_batch_size", 20),
                    ),
                    routing=RoutingConfig(
                        weight_distance=base_routing.get("weight_distance", 0.25),
                        weight_risk=base_routing.get("weight_risk", 0.45),
                        weight_road_quality=base_routing.get("weight_road_quality", 0.20),
                        weight_time=base_routing.get("weight_time", 0.10),
                    ),
                    execution=ExecutionConfig(
                        mode=mode,
                        n_workers=n_workers,
                    ),
                )

                try:
                    from src.benchmark.benchmark_runner import BenchmarkRunner
                    if mode == "naive":
                        from src.hpc.naive_runner import NaiveRunner
                        import time as tmod
                        import tracemalloc
                        tracemalloc.start()
                        t0 = tmod.perf_counter()
                        runner = NaiveRunner(config)
                        result, villages, shelters, _, timings = runner.run()
                        wall_time = tmod.perf_counter() - t0
                        _, peak_mem = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                    elif mode == "parallel":
                        from src.hpc.parallel_runner import ParallelRunner
                        import time as tmod
                        import tracemalloc
                        tracemalloc.start()
                        t0 = tmod.perf_counter()
                        runner = ParallelRunner(config, n_workers=n_workers)
                        result, villages, shelters, _, timings = runner.run()
                        wall_time = tmod.perf_counter() - t0
                        _, peak_mem = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                    else:
                        from src.hpc.distributed_runner import DistributedRunner
                        import time as tmod
                        import tracemalloc
                        tracemalloc.start()
                        t0 = tmod.perf_counter()
                        runner = DistributedRunner(config)
                        result, villages, shelters, _, timings = runner.run()
                        wall_time = tmod.perf_counter() - t0
                        _, peak_mem = tracemalloc.get_traced_memory()
                        tracemalloc.stop()

                    results.append({
                        "run_id": run_id,
                        "scenario": scenario_id,
                        "mode": mode,
                        "n_workers": n_workers,
                        "repeat": repeat,
                        "wall_time_s": round(wall_time, 3),
                        "peak_mem_mb": round(peak_mem / 1024 / 1024, 1),
                        "n_villages": len(villages),
                        "n_shelters": len(shelters),
                        "total_population": result.total_population,
                        "total_evacuated": result.total_evacuated,
                        "evacuation_ratio": round(result.evacuation_ratio, 4),
                        "avg_risk": round(result.avg_route_risk, 4),
                        "avg_dist_km": round(result.avg_route_distance_km, 2),
                        "avg_time_min": round(result.avg_route_time_min, 1),
                        **{f"time_{k}": round(v, 3) for k, v in timings.items()},
                    })
                    logger.info(f"  ✓ {run_id}: {wall_time:.2f}s, "
                                f"evacuated={result.total_evacuated}/{result.total_population}")

                except Exception as e:
                    logger.error(f"  ✗ {run_id} FAILED: {e}")
                    results.append({
                        "run_id": run_id, "scenario": scenario_id,
                        "mode": mode, "n_workers": n_workers,
                        "repeat": repeat, "error": str(e),
                    })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/benchmark_scenarios.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="Run only first scenario, single repeat")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        import json
        with open(args.config) as f:
            cfg = json.load(f)

    scenarios = cfg["scenarios"]
    if args.quick:
        scenarios = scenarios[:1]

    base_extraction = cfg.get("base_extraction", {})
    base_routing = cfg.get("base_routing", {})
    output_dir = args.output or cfg.get("output_dir", "output/experiments")
    n_repeats = 1 if args.quick else cfg.get("n_repeats", 3)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_results = []
    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario: {scenario['scenario_id']}")
        logger.info("=" * 60)
        results = run_scenario_benchmark(
            scenario, base_extraction, base_routing,
            output_dir, n_repeats
        )
        all_results.extend(results)

    # Save all results
    results_path = Path(output_dir) / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nAll results saved to {results_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Run ID':<40} {'Mode':<10} {'Workers':>8} {'Wall(s)':>10} "
          f"{'Evacuated%':>12} {'Speedup':>8}")
    print("-" * 90)

    # Compute speedups
    naive_times = {}
    for r in all_results:
        if r.get("mode") == "naive" and "wall_time_s" in r:
            key = (r["scenario"], r.get("repeat", 0))
            naive_times[key] = r["wall_time_s"]

    for r in all_results:
        if "error" in r:
            continue
        speedup = 1.0
        key = (r["scenario"], r.get("repeat", 0))
        if r.get("mode") != "naive" and key in naive_times and r["wall_time_s"] > 0:
            speedup = naive_times[key] / r["wall_time_s"]

        ratio = r.get("evacuation_ratio", 0) * 100
        print(f"{r['run_id']:<40} {r['mode']:<10} {r['n_workers']:>8} "
              f"{r.get('wall_time_s', 0):>10.2f} "
              f"{ratio:>11.1f}% {speedup:>8.2f}x")

    print("=" * 90)

    # Generate visualization
    try:
        _generate_experiment_charts(all_results, output_dir)
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")


def _generate_experiment_charts(results: List[dict], output_dir: str):
    """Generate speedup and scaling charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Group by scenario
    scenarios = list(set(r["scenario"] for r in results if "error" not in r))

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5))
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        scen_results = [r for r in results if r.get("scenario") == scenario and "error" not in r]

        # Get naive baseline (mean of repeats)
        naive = [r for r in scen_results if r.get("mode") == "naive"]
        if not naive:
            continue
        naive_time = np.mean([r["wall_time_s"] for r in naive])

        # Parallel speedups
        par_data = {}
        for r in scen_results:
            if r.get("mode") == "parallel":
                w = r["n_workers"]
                par_data.setdefault(w, []).append(r["wall_time_s"])

        workers = sorted(par_data.keys())
        speedups = [naive_time / np.mean(par_data[w]) for w in workers]
        ideal = workers

        ax.plot(workers, speedups, "bo-", label="Actual speedup", linewidth=2, markersize=8)
        ax.plot(workers, ideal, "g--", label="Ideal (linear)", alpha=0.7)
        ax.set_xlabel("Number of Workers")
        ax.set_ylabel("Speedup")
        ax.set_title(f"Speedup: {scenario}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    chart_path = Path(output_dir) / "speedup_chart.png"
    plt.savefig(str(chart_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Speedup chart saved to {chart_path}")


if __name__ == "__main__":
    main()
