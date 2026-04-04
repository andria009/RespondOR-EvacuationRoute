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
import sys
from pathlib import Path


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
    return p.parse_args()


def run_optimization(config, mode_override=None, workers_override=None):
    """Run the optimization pipeline in the specified mode."""
    from src.data.models import ExecutionMode

    if mode_override:
        config.execution.mode = mode_override
    if workers_override:
        config.execution.n_workers = workers_override

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
    import json
    from pathlib import Path

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


def _load_viz_extras(config) -> tuple:
    """
    Load village/shelter polygon geometries and node_coords dict for rich visualization.
    Returns (village_geoms, shelter_geoms, node_coords) — dicts keyed by ID / node_id.
    Falls back gracefully if files are missing.
    """
    import json as _json
    from pathlib import Path as _Path
    from shapely.geometry import shape as _shape

    cache_dir = _Path(config.extraction.osm_cache_dir)

    def _latest(pattern):
        candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        return candidates[-1] if candidates else None

    def _load_geojson_geoms(path, id_field):
        geoms = {}
        try:
            with open(path) as f:
                for feat in _json.load(f).get("features", []):
                    fid = str(feat.get("properties", {}).get(id_field, ""))
                    g = feat.get("geometry")
                    if fid and g:
                        try:
                            geoms[fid] = _shape(g)
                        except Exception:
                            pass
        except Exception:
            pass
        return geoms

    vpath = (_Path(config.preloaded_villages_geojson)
             if config.preloaded_villages_geojson else _latest("villages_*.geojson"))
    spath = (_Path(config.preloaded_shelters_geojson)
             if config.preloaded_shelters_geojson else _latest("shelters_*.geojson"))
    npath = (_Path(config.preloaded_network_json)
             if config.preloaded_network_json else _latest("network_*.json"))

    village_geoms = _load_geojson_geoms(vpath, "village_id") if vpath and vpath.exists() else {}
    shelter_geoms = _load_geojson_geoms(spath, "shelter_id") if spath and spath.exists() else {}

    node_coords = {}
    if npath and npath.exists():
        try:
            with open(npath) as f:
                nd = _json.load(f)
            for n in nd.get("nodes", []):
                node_coords[n["id"]] = (n["lat"], n["lon"])
        except Exception:
            pass

    return village_geoms, shelter_geoms, node_coords


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
    result, villages, shelters, routes_by_village, timings = run_optimization(
        config,
        mode_override=args.mode,
        workers_override=args.workers,
    )

    # Save results
    summary = save_optimization_result(
        result, villages, shelters, routes_by_village, config.output_dir
    )

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info(f"EVACUATION SUMMARY")
    logger.info(f"  Total population:  {result.total_population:,}")
    logger.info(f"  Evacuated:         {result.total_evacuated:,} ({100*result.evacuation_ratio:.1f}%)")
    logger.info(f"  Unmet demand:      {result.total_unmet:,}")
    logger.info(f"  Avg route risk:    {result.avg_route_risk:.3f}")
    logger.info(f"  Avg distance:      {result.avg_route_distance_km:.1f} km")
    logger.info(f"  Avg travel time:   {result.avg_route_time_min:.0f} min")
    logger.info(f"  Runtime:           {result.runtime_s:.2f}s")
    logger.info("=" * 50)

    # Visualization
    from src.visualization.visualizer import EvacuationVisualizer
    viz = EvacuationVisualizer(config.output_dir)

    village_geoms, shelter_geoms, node_coords = _load_viz_extras(config)
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
    )
    viz.create_evacuation_summary_chart(result)
    viz.export_result_csv(result, villages, shelters)

    logger.info("Done.")
    logger.info(f"  Prepare GAMA inputs: python -m experiments.prepare_gama_inputs "
                f"--config {args.config}")


if __name__ == "__main__":
    main()
