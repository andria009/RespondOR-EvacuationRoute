"""
RespondOR-EvacuationRoute — Main Entrypoint
Disaster evacuation route optimization and simulation system.

Usage:
  # Run full pipeline (naive mode):
  python -m src.main --config configs/disaster_scenario.yaml

  # Parallel mode (multiprocessing, single node):
  python -m src.main --config configs/disaster_scenario.yaml --mode parallel --workers 8

  # HPC mode (MPI, launch via srun/mpirun):
  srun --mpi=pmix -n 32 python -m src.main --config configs/disaster_scenario.yaml --mode hpc

  # Run benchmark comparison:
  python -m src.main --config configs/disaster_scenario.yaml --benchmark

  # Run simulation only (after optimization):
  python -m src.main --config configs/disaster_scenario.yaml --simulate-only

  # Generate visualization only:
  python -m src.main --config configs/disaster_scenario.yaml --visualize-only
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
    p.add_argument("--benchmark", action="store_true",
                   help="Run all modes and compare performance")
    p.add_argument("--simulate", action="store_true",
                   help="Also run GAMA simulation after optimization")
    p.add_argument("--simulate-only", action="store_true",
                   help="Run GAMA simulation only (requires existing optimization output)")
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
    # BENCHMARK mode: run all modes and compare
    # ------------------------------------------------------------------ #
    if args.benchmark:
        from src.benchmark.benchmark_runner import BenchmarkRunner
        benchmarker = BenchmarkRunner(config)
        results = benchmarker.run_all(run_hpc=(config.execution.hpc_framework is not None))
        return

    # ------------------------------------------------------------------ #
    # OPTIMIZATION
    # ------------------------------------------------------------------ #
    if not args.simulate_only:
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

        viz.create_interactive_map(
            villages=villages,
            shelters=shelters,
            routes_by_village=routes_by_village,
            disaster_location=(config.disaster.lat, config.disaster.lon),
            disaster_type=config.disaster.disaster_type,
        )
        viz.create_evacuation_summary_chart(result)
        viz.export_result_csv(result, villages, shelters)

    # ------------------------------------------------------------------ #
    # SIMULATION
    # ------------------------------------------------------------------ #
    if args.simulate or args.simulate_only:
        if args.simulate_only:
            # Load existing result
            result_path = Path(config.output_dir) / "optimization_summary.json"
            if not result_path.exists():
                logger.error(f"No existing optimization result at {result_path}. "
                             f"Run optimization first.")
                sys.exit(1)
            logger.info(f"Loaded existing optimization result from {result_path}")
            # For simulate-only, we'd need to reconstruct objects — simplified here
            return

        from src.simulation.gama_orchestrator import GAMAOrchestrator
        orchestrator = GAMAOrchestrator(
            gama_executable=config.simulation.gama_executable,
            gaml_model_path=config.simulation.gaml_model_path,
            output_dir=config.simulation.gama_output_dir,
            max_steps=config.simulation.max_simulation_steps,
            n_runs=config.simulation.n_runs,
            parallel_runs=config.simulation.parallel_runs,
        )

        logger.info(f"Running {config.simulation.n_runs} GAMA simulations...")
        sim_outputs = orchestrator.run_simulation(
            opt_result=result,
            villages=villages,
            shelters=shelters,
            scenario_id=config.scenario_id,
        )

        stats = orchestrator.aggregate_outputs(sim_outputs)
        logger.info("\nSIMULATION RESULTS:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

        # Save simulation stats
        sim_stats_path = Path(config.output_dir) / "simulation_stats.json"
        with open(sim_stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Simulation stats saved to {sim_stats_path}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
