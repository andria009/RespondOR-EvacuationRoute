"""
Benchmark runner.
Compares naive, parallel, and HPC execution modes on identical scenarios.
Measures: wall time, CPU time, peak memory, speedup, efficiency.
"""

import gc
import json
import time
import tracemalloc
import logging
from pathlib import Path
from typing import List, Dict, Optional

from src.config.config_loader import AppConfig
from src.data.models import BenchmarkResult, ExecutionMode

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Executes the same scenario under naive, parallel, and HPC modes
    and records computational performance metrics.
    """

    def __init__(self, config: AppConfig, output_dir: Optional[str] = None):
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir) / "benchmark"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def run_all(self, run_hpc: bool = False) -> List[BenchmarkResult]:
        """
        Run benchmark for naive, then parallel, optionally HPC.
        Returns list of BenchmarkResult.
        """
        naive_result = self.run_naive()
        parallel_result = self.run_parallel()

        results = [naive_result, parallel_result]
        if run_hpc:
            hpc_result = self.run_hpc()
            results.append(hpc_result)

        self._compute_speedup(results)
        self._save_results(results)
        self._print_report(results)
        return results

    def run_naive(self) -> BenchmarkResult:
        """Benchmark naive sequential mode."""
        from src.hpc.naive_runner import NaiveRunner
        logger.info("Benchmarking NAIVE mode...")

        gc.collect()
        tracemalloc.start()
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        runner = NaiveRunner(self.config)
        opt_result, villages, shelters, routes_by_village, timings = runner.run()

        t_wall = time.perf_counter() - t_wall_start
        t_cpu = time.process_time() - t_cpu_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            scenario_id=self.config.scenario_id,
            mode=ExecutionMode.NAIVE,
            phase="full_pipeline",
            n_workers=1,
            n_nodes=self._count_nodes(timings),
            n_edges=0,
            n_villages=len(villages),
            n_shelters=len(shelters),
            wall_time_s=t_wall,
            cpu_time_s=t_cpu,
            peak_memory_mb=peak_mem / 1024 / 1024,
            speedup=1.0,
            efficiency=1.0,
        )

    def run_parallel(self) -> BenchmarkResult:
        """Benchmark parallel single-machine mode."""
        from src.hpc.parallel_runner import ParallelRunner
        n_workers = self.config.execution.n_workers
        logger.info(f"Benchmarking PARALLEL mode ({n_workers} workers)...")

        gc.collect()
        tracemalloc.start()
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        runner = ParallelRunner(self.config)
        opt_result, villages, shelters, routes_by_village, timings = runner.run()

        t_wall = time.perf_counter() - t_wall_start
        t_cpu = time.process_time() - t_cpu_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            scenario_id=self.config.scenario_id,
            mode=ExecutionMode.PARALLEL,
            phase="full_pipeline",
            n_workers=n_workers,
            n_nodes=0, n_edges=0,
            n_villages=len(villages),
            n_shelters=len(shelters),
            wall_time_s=t_wall,
            cpu_time_s=t_cpu,
            peak_memory_mb=peak_mem / 1024 / 1024,
        )

    def run_hpc(self) -> BenchmarkResult:
        """Benchmark HPC/distributed mode."""
        from src.hpc.distributed_runner import DistributedRunner
        logger.info("Benchmarking HPC mode...")

        gc.collect()
        tracemalloc.start()
        t_wall_start = time.perf_counter()
        t_cpu_start = time.process_time()

        runner = DistributedRunner(self.config)
        opt_result, villages, shelters, routes_by_village, timings = runner.run()

        t_wall = time.perf_counter() - t_wall_start
        t_cpu = time.process_time() - t_cpu_start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            scenario_id=self.config.scenario_id,
            mode=ExecutionMode.HPC,
            phase="full_pipeline",
            n_workers=self.config.execution.n_workers,
            n_nodes=0, n_edges=0,
            n_villages=len(villages),
            n_shelters=len(shelters),
            wall_time_s=t_wall,
            cpu_time_s=t_cpu,
            peak_memory_mb=peak_mem / 1024 / 1024,
        )

    def run_phase_benchmarks(self) -> Dict[str, List[BenchmarkResult]]:
        """
        Run per-phase benchmarks for routing stage only.
        Useful for fine-grained analysis.
        """
        from src.hpc.naive_runner import NaiveRunner
        import networkx as nx
        from src.data.models import DisasterType, RegionType, RegionOfInterest, DisasterInput

        cfg = self.config
        runner = NaiveRunner(cfg)

        disaster = DisasterInput(
            location=(cfg.disaster.lat, cfg.disaster.lon),
            disaster_type=DisasterType(cfg.disaster.disaster_type),
            name=cfg.disaster.name,
        )
        region = RegionOfInterest(
            region_type=RegionType(cfg.region.region_type),
            center=tuple(cfg.region.center) if cfg.region.center else None,
            radius_km=cfg.region.radius_km,
        )

        # Extract once
        nodes, edges, villages, shelters = runner._extract_data(region, cfg)
        builder = EvacuationGraphBuilder()
        G = builder.build(nodes, edges)
        builder.attach_pois_to_graph(villages, shelters)

        phase_results: Dict[str, List[BenchmarkResult]] = {
            "routing_naive": [],
            "routing_parallel": [],
        }

        for n_villages in [10, 25, 50, len(villages)]:
            subset = villages[:n_villages]

            # Naive routing
            t_start = time.perf_counter()
            from src.routing.heuristic_optimizer import HeuristicOptimizer
            opt = HeuristicOptimizer()
            opt.compute_routes(G, subset, shelters, mode=ExecutionMode.NAIVE)
            t_naive = time.perf_counter() - t_start

            phase_results["routing_naive"].append(BenchmarkResult(
                scenario_id=f"{cfg.scenario_id}_nv{n_villages}",
                mode=ExecutionMode.NAIVE, phase="routing",
                n_workers=1, n_nodes=G.number_of_nodes(),
                n_edges=G.number_of_edges(),
                n_villages=n_villages, n_shelters=len(shelters),
                wall_time_s=t_naive, cpu_time_s=t_naive,
                peak_memory_mb=0,
            ))

            # Parallel routing
            for n_workers in [2, 4, 8]:
                t_start = time.perf_counter()
                opt.compute_routes(G, subset, shelters,
                                   mode=ExecutionMode.PARALLEL,
                                   n_workers=n_workers)
                t_par = time.perf_counter() - t_start

                phase_results["routing_parallel"].append(BenchmarkResult(
                    scenario_id=f"{cfg.scenario_id}_nv{n_villages}_w{n_workers}",
                    mode=ExecutionMode.PARALLEL, phase="routing",
                    n_workers=n_workers, n_nodes=G.number_of_nodes(),
                    n_edges=G.number_of_edges(),
                    n_villages=n_villages, n_shelters=len(shelters),
                    wall_time_s=t_par, cpu_time_s=t_par,
                    peak_memory_mb=0,
                    speedup=t_naive / t_par if t_par > 0 else 1.0,
                    efficiency=(t_naive / t_par) / n_workers if t_par > 0 else 1.0,
                ))

        return phase_results

    def _compute_speedup(self, results: List[BenchmarkResult]):
        """Compute speedup relative to naive baseline."""
        naive_time = next(
            (r.wall_time_s for r in results if r.mode == ExecutionMode.NAIVE), None
        )
        if naive_time is None or naive_time <= 0:
            return
        for r in results:
            if r.mode != ExecutionMode.NAIVE and r.wall_time_s > 0:
                r.speedup = naive_time / r.wall_time_s
                r.efficiency = r.speedup / r.n_workers if r.n_workers > 0 else r.speedup

    def _save_results(self, results: List[BenchmarkResult]):
        out = []
        for r in results:
            out.append({
                "scenario_id": r.scenario_id,
                "mode": r.mode.value,
                "phase": r.phase,
                "n_workers": r.n_workers,
                "n_villages": r.n_villages,
                "n_shelters": r.n_shelters,
                "wall_time_s": round(r.wall_time_s, 3),
                "cpu_time_s": round(r.cpu_time_s, 3),
                "peak_memory_mb": round(r.peak_memory_mb, 1),
                "speedup": round(r.speedup, 2),
                "efficiency": round(r.efficiency, 2),
            })
        path = self.output_dir / "benchmark_results.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Benchmark results saved to {path}")

    def _print_report(self, results: List[BenchmarkResult]):
        print("\n" + "=" * 70)
        print(f"{'Mode':<12} {'Workers':>8} {'Wall(s)':>10} {'CPU(s)':>10} "
              f"{'Mem(MB)':>10} {'Speedup':>8} {'Eff':>6}")
        print("-" * 70)
        for r in results:
            print(f"{r.mode.value:<12} {r.n_workers:>8} {r.wall_time_s:>10.2f} "
                  f"{r.cpu_time_s:>10.2f} {r.peak_memory_mb:>10.1f} "
                  f"{r.speedup:>8.2f} {r.efficiency:>6.2f}")
        print("=" * 70 + "\n")

    def _count_nodes(self, timings: dict) -> int:
        return 0  # Filled by runner metadata
