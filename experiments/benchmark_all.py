#!/usr/bin/env python3
"""
Benchmark all scenarios across execution modes.
Records per-phase timing and memory consumption for each run.

Execution matrix
----------------
  naive                                        — 1 run
  parallel  workers ∈ {2, 4, 8, 16, 32, 64}  — 6 runs
  hpc       ranks × workers ∈ {2,4} × {8,16,64} — 6 runs
  ─────────────────────────────────────────────────────
  Total: 13 runs × N scenarios

Each run writes a timings JSON to output/<scenario>/timings_*.json.
This script reads those files immediately after each run and aggregates
into output/benchmark_results.json (incremental — survives interruption).
A summary CSV is written at the end.

Usage
-----
  # All scenarios, all modes (may take hours on fresh caches)
  python -m experiments.benchmark_all

  # Specific scenarios only
  python -m experiments.benchmark_all --scenarios banjarnegara_landslide_2021 merapi_eruption_2023

  # Subset of modes
  python -m experiments.benchmark_all --modes naive parallel_4 parallel_8 hpc_2r_8w

  # Dry-run: print commands without executing
  python -m experiments.benchmark_all --dry-run

  # Skip runs already present in benchmark_results.json (resume)
  python -m experiments.benchmark_all --resume

  # Limit villages per scenario (useful for quick tests)
  python -m experiments.benchmark_all --benchmark-village-limit 100

Notes
-----
  - All scenarios should have cached OSM and InaRISK data before benchmarking.
    Benchmarking with live API calls inflates timings unpredictably.
  - MPI runs require mpi4py: pip install mpi4py
  - Memory tracking requires psutil: pip install psutil
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_all")

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR  = PROJECT_ROOT / "configs"
OUTPUT_DIR   = PROJECT_ROOT / "output"
RESULTS_JSON = OUTPUT_DIR / "benchmark_results.json"
RESULTS_CSV  = OUTPUT_DIR / "benchmark_results.csv"


# ------------------------------------------------------------------ #
# Mode definitions
# ------------------------------------------------------------------ #

def _build_mode_matrix(
    parallel_workers: list[int],
    hpc_ranks: list[int],
    hpc_workers: list[int],
) -> list[dict]:
    """Return list of mode descriptors."""
    modes = []

    # naive
    modes.append({
        "id":        "naive",
        "mode":      "naive",
        "n_workers": 1,
        "n_ranks":   1,
        "total_cores": 1,
        "cmd_extra": [],
        "timings_file": "timings_naive.json",
    })

    # parallel
    for w in parallel_workers:
        modes.append({
            "id":        f"parallel_{w}w",
            "mode":      "parallel",
            "n_workers": w,
            "n_ranks":   1,
            "total_cores": w,
            "cmd_extra": ["--mode", "parallel", "--workers", str(w)],
            "timings_file": f"timings_parallel_{w}w.json",
        })

    # hpc (MPI)
    for r in hpc_ranks:
        for w in hpc_workers:
            modes.append({
                "id":        f"hpc_{r}r_{w}w",
                "mode":      "hpc",
                "n_workers": w,
                "n_ranks":   r,
                "total_cores": r * w,
                "cmd_extra": ["--mode", "hpc", "--workers", str(w)],
                "mpi_ranks": r,
                "timings_file": f"timings_hpc_{r}r_{w}w.json",
            })

    return modes


# ------------------------------------------------------------------ #
# Runner
# ------------------------------------------------------------------ #

def _mpi_available() -> bool:
    try:
        import mpi4py  # noqa: F401
        result = subprocess.run(
            ["mpirun", "--version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def _build_command(
    scenario_config: Path, mode: dict, village_limit: Optional[int] = None,
    mpi_launcher: str = "mpirun",
) -> list[str]:
    """Build the subprocess command for a given mode."""
    cmd = []
    if mode.get("mpi_ranks", 1) > 1:
        if mpi_launcher == "srun":
            cmd += ["srun", "--mpi=pmix", "-n", str(mode["mpi_ranks"])]
        else:
            cmd += ["mpirun", "-n", str(mode["mpi_ranks"])]
    cmd += [
        sys.executable, "-m", "src.main",
        "--config", str(scenario_config),
    ]
    cmd += mode["cmd_extra"]
    if village_limit:
        cmd += ["--village-limit", str(village_limit)]
    return cmd


def _run_mode(
    scenario_config: Path,
    mode: dict,
    timeout_s: int = 3600,
    village_limit: Optional[int] = None,
    dry_run: bool = False,
    mpi_launcher: str = "mpirun",
) -> dict:
    """
    Execute one (scenario, mode) combination.
    Returns a result dict with timings + memory or an error entry.
    """
    cmd = _build_command(scenario_config, mode, village_limit, mpi_launcher=mpi_launcher)
    scenario_id = scenario_config.stem

    log.info(
        f"  [{mode['id']}] {scenario_id} — "
        f"{mode['total_cores']} core(s)"
        + (f" ({mpi_launcher} -n {mode['n_ranks']})" if mode.get("mpi_ranks", 1) > 1 else "")
    )
    log.info(f"    cmd: {' '.join(cmd)}")

    if dry_run:
        return {"dry_run": True}

    t_wall_start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"timeout after {timeout_s}s",
            "wall_time_s": timeout_s,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

    wall_time_s = time.perf_counter() - t_wall_start

    if proc.returncode != 0:
        # Extract last 20 lines of stderr for diagnostics
        err_lines = (proc.stderr or "").strip().splitlines()
        return {
            "success":    False,
            "returncode": proc.returncode,
            "error":      "\n".join(err_lines[-20:]),
            "wall_time_s": round(wall_time_s, 2),
        }

    # Read the timings JSON written by the runner
    timings_path = OUTPUT_DIR / scenario_id / mode["timings_file"]
    if not timings_path.exists():
        return {
            "success":    False,
            "error":      f"timings file not found: {timings_path}",
            "wall_time_s": round(wall_time_s, 2),
        }

    with open(timings_path) as f:
        timings_data = json.load(f)

    return {
        "success":      True,
        "wall_time_s":  round(wall_time_s, 2),
        "timings":      timings_data.get("timings", {}),
        "total_s":      timings_data.get("total", 0.0),
        "memory_mb":    timings_data.get("memory_mb", {}),
        "peak_rss_mb":  timings_data.get("peak_rss_mb", 0.0),
    }


# ------------------------------------------------------------------ #
# Result aggregation
# ------------------------------------------------------------------ #

def _load_results() -> list[dict]:
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON) as f:
            return json.load(f).get("results", [])
    return []


def _save_results(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump({"results": results, "updated": datetime.now().isoformat()}, f, indent=2)


def _result_key(scenario_id: str, mode_id: str) -> str:
    return f"{scenario_id}__{mode_id}"


def _write_csv(results: list[dict]) -> None:
    phases = ["extraction", "risk_scoring", "graph_build", "routing", "assignment"]
    fieldnames = [
        "scenario", "mode_id", "mode", "n_ranks", "n_workers", "total_cores",
        "success", "wall_time_s", "total_pipeline_s", "peak_rss_mb",
    ]
    for p in phases:
        fieldnames += [f"{p}_time_s", f"{p}_delta_mb"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {
                "scenario":        r.get("scenario"),
                "mode_id":         r.get("mode_id"),
                "mode":            r.get("mode"),
                "n_ranks":         r.get("n_ranks"),
                "n_workers":       r.get("n_workers"),
                "total_cores":     r.get("total_cores"),
                "success":         r.get("success"),
                "wall_time_s":     r.get("wall_time_s"),
                "total_pipeline_s": r.get("total_s"),
                "peak_rss_mb":     r.get("peak_rss_mb"),
            }
            timings  = r.get("timings", {})
            mem      = r.get("memory_mb", {})
            for p in phases:
                row[f"{p}_time_s"]   = timings.get(p, "")
                row[f"{p}_delta_mb"] = mem.get(p, {}).get("delta_mb", "")
            writer.writerow(row)
    log.info(f"CSV written → {RESULTS_CSV}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RespondOR pipeline across all scenarios and execution modes"
    )
    parser.add_argument(
        "--scenarios", nargs="*",
        help="Scenario IDs to benchmark (default: all configs/*.yaml). "
             "E.g. banjarnegara_landslide_2021 merapi_eruption_2023",
    )
    parser.add_argument(
        "--modes", nargs="*",
        help="Mode IDs to run. Available: naive, parallel_Nw, hpc_Nr_Mw. "
             "Default: all modes in the matrix.",
    )
    parser.add_argument(
        "--parallel-workers", nargs="*", type=int,
        default=[2, 4, 8, 16, 32, 64],
        metavar="N",
        help="Worker counts for parallel mode (default: 2 4 8 16 32 64)",
    )
    parser.add_argument(
        "--hpc-ranks", nargs="*", type=int,
        default=[2, 4],
        metavar="R",
        help="MPI rank counts for HPC mode (default: 2 4)",
    )
    parser.add_argument(
        "--hpc-workers", nargs="*", type=int,
        default=[8, 16, 64],
        metavar="W",
        help="Workers per rank for HPC mode (default: 8 16 64)",
    )
    parser.add_argument(
        "--timeout", type=int, default=3600,
        help="Per-run timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--benchmark-village-limit", type=int, default=None,
        metavar="N",
        help="Limit villages per scenario for quick tests (e.g. 100)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip runs whose result already exists in benchmark_results.json",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--skip-mpi", action="store_true",
        help="Skip HPC/MPI runs even if mpi4py is available",
    )
    parser.add_argument(
        "--mpi-launcher", choices=["mpirun", "srun"], default="mpirun",
        help="MPI launcher for HPC runs (default: mpirun). "
             "Use 'srun' on SLURM clusters with PMIx support.",
    )
    args = parser.parse_args()

    # ---- Resolve scenario list ----
    if args.scenarios:
        configs = [CONFIGS_DIR / f"{s}.yaml" for s in args.scenarios]
        missing = [c for c in configs if not c.exists()]
        if missing:
            parser.error(f"Config(s) not found: {missing}")
    else:
        configs = sorted(CONFIGS_DIR.glob("*.yaml"))

    # ---- Build mode matrix ----
    all_modes = _build_mode_matrix(
        parallel_workers=args.parallel_workers,
        hpc_ranks=args.hpc_ranks,
        hpc_workers=args.hpc_workers,
    )

    if args.modes:
        mode_ids = set(args.modes)
        all_modes = [m for m in all_modes if m["id"] in mode_ids]
        unknown = mode_ids - {m["id"] for m in all_modes}
        if unknown:
            log.warning(f"Unknown mode IDs: {unknown}")

    mpi_ok = _mpi_available() and not args.skip_mpi
    if not mpi_ok:
        skipped = [m["id"] for m in all_modes if m.get("mpi_ranks", 1) > 1]
        if skipped:
            log.warning(f"MPI not available — skipping: {skipped}")
        all_modes = [m for m in all_modes if m.get("mpi_ranks", 1) <= 1]

    # ---- Load existing results ----
    existing = _load_results()
    existing_keys = {_result_key(r["scenario"], r["mode_id"]) for r in existing}

    n_total = len(configs) * len(all_modes)
    log.info(
        f"Benchmark plan: {len(configs)} scenario(s) × "
        f"{len(all_modes)} mode(s) = {n_total} run(s)"
    )
    if args.resume and existing:
        log.info(f"  Resuming — {len(existing_keys)} run(s) already complete")

    results = list(existing)
    n_done = 0

    for cfg_path in configs:
        scenario_id = cfg_path.stem
        log.info(f"\n{'='*60}")
        log.info(f"Scenario: {scenario_id}")
        log.info(f"{'='*60}")

        for mode in all_modes:
            key = _result_key(scenario_id, mode["id"])
            if args.resume and key in existing_keys:
                log.info(f"  [{mode['id']}] SKIP (already in results)")
                n_done += 1
                continue

            run_result = _run_mode(
                scenario_config=cfg_path,
                mode=mode,
                timeout_s=args.timeout,
                village_limit=args.benchmark_village_limit,
                dry_run=args.dry_run,
                mpi_launcher=args.mpi_launcher,
            )
            n_done += 1

            if args.dry_run:
                log.info(f"  [{mode['id']}] DRY-RUN")
                continue

            entry = {
                "scenario":    scenario_id,
                "mode_id":     mode["id"],
                "mode":        mode["mode"],
                "n_ranks":     mode["n_ranks"],
                "n_workers":   mode["n_workers"],
                "total_cores": mode["total_cores"],
                "timestamp":   datetime.now().isoformat(),
                **run_result,
            }
            results.append(entry)
            _save_results(results)

            if run_result.get("success"):
                log.info(
                    f"  [{mode['id']}] OK — "
                    f"total={run_result['total_s']:.2f}s  "
                    f"wall={run_result['wall_time_s']:.2f}s  "
                    f"peak={run_result['peak_rss_mb']:.0f} MiB"
                )
            else:
                log.warning(
                    f"  [{mode['id']}] FAILED — {run_result.get('error', '?')[:120]}"
                )

            log.info(f"  Progress: {n_done}/{n_total}")

    if not args.dry_run:
        _write_csv(results)
        log.info(f"\nDone. Results → {RESULTS_JSON}")
        _print_summary(results)


def _print_summary(results: list[dict]) -> None:
    """Print a compact timing table to stdout."""
    successful = [r for r in results if r.get("success")]
    if not successful:
        return

    log.info("\n── Timing summary (pipeline total, seconds) ──")
    scenarios = sorted({r["scenario"] for r in successful})
    modes     = sorted({r["mode_id"]  for r in successful},
                       key=lambda m: ({"naive": 0, "parallel": 1, "hpc": 2}.get(
                           m.split("_")[0], 9), m))

    # Header
    col_w = max(len(m) for m in modes) + 2
    hdr = f"{'scenario':<40}" + "".join(f"{m:>{col_w}}" for m in modes)
    log.info(hdr)
    log.info("-" * len(hdr))

    for sc in scenarios:
        row = f"{sc:<40}"
        by_mode = {r["mode_id"]: r for r in successful if r["scenario"] == sc}
        for m in modes:
            if m in by_mode:
                row += f"{by_mode[m]['total_s']:>{col_w}.1f}"
            else:
                row += f"{'—':>{col_w}}"
        log.info(row)


if __name__ == "__main__":
    main()
