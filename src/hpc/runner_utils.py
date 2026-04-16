"""
Shared helpers for all pipeline runners (naive, parallel, hpc/MPI).
"""

import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from src.data.models import DisasterType

logger = logging.getLogger(__name__)


class MemoryTracker:
    """
    Snapshots process RSS memory at pipeline phase boundaries.

    Uses psutil when available (accurate, cross-platform).
    Falls back to the stdlib resource module (Unix only) for peak RSS.
    Returns 0.0 silently on Windows or when neither library is available.

    Usage:
        mem = MemoryTracker()
        m0 = mem.rss_mb()
        # ... do work ...
        delta = mem.rss_mb() - m0
        peak = mem.peak_rss_mb()   # max RSS since process start
    """

    def __init__(self):
        self._proc = None
        try:
            import psutil
            self._proc = psutil.Process(os.getpid())
        except ImportError:
            pass

    def rss_mb(self) -> float:
        """Current resident set size in MiB."""
        if self._proc is not None:
            try:
                return self._proc.memory_info().rss / 1_048_576
            except Exception:
                pass
        return 0.0

    def peak_rss_mb(self) -> float:
        """
        Peak RSS since process start (MiB).
        Uses resource.getrusage — macOS returns bytes, Linux returns KB.
        Falls back to current RSS when resource is unavailable.
        """
        try:
            import resource
            ru = resource.getrusage(resource.RUSAGE_SELF)
            if sys.platform == "darwin":
                return ru.ru_maxrss / 1_048_576      # bytes on macOS
            return ru.ru_maxrss / 1024.0              # KB on Linux
        except Exception:
            return self.rss_mb()

    def snapshot(self, phase: str, before_mb: float, memory_mb: dict) -> float:
        """
        Record memory delta for a completed phase.

        Call before the phase:  m0 = mem.rss_mb()
        Call after  the phase:  mem.snapshot("routing", m0, memory_mb)

        Returns current RSS in MiB so it can be chained as the next m0.
        """
        after_mb = self.rss_mb()
        memory_mb[phase] = {
            "before_mb": round(before_mb, 1),
            "after_mb":  round(after_mb, 1),
            "delta_mb":  round(after_mb - before_mb, 1),
        }
        return after_mb


def resolve_hazard_layers(cfg, disaster) -> dict:
    """
    Return {DisasterType: weight} for risk enrichment.
    Uses cfg.routing.hazard_layers when configured; falls back to
    {disaster.disaster_type: 1.0} for single-layer scenarios.
    """
    raw = cfg.routing.hazard_layers  # {str: float}
    if raw:
        resolved = {}
        for name, weight in raw.items():
            try:
                resolved[DisasterType(name)] = float(weight)
            except ValueError:
                logger.warning(f"Unknown hazard layer '{name}' in hazard_layers — skipping")
        if resolved:
            return resolved
    return {disaster.disaster_type: 1.0}


def apply_risk_parallel(cfg, villages, shelters, disaster, n_threads: int = 4) -> None:
    """
    Parallel InaRISK risk scoring for villages and shelters.

    Parallelism strategy
    --------------------
    Tasks are defined as (hazard_layer × poi_type) pairs.
    For a single-hazard scenario: 2 tasks (villages, shelters).
    For a compound scenario with N layers: 2×N tasks.

    All tasks run concurrently up to n_threads. Since each task writes
    to a unique cache key, dict access is safe under CPython's GIL.
    File saves are serialised with a threading.Lock so the JSON cache
    is never corrupted by concurrent writes.

    Per-batch intermediate saves are disabled during parallel execution
    (cache_path=None is passed to _get_poi_scores_cached). A single
    save is issued after each task completes (protected by the lock),
    preserving partial-progress recovery granularity at the task level.

    Falls back to serial execution when n_threads == 1.

    Parameters
    ----------
    n_threads : int
        Maximum concurrent tasks. Set to cfg.execution.n_workers in
        parallel_runner; cfg.execution.n_workers (per MPI rank) in
        distributed_runner.
    """
    from src.data.inarisk_client import InaRISKClient

    if cfg.skip_inarisk:
        logger.warning("skip_inarisk=true — all risk scores set to 0.0 (InaRISK bypassed)")
        for v in villages:
            v.risk_scores["composite"] = 0.0
        for s in shelters:
            s.risk_scores["composite"] = 0.0
        return

    hazard_layers = resolve_hazard_layers(cfg, disaster)
    cache_path = Path(cfg.extraction.inarisk_cache_dir) / "poi_risk_cache.json"
    grid_cache_path = Path(cfg.extraction.inarisk_cache_dir) / "road_risk_cache.json"
    use_cache = cfg.extraction.use_cached_inarisk
    aggregation = cfg.routing.hazard_aggregation
    total_weight = sum(hazard_layers.values()) or 1.0

    is_compound = len(hazard_layers) > 1
    if is_compound:
        logger.info(
            f"Compound hazard: {', '.join(f'{dt.value}×{w}' for dt, w in hazard_layers.items())}"
            f" [{aggregation}] (parallel, up to {n_threads} threads)"
        )

    inarisk = InaRISKClient(
        batch_size=cfg.extraction.inarisk_batch_size,
        rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
    )

    # Load cache once — shared across all threads (different keys, no conflict)
    cache = inarisk._load_poi_cache(cache_path)
    cache_lock = threading.Lock()

    village_points = [(v.centroid_lat, v.centroid_lon) for v in villages]
    shelter_points = [(s.centroid_lat, s.centroid_lon) for s in shelters]

    # Build task list: each (dt, prefix) pair is one independent task
    tasks = []
    for dt in hazard_layers:
        tasks.append((dt, "villages", village_points))
        tasks.append((dt, "shelters", shelter_points))

    # layer_poi_scores[(dt, prefix)] = [scores per object]
    layer_poi_scores = {}

    def _query_task(dt: DisasterType, prefix: str, points: List) -> List[float]:
        """Query one (hazard_layer × poi_type) combination."""
        cache_key = f"{prefix}_{dt.value}"
        scores = inarisk._get_poi_scores_cached(
            points=points,
            disaster_type=dt,
            cache=cache,
            cache_key=cache_key,
            use_cache=use_cache,
            cache_path=None,          # disable per-batch file saves; we save below
            grid_cache_path=grid_cache_path,
        )
        # Save after each task so partial progress survives a crash
        with cache_lock:
            inarisk._save_poi_cache(cache, cache_path)
        return scores

    effective_threads = min(n_threads, len(tasks)) if n_threads > 1 else 1

    if effective_threads > 1:
        logger.info(
            f"Risk scoring: {len(tasks)} tasks "
            f"({len(hazard_layers)} layer(s) × 2 poi types) "
            f"on {effective_threads} threads"
        )
        with ThreadPoolExecutor(max_workers=effective_threads) as executor:
            futures = {
                executor.submit(_query_task, dt, prefix, points): (dt, prefix)
                for dt, prefix, points in tasks
            }
            for future in as_completed(futures):
                dt, prefix = futures[future]
                try:
                    layer_poi_scores[(dt, prefix)] = future.result()
                except Exception as e:
                    logger.warning(
                        f"Risk scoring failed for {prefix}/{dt.value}: {e} — using 0.0"
                    )
                    n = len(village_points) if prefix == "villages" else len(shelter_points)
                    layer_poi_scores[(dt, prefix)] = [0.0] * n
    else:
        for dt, prefix, points in tasks:
            layer_poi_scores[(dt, prefix)] = _query_task(dt, prefix, points)

    # Assign per-layer scores and compute composite on each object
    def _assign(objects: list, prefix: str) -> None:
        for i, obj in enumerate(objects):
            per_layer = {dt.value: layer_poi_scores[(dt, prefix)][i] for dt in hazard_layers}
            obj.risk_scores.update(per_layer)

            if aggregation == "max":
                composite = max(per_layer.values()) if per_layer else 0.0
            else:  # weighted_sum (normalised)
                composite = sum(
                    layer_poi_scores[(dt, prefix)][i] * (w / total_weight)
                    for dt, w in hazard_layers.items()
                )
            obj.risk_scores["composite"] = round(composite, 4)

    _assign(villages, "villages")
    _assign(shelters, "shelters")

    # Final cache save (in case the lock-save above missed anything on a cached run)
    inarisk._save_poi_cache(cache, cache_path)

    composites_v = [v.risk_scores.get("composite", 0.0) for v in villages]
    composites_s = [s.risk_scores.get("composite", 0.0) for s in shelters]
    if composites_v:
        logger.info(
            f"Village risk: min={min(composites_v):.3f} max={max(composites_v):.3f} "
            f"mean={sum(composites_v)/len(composites_v):.3f}"
        )
    if composites_s:
        logger.info(
            f"Shelter risk: min={min(composites_s):.3f} max={max(composites_s):.3f} "
            f"mean={sum(composites_s)/len(composites_s):.3f}"
        )
