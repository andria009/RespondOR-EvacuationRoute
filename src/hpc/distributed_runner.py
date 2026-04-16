"""
HPC runner using MPI (mpi4py) + SLURM for distributed route computation.

Execution model:
  - All MPI ranks run the same Python script.
  - Rank 0 (master): loads data, builds the graph, broadcasts to workers,
    scatters village partitions, gathers routes, runs assignment.
  - Ranks 1..N-1 (workers): receive graph + shelters, compute routes for
    their assigned village partition, send results back to rank 0.

Hybrid MPI × multiprocessing:
  Each MPI rank processes its village partition using ProcessPoolExecutor
  with n_workers workers (controlled by execution.n_workers / --workers).
  This lets you use all cores on each node:

    mpirun -n 4 python -m src.main --config ... --mode hpc --workers 32
    # → 4 MPI ranks × 32 processes each = 128 cores total

  Set --workers 1 (or execution.n_workers: 1) to disable intra-rank
  parallelism and fall back to serial routing per rank.

  ProcessPoolExecutor uses the 'spawn' start method (not 'fork') to avoid
  MPI + fork deadlocks on Linux. This adds ~0.5s startup overhead per rank,
  amortised over large village partitions.

SLURM launch:
  srun --mpi=pmix -n $SLURM_NTASKS python -m src.main --config ... --mode hpc --workers $SLURM_CPUS_PER_TASK

Fallback (no mpi4py installed):
  ProcessPoolExecutor on a single node.
"""

import multiprocessing as mp
import sys
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from src.config.config_loader import AppConfig
from src.data.models import (
    DisasterInput, RegionOfInterest, RegionType, DisasterType,
    ExecutionMode, Village, Shelter, EvacuationRoute,
)
from src.data.osm_extractor import OSMExtractor
from src.data.inarisk_client import InaRISKClient
from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
from src.graph.graph_builder import EvacuationGraphBuilder
from src.routing.heuristic_optimizer import (
    HeuristicOptimizer, _routes_for_village_standalone,
)
from src.routing.assignment import PopulationAssigner
from src.hpc.runner_utils import resolve_hazard_layers, apply_risk_parallel, MemoryTracker

logger = logging.getLogger(__name__)


class MPIRunner:
    """
    HPC-grade runner using MPI for distributed route computation.
    Falls back to ProcessPoolExecutor when mpi4py is not installed.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self):
        """
        Execute the full pipeline.
        Only rank 0 returns (result, villages, shelters, routes_by_village, timings).
        Worker ranks (rank > 0) exit after route computation.
        """
        comm, rank, size = _init_mpi()
        logger.info(f"=== HPC/MPI MODE | rank={rank}/{size} ===")

        cfg = self.config
        timings = {}
        memory_mb = {}
        mem = MemoryTracker()

        # ---- Rank 0: full data loading ----
        if rank == 0:
            disaster, region = self._build_inputs(cfg)

            m0 = mem.rss_mb()
            t0 = time.perf_counter()
            nodes, edges, villages, shelters = self._extract_data(region, cfg)
            timings["extraction"] = time.perf_counter() - t0
            m0 = mem.snapshot("extraction", m0, memory_mb)
            logger.info(f"[rank 0] Extracted: {len(villages)}V {len(shelters)}S")

            t0 = time.perf_counter()
            self._apply_risk(villages, shelters, disaster, cfg)
            timings["risk_scoring"] = time.perf_counter() - t0
            m0 = mem.snapshot("risk_scoring", m0, memory_mb)

            t0 = time.perf_counter()
            builder = EvacuationGraphBuilder()
            G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
            builder.attach_pois_to_graph(villages, shelters)
            if not cfg.skip_inarisk:
                hazard_layers_edges = resolve_hazard_layers(cfg, disaster)
                inarisk = InaRISKClient(
                    batch_size=cfg.extraction.inarisk_batch_size,
                    rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
                )
                builder.apply_inarisk_to_edges(
                    inarisk=inarisk,
                    hazard_layers=hazard_layers_edges,
                    aggregation=cfg.routing.hazard_aggregation,
                    cache_path=Path(cfg.extraction.inarisk_cache_dir) / "road_risk_cache.json",
                    use_cache=cfg.extraction.use_cached_inarisk,
                )
            builder.propagate_poi_risk_to_graph(villages, shelters)
            timings["graph_build"] = time.perf_counter() - t0
            m0 = mem.snapshot("graph_build", m0, memory_mb)
            logger.info(f"[rank 0] Graph: {G.number_of_nodes()}N {G.number_of_edges()}E")
        else:
            disaster = villages = shelters = G = None
            timings = {}

        # ---- Broadcast shared state to all ranks ----
        if comm is not None:
            G = comm.bcast(G, root=0)
            shelters = comm.bcast(shelters, root=0)
            disaster = comm.bcast(disaster, root=0)

        # ---- Scatter village partitions ----
        routing_params = {
            "w_dist":            cfg.routing.weight_distance,
            "w_risk":            cfg.routing.weight_risk,
            "w_quality":         cfg.routing.weight_road_quality,
            "w_time":            cfg.routing.weight_time,
            "w_disaster":        cfg.routing.weight_disaster_distance,
            "max_routes":        cfg.routing.max_routes_per_village,
            "disaster_location": (cfg.disaster.lat, cfg.disaster.lon),
        }

        if rank == 0:
            valid = [v for v in villages
                     if v.nearest_node_id is not None and v.nearest_node_id in G]
            partitions = _partition_villages(valid, size)
            logger.info(
                f"[rank 0] Distributing {len(valid)} villages across {size} ranks"
            )
        else:
            partitions = None

        if comm is not None:
            my_villages = comm.scatter(partitions, root=0)
        else:
            my_villages = partitions[0] if partitions else []

        # ---- All ranks compute routes for their partition ----
        shelter_nodes = {s.nearest_node_id: s for s in shelters
                         if s.nearest_node_id is not None}

        n_workers_per_rank = cfg.execution.n_workers

        t0_routing = time.perf_counter()
        my_routes: List[EvacuationRoute] = []
        if n_workers_per_rank > 1 and len(my_villages) > 1:
            compute_fn = partial(
                _routes_for_village_standalone,
                G=G,
                shelters=shelters,
                shelter_nodes=shelter_nodes,
                **routing_params,
            )
            mp_ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=n_workers_per_rank, mp_context=mp_ctx
            ) as executor:
                futures = {executor.submit(compute_fn, v): v for v in my_villages}
                for future in as_completed(futures):
                    try:
                        my_routes.extend(future.result())
                    except Exception as e:
                        v = futures[future]
                        logger.warning(
                            f"[rank {rank}] Village {v.village_id} routing failed: {e}"
                        )
        else:
            for v in my_villages:
                routes = _routes_for_village_standalone(
                    village=v,
                    G=G,
                    shelters=shelters,
                    shelter_nodes=shelter_nodes,
                    **routing_params,
                )
                my_routes.extend(routes)
        routing_time = time.perf_counter() - t0_routing
        logger.info(
            f"[rank {rank}] Computed {len(my_routes)} routes "
            f"for {len(my_villages)} villages in {routing_time:.1f}s "
            f"({n_workers_per_rank} worker{'s' if n_workers_per_rank != 1 else ''})"
        )

        # ---- Gather routes at rank 0 ----
        if comm is not None:
            all_routes_gathered = comm.gather(my_routes, root=0)
        else:
            all_routes_gathered = [my_routes]

        # Worker ranks are done — exit cleanly.
        # comm.gather() is itself a collective that synchronises all ranks,
        # so no additional Barrier is needed here.  Calling Barrier after
        # gather would hang workers indefinitely because rank 0 never
        # reaches that Barrier — it continues with assignment and I/O instead.
        if rank != 0:
            if comm is not None:
                from mpi4py import MPI
                MPI.Finalize()
            sys.exit(0)

        # ---- Rank 0: aggregate, assign, save ----
        all_routes: List[EvacuationRoute] = []
        for chunk in all_routes_gathered:
            all_routes.extend(chunk)
        timings["routing"] = sum(
            t for chunk_time_list in [all_routes_gathered]
            for t in []
        ) or routing_time  # rank 0's own routing time as proxy

        logger.info(f"[rank 0] Total routes gathered: {len(all_routes)}")

        optimizer = HeuristicOptimizer()
        routes_by_village = optimizer.rank_routes(all_routes)

        t0 = time.perf_counter()
        assigner = PopulationAssigner(method=cfg.routing.assignment_method)
        total_runtime = sum(timings.values())
        result = assigner.assign(
            villages=villages,
            shelters=shelters,
            routes_by_village=routes_by_village,
            scenario_id=cfg.scenario_id,
            disaster=disaster,
            mode=ExecutionMode.HPC,
            runtime_s=total_runtime,
        )
        timings["assignment"] = time.perf_counter() - t0
        mem.snapshot("assignment", mem.rss_mb(), memory_mb)

        # Gather peak RSS from all ranks (workers already exited, so only rank 0 here)
        rank0_peak = mem.peak_rss_mb()

        total_time = sum(timings.values())
        result.runtime_s = total_time

        logger.info(
            f"=== HPC DONE [{total_time:.2f}s] | "
            f"Evacuated: {result.total_evacuated}/{result.total_population} "
            f"({100*result.evacuation_ratio:.1f}%) | ranks={size} ==="
        )

        self._save_timings(
            timings, memory_mb, rank0_peak,
            n_ranks=size, n_workers_per_rank=cfg.execution.n_workers,
        )
        return result, villages, shelters, routes_by_village, timings, G

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_inputs(self, cfg):
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
        return disaster, region

    def _extract_data(self, region, cfg):
        extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)

        if cfg.preloaded_network_pycgr:
            nodes, edges = extractor.load_network_from_pycgr(cfg.preloaded_network_pycgr)
        elif cfg.preloaded_network_json:
            nodes, edges = extractor.load_network_from_json(cfg.preloaded_network_json)
        else:
            nodes, edges = extractor.extract_road_network(
                region,
                network_type=cfg.extraction.network_type,
                road_types=cfg.extraction.road_types,
                use_cache=cfg.extraction.use_cached_osm,
            )

        if cfg.preloaded_villages_geojson:
            villages = extractor.load_villages_from_geojson(cfg.preloaded_villages_geojson)
        else:
            villages = extractor.extract_villages(
                region,
                admin_levels=cfg.extraction.village_admin_levels,
                population_density_per_km2=cfg.extraction.village_pop_density,
                max_population_per_village=cfg.extraction.village_max_pop,
                use_cache=cfg.extraction.use_cached_osm,
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

        if cfg.preloaded_shelters_geojson:
            shelters = extractor.load_shelters_from_geojson(cfg.preloaded_shelters_geojson)
        else:
            shelters = extractor.extract_shelters(
                region,
                shelter_tags=cfg.extraction.shelter_tags,
                min_area_m2=cfg.extraction.shelter_min_area_m2,
                m2_per_person=cfg.extraction.shelter_m2_per_person,
                use_cache=cfg.extraction.use_cached_osm,
                cluster_eps_m=cfg.extraction.shelter_cluster_eps_m,
                cluster_min_shelters=cfg.extraction.shelter_cluster_min_shelters,
            )

        PopulationLoader().apply_population(
            villages,
            population_csv=cfg.extraction.population_csv,
            density_per_km2=cfg.extraction.village_pop_density,
        )
        ShelterCapacityLoader().apply_capacity(
            shelters,
            capacity_csv=cfg.extraction.shelter_capacity_csv,
            m2_per_person=cfg.extraction.shelter_m2_per_person,
        )
        return nodes, edges, villages, shelters

    def _apply_risk(self, villages, shelters, disaster, cfg):
        apply_risk_parallel(cfg, villages, shelters, disaster, n_threads=cfg.execution.n_workers)

    def _save_timings(
        self,
        timings: dict,
        memory_mb: dict,
        peak_rss_mb: float,
        n_ranks: int = 1,
        n_workers_per_rank: int = 1,
    ):
        import json
        out_path = self.output_dir / f"timings_hpc_{n_ranks}r_{n_workers_per_rank}w.json"
        with open(out_path, "w") as f:
            json.dump({
                "mode": "hpc",
                "framework": "mpi",
                "n_ranks": n_ranks,
                "n_workers_per_rank": n_workers_per_rank,
                "total_cores": n_ranks * n_workers_per_rank,
                "timings": timings,
                "total": sum(timings.values()),
                "memory_mb": memory_mb,
                "peak_rss_mb": round(peak_rss_mb, 1),
            }, f, indent=2)
        logger.info(f"Timings saved to {out_path}")


# ------------------------------------------------------------------ #
# MPI helpers
# ------------------------------------------------------------------ #

def _init_mpi():
    """
    Initialize MPI communicator.
    Returns (comm, rank, size).
    comm=None means no MPI (single process).
    """
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.Get_rank(), comm.Get_size()
    except ImportError:
        logger.warning("mpi4py not installed — HPC mode running on single node "
                       "with ProcessPoolExecutor fallback")
        return None, 0, 1


def _partition_villages(
    villages: List[Village],
    n_parts: int,
) -> List[List[Village]]:
    """
    Split villages into n_parts partitions (round-robin) for scatter.
    Returns a list of n_parts lists.
    """
    partitions: List[List[Village]] = [[] for _ in range(n_parts)]
    for i, v in enumerate(villages):
        partitions[i % n_parts].append(v)
    return partitions
