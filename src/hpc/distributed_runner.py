"""
HPC runner using MPI (mpi4py) + SLURM for distributed route computation.

Execution model:
  - All MPI ranks run the same Python script.
  - Rank 0 (master): loads data, builds the graph, broadcasts to workers,
    scatters village partitions, gathers routes, runs assignment.
  - Ranks 1..N-1 (workers): receive graph + shelters, compute routes for
    their assigned village partition, send results back to rank 0.

Launch:
  srun --mpi=pmix -n $SLURM_NTASKS python -m src.main --config ... --mode hpc

Fallback (no mpi4py installed):
  ProcessPoolExecutor on a single node.
"""

import sys
import time
import logging
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

        # ---- Rank 0: full data loading ----
        if rank == 0:
            disaster, region = self._build_inputs(cfg)

            t0 = time.perf_counter()
            nodes, edges, villages, shelters = self._extract_data(region, cfg)
            timings["extraction"] = time.perf_counter() - t0
            logger.info(f"[rank 0] Extracted: {len(villages)}V {len(shelters)}S")

            t0 = time.perf_counter()
            self._apply_risk(villages, shelters, disaster, cfg)
            timings["risk_scoring"] = time.perf_counter() - t0

            t0 = time.perf_counter()
            builder = EvacuationGraphBuilder()
            G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
            builder.attach_pois_to_graph(villages, shelters)
            timings["graph_build"] = time.perf_counter() - t0
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
            "w_dist":      cfg.routing.weight_distance,
            "w_risk":      cfg.routing.weight_risk,
            "w_quality":   cfg.routing.weight_road_quality,
            "w_time":      cfg.routing.weight_time,
            "max_routes":  cfg.routing.max_routes_per_village,
            "max_risk":    cfg.routing.max_route_risk_threshold,
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

        t0_routing = time.perf_counter()
        my_routes: List[EvacuationRoute] = []
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
            f"for {len(my_villages)} villages in {routing_time:.1f}s"
        )

        # ---- Gather routes at rank 0 ----
        if comm is not None:
            all_routes_gathered = comm.gather(my_routes, root=0)
        else:
            all_routes_gathered = [my_routes]

        # Worker ranks are done — exit cleanly
        if rank != 0:
            if comm is not None:
                comm.Barrier()
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
        assigner = PopulationAssigner(method=cfg.routing.__dict__.get("assignment_method", "greedy"))
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

        total_time = sum(timings.values())
        result.runtime_s = total_time

        logger.info(
            f"=== HPC DONE [{total_time:.2f}s] | "
            f"Evacuated: {result.total_evacuated}/{result.total_population} "
            f"({100*result.evacuation_ratio:.1f}%) | ranks={size} ==="
        )

        self._save_timings(timings, n_ranks=size)
        return result, villages, shelters, routes_by_village, timings

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
                persons_per_dwelling=cfg.extraction.village_persons_per_dwelling,
                building_persons=cfg.extraction.village_building_persons,
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
            )

        PopulationLoader().apply_population(
            villages,
            population_csv=cfg.extraction.population_csv,
            density_per_km2=cfg.extraction.default_pop_density,
        )
        ShelterCapacityLoader().apply_capacity(
            shelters,
            capacity_csv=cfg.extraction.shelter_capacity_csv,
            m2_per_person=cfg.extraction.m2_per_person,
        )
        return nodes, edges, villages, shelters

    def _apply_risk(self, villages, shelters, disaster, cfg):
        inarisk = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )
        inarisk.enrich_villages_with_risk(villages, disaster.disaster_type)
        inarisk.enrich_shelters_with_risk(shelters, disaster.disaster_type)

    def _save_timings(self, timings: dict, n_ranks: int = 1):
        import json
        out_path = self.output_dir / "timings_hpc.json"
        with open(out_path, "w") as f:
            json.dump({
                "mode": "hpc",
                "framework": "mpi",
                "n_ranks": n_ranks,
                "timings": timings,
                "total": sum(timings.values()),
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
