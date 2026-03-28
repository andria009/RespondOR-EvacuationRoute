"""
Parallel (single-machine) runner.
Uses Python multiprocessing.Pool and concurrent.futures for:
- Parallel OSM data extraction (by POI type)
- Parallel risk score querying (by batch)
- Parallel route computation (by village, using ProcessPoolExecutor)
- Parallel simulation batch runs

Benchmarked against naive runner to measure speedup.
"""

import time
import logging
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial
from typing import List, Optional, Tuple

from src.config.config_loader import AppConfig
from src.data.models import (
    DisasterInput, RegionOfInterest, RegionType, DisasterType,
    ExecutionMode, Village, Shelter, NetworkNode, NetworkEdge
)
from src.data.osm_extractor import OSMExtractor
from src.data.inarisk_client import InaRISKClient
from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
from src.graph.graph_builder import EvacuationGraphBuilder
from src.routing.heuristic_optimizer import HeuristicOptimizer, _routes_for_village_standalone
from src.routing.assignment import PopulationAssigner

logger = logging.getLogger(__name__)


class ParallelRunner:
    """
    Parallel execution using multiprocessing on a single machine.
    """

    def __init__(self, config: AppConfig, n_workers: Optional[int] = None):
        self.config = config
        self.n_workers = n_workers or min(config.execution.n_workers, cpu_count())
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute full pipeline in parallel. Returns OptimizationResult."""
        cfg = self.config
        timings = {}

        logger.info(f"=== PARALLEL MODE ({self.n_workers} workers) ===")

        disaster = DisasterInput(
            location=(cfg.disaster.lat, cfg.disaster.lon),
            disaster_type=DisasterType(cfg.disaster.disaster_type),
            name=cfg.disaster.name,
        )
        region = RegionOfInterest(
            region_type=RegionType(cfg.region.region_type),
            bbox=tuple(cfg.region.bbox) if cfg.region.bbox else None,
            center=tuple(cfg.region.center) if cfg.region.center else None,
            radius_km=cfg.region.radius_km,
        )

        # ---- Stage 1: Parallel data extraction ----
        t0 = time.perf_counter()
        nodes, edges, villages, shelters = self._extract_parallel(region, cfg)
        timings["extraction"] = time.perf_counter() - t0
        logger.info(f"[{timings['extraction']:.2f}s] Extracted (parallel): "
                    f"{len(nodes)}N {len(edges)}E {len(villages)}V {len(shelters)}S")

        # ---- Stage 2: Parallel risk scoring ----
        t0 = time.perf_counter()
        self._apply_risk_parallel(villages, shelters, disaster, cfg)
        timings["risk_scoring"] = time.perf_counter() - t0
        logger.info(f"[{timings['risk_scoring']:.2f}s] Risk scores (parallel)")

        # ---- Stage 3: Graph build (single-core, fast) ----
        t0 = time.perf_counter()
        builder = EvacuationGraphBuilder()
        G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
        builder.attach_pois_to_graph(villages, shelters)
        timings["graph_build"] = time.perf_counter() - t0

        # ---- Optional: limit villages for benchmarking ----
        limit = cfg.benchmark_village_limit
        if limit and limit > 0 and len(villages) > limit:
            logger.info(f"Benchmark limit: using {limit}/{len(villages)} villages")
            villages = villages[:limit]

        # ---- Stage 4: Parallel routing ----
        t0 = time.perf_counter()
        optimizer = HeuristicOptimizer(
            weight_distance=cfg.routing.weight_distance,
            weight_risk=cfg.routing.weight_risk,
            weight_road_quality=cfg.routing.weight_road_quality,
            weight_time=cfg.routing.weight_time,
            max_routes_per_village=cfg.routing.max_routes_per_village,
            max_risk_threshold=cfg.routing.max_route_risk_threshold,
        )
        routes = optimizer.compute_routes(
            G, villages, shelters,
            mode=ExecutionMode.PARALLEL,
            n_workers=self.n_workers,
        )
        routes_by_village = optimizer.rank_routes(routes)
        timings["routing"] = time.perf_counter() - t0
        logger.info(f"[{timings['routing']:.2f}s] Routing (parallel): {len(routes)} routes")

        # ---- Stage 5: Assignment ----
        t0 = time.perf_counter()
        assigner = PopulationAssigner(method="greedy")
        result = assigner.assign(
            villages, shelters, routes_by_village,
            scenario_id=cfg.scenario_id,
            disaster=disaster,
            mode=ExecutionMode.PARALLEL,
        )
        timings["assignment"] = time.perf_counter() - t0

        total_time = sum(timings.values())
        result.runtime_s = total_time
        logger.info(f"=== DONE [{total_time:.2f}s] | "
                    f"Evacuated: {result.total_evacuated}/{result.total_population} ===")

        self._save_timings(timings)
        return result, villages, shelters, routes_by_village, timings

    # ------------------------------------------------------------------ #
    # Parallel extraction: run OSM queries for roads, villages, shelters
    # concurrently using threads (I/O-bound)
    # ------------------------------------------------------------------ #

    def _extract_parallel(self, region, cfg):
        extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)

        with ThreadPoolExecutor(max_workers=3) as tpe:
            # Roads
            if cfg.preloaded_network_pycgr:
                net_future = tpe.submit(extractor.load_network_from_pycgr,
                                        cfg.preloaded_network_pycgr)
            else:
                net_future = tpe.submit(extractor.extract_road_network, region,
                                        cfg.extraction.network_type,
                                        cfg.extraction.use_cached_osm)

            if cfg.preloaded_poi_csv:
                # Legacy RespondOR v1 format: single CSV with both villages and shelters
                poi_future = tpe.submit(extractor.load_pois_from_csv, cfg.preloaded_poi_csv)
                nodes, edges = net_future.result()
                villages, shelters = poi_future.result()
            else:
                # Villages
                if cfg.preloaded_villages_geojson:
                    vil_future = tpe.submit(extractor.load_villages_from_geojson,
                                            cfg.preloaded_villages_geojson)
                else:
                    vil_future = tpe.submit(extractor.extract_villages, region,
                                            cfg.extraction.village_admin_level,
                                            cfg.extraction.use_cached_osm)

                # Shelters
                if cfg.preloaded_shelters_geojson:
                    shel_future = tpe.submit(extractor.load_shelters_from_geojson,
                                             cfg.preloaded_shelters_geojson)
                else:
                    shel_future = tpe.submit(extractor.extract_shelters, region,
                                             None, cfg.extraction.use_cached_osm)

                nodes, edges = net_future.result()
                villages = vil_future.result()
                shelters = shel_future.result()

        pop_loader = PopulationLoader()
        pop_loader.apply_population(villages, cfg.extraction.population_csv,
                                    cfg.extraction.default_pop_density)
        cap_loader = ShelterCapacityLoader()
        cap_loader.apply_capacity(shelters, cfg.extraction.shelter_capacity_csv,
                                  cfg.extraction.m2_per_person)
        return nodes, edges, villages, shelters

    # ------------------------------------------------------------------ #
    # Parallel risk scoring: query villages and shelters concurrently
    # ------------------------------------------------------------------ #

    def _apply_risk_parallel(self, villages, shelters, disaster, cfg):
        inarisk = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )

        with ThreadPoolExecutor(max_workers=2) as tpe:
            vf = tpe.submit(inarisk.enrich_villages_with_risk, villages, disaster.disaster_type)
            sf = tpe.submit(inarisk.enrich_shelters_with_risk, shelters, disaster.disaster_type)
            vf.result()
            sf.result()

    def _save_timings(self, timings: dict):
        import json
        out_path = self.output_dir / f"timings_parallel_{self.n_workers}w.json"
        with open(out_path, "w") as f:
            json.dump({"mode": "parallel", "n_workers": self.n_workers,
                       "timings": timings, "total": sum(timings.values())}, f, indent=2)
