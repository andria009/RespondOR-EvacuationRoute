"""
Naive (sequential) runner.
Executes all pipeline stages in a single process with no parallelism.
Serves as the baseline for speedup benchmarking.
"""

import time
import logging
from pathlib import Path
from typing import Optional

from src.config.config_loader import AppConfig
from src.data.models import (
    DisasterInput, RegionOfInterest, RegionType, DisasterType, ExecutionMode
)
from src.data.osm_extractor import OSMExtractor
from src.data.inarisk_client import InaRISKClient
from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
from src.graph.graph_builder import EvacuationGraphBuilder
from src.routing.heuristic_optimizer import HeuristicOptimizer
from src.routing.assignment import PopulationAssigner

logger = logging.getLogger(__name__)


class NaiveRunner:
    """
    Executes the full evacuation optimization pipeline sequentially.
    All stages run in a single process.
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        """Execute full pipeline sequentially. Returns OptimizationResult."""
        cfg = self.config
        timings = {}

        logger.info("=== NAIVE (SEQUENTIAL) MODE ===")

        # ---- Stage 1: Build inputs ----
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

        # ---- Stage 2: Extract data ----
        t0 = time.perf_counter()
        nodes, edges, villages, shelters = self._extract_data(region, cfg)
        timings["extraction"] = time.perf_counter() - t0
        logger.info(f"[{timings['extraction']:.2f}s] Extracted: "
                    f"{len(nodes)}N {len(edges)}E {len(villages)}V {len(shelters)}S")

        # ---- Stage 3: Risk scoring ----
        t0 = time.perf_counter()
        self._apply_risk_scores(villages, shelters, disaster, cfg)
        timings["risk_scoring"] = time.perf_counter() - t0
        logger.info(f"[{timings['risk_scoring']:.2f}s] Risk scores applied")

        # ---- Stage 4: Graph construction ----
        t0 = time.perf_counter()
        builder = EvacuationGraphBuilder()
        G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
        builder.attach_pois_to_graph(villages, shelters)
        timings["graph_build"] = time.perf_counter() - t0
        logger.info(f"[{timings['graph_build']:.2f}s] Graph built: "
                    f"{G.number_of_nodes()}N {G.number_of_edges()}E")

        # ---- Optional: limit villages for benchmarking ----
        limit = cfg.benchmark_village_limit
        if limit and limit > 0 and len(villages) > limit:
            logger.info(f"Benchmark limit: using {limit}/{len(villages)} villages")
            villages = villages[:limit]

        # ---- Stage 5: Route computation ----
        t0 = time.perf_counter()
        optimizer = HeuristicOptimizer(
            weight_distance=cfg.routing.weight_distance,
            weight_risk=cfg.routing.weight_risk,
            weight_road_quality=cfg.routing.weight_road_quality,
            weight_time=cfg.routing.weight_time,
            max_routes_per_village=cfg.routing.max_routes_per_village,
            max_risk_threshold=cfg.routing.max_route_risk_threshold,
        )
        routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
        routes_by_village = optimizer.rank_routes(routes)
        timings["routing"] = time.perf_counter() - t0
        logger.info(f"[{timings['routing']:.2f}s] Computed {len(routes)} candidate routes")

        # ---- Stage 6: Population assignment ----
        t0 = time.perf_counter()
        assigner = PopulationAssigner(method="greedy")
        result = assigner.assign(
            villages=villages,
            shelters=shelters,
            routes_by_village=routes_by_village,
            scenario_id=cfg.scenario_id,
            disaster=disaster,
            mode=ExecutionMode.NAIVE,
            runtime_s=sum(timings.values()),
        )
        timings["assignment"] = time.perf_counter() - t0

        total_time = sum(timings.values())
        result.runtime_s = total_time

        logger.info(
            f"=== DONE [{total_time:.2f}s] | "
            f"Evacuated: {result.total_evacuated}/{result.total_population} "
            f"({100*result.evacuation_ratio:.1f}%) ==="
        )

        self._save_timings(timings)
        return result, villages, shelters, routes_by_village, timings

    def _extract_data(self, region, cfg):
        extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)

        # Use preloaded data if available
        if cfg.preloaded_network_pycgr:
            nodes, edges = extractor.load_network_from_pycgr(cfg.preloaded_network_pycgr)
        elif cfg.preloaded_network_json:
            nodes, edges = extractor.load_network_from_json(cfg.preloaded_network_json)
        else:
            nodes, edges = extractor.extract_road_network(
                region,
                network_type=cfg.extraction.network_type,
                use_cache=cfg.extraction.use_cached_osm,
            )

        if cfg.preloaded_poi_csv:
            # Legacy RespondOR v1 format: single CSV with both villages and shelters
            villages, shelters = extractor.load_pois_from_csv(cfg.preloaded_poi_csv)
        else:
            if cfg.preloaded_villages_geojson:
                villages = extractor.load_villages_from_geojson(cfg.preloaded_villages_geojson)
            else:
                villages = extractor.extract_villages(
                    region,
                    admin_level=cfg.extraction.village_admin_level,
                    use_cache=cfg.extraction.use_cached_osm,
                )

            if cfg.preloaded_shelters_geojson:
                shelters = extractor.load_shelters_from_geojson(cfg.preloaded_shelters_geojson)
            else:
                shelters = extractor.extract_shelters(
                    region,
                    use_cache=cfg.extraction.use_cached_osm,
                )

        # Apply population / capacity
        pop_loader = PopulationLoader()
        pop_loader.apply_population(
            villages,
            population_csv=cfg.extraction.population_csv,
            density_per_km2=cfg.extraction.default_pop_density,
        )

        cap_loader = ShelterCapacityLoader()
        cap_loader.apply_capacity(
            shelters,
            capacity_csv=cfg.extraction.shelter_capacity_csv,
            m2_per_person=cfg.extraction.m2_per_person,
        )

        return nodes, edges, villages, shelters

    def _apply_risk_scores(self, villages, shelters, disaster, cfg):
        inarisk = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )
        inarisk.enrich_villages_with_risk(villages, disaster.disaster_type)
        inarisk.enrich_shelters_with_risk(shelters, disaster.disaster_type)

    def _save_timings(self, timings: dict):
        import json
        out_path = self.output_dir / "timings_naive.json"
        with open(out_path, "w") as f:
            json.dump({"mode": "naive", "timings": timings,
                       "total": sum(timings.values())}, f, indent=2)
        logger.info(f"Timings saved to {out_path}")
