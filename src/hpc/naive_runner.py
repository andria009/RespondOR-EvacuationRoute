"""
Naive (sequential) runner.
Executes all pipeline stages in a single process with no parallelism.
Serves as the baseline for speedup benchmarking.
"""

import time
import logging
from pathlib import Path

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


def _resolve_hazard_layers(cfg, disaster) -> dict:
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
        if not cfg.skip_inarisk:
            hazard_layers = _resolve_hazard_layers(cfg, disaster)
            inarisk = InaRISKClient(
                batch_size=cfg.extraction.inarisk_batch_size,
                rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
            )
            builder.apply_inarisk_to_edges(
                inarisk=inarisk,
                hazard_layers=hazard_layers,
                aggregation=cfg.routing.hazard_aggregation,
                cache_path=Path(cfg.extraction.inarisk_cache_dir) / "road_risk_cache.json",
                use_cache=cfg.extraction.use_cached_inarisk,
            )
        builder.propagate_poi_risk_to_graph(villages, shelters)
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
            weight_disaster_distance=cfg.routing.weight_disaster_distance,
            max_routes_per_village=cfg.routing.max_routes_per_village,
            disaster_location=(cfg.disaster.lat, cfg.disaster.lon),
        )
        routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
        routes_by_village = optimizer.rank_routes(routes)
        timings["routing"] = time.perf_counter() - t0
        logger.info(f"[{timings['routing']:.2f}s] Computed {len(routes)} candidate routes")

        # ---- Stage 6: Population assignment ----
        t0 = time.perf_counter()
        assigner = PopulationAssigner(method=cfg.routing.assignment_method)
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
        return result, villages, shelters, routes_by_village, timings, G

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
                road_types=cfg.extraction.road_types,
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

        # Apply population / capacity
        pop_loader = PopulationLoader()
        pop_loader.apply_population(
            villages,
            population_csv=cfg.extraction.population_csv,
            density_per_km2=cfg.extraction.village_pop_density,
        )

        cap_loader = ShelterCapacityLoader()
        cap_loader.apply_capacity(
            shelters,
            capacity_csv=cfg.extraction.shelter_capacity_csv,
            m2_per_person=cfg.extraction.m2_per_person,
        )

        return nodes, edges, villages, shelters

    def _apply_risk_scores(self, villages, shelters, disaster, cfg):
        if cfg.skip_inarisk:
            logger.warning("skip_inarisk=true — all risk scores set to 0.0 (InaRISK bypassed)")
            for v in villages:
                v.risk_scores["composite"] = 0.0
            for s in shelters:
                s.risk_scores["composite"] = 0.0
            return

        inarisk = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )
        hazard_layers = _resolve_hazard_layers(cfg, disaster)
        cache_path = Path(cfg.extraction.inarisk_cache_dir) / "poi_risk_cache.json"
        grid_cache_path = Path(cfg.extraction.inarisk_cache_dir) / "road_risk_cache.json"
        use_cache = cfg.extraction.use_cached_inarisk
        if len(hazard_layers) > 1:
            logger.info(
                f"Compound hazard: {', '.join(f'{dt.value}×{w}' for dt, w in hazard_layers.items())}"
                f" [{cfg.routing.hazard_aggregation}]"
            )
            inarisk.enrich_villages_compound(villages, hazard_layers, cfg.routing.hazard_aggregation, cache_path=cache_path, use_cache=use_cache, grid_cache_path=grid_cache_path)
            inarisk.enrich_shelters_compound(shelters, hazard_layers, cfg.routing.hazard_aggregation, cache_path=cache_path, use_cache=use_cache, grid_cache_path=grid_cache_path)
        else:
            inarisk.enrich_villages_with_risk(villages, disaster.disaster_type, cache_path=cache_path, use_cache=use_cache, grid_cache_path=grid_cache_path)
            inarisk.enrich_shelters_with_risk(shelters, disaster.disaster_type, cache_path=cache_path, use_cache=use_cache, grid_cache_path=grid_cache_path)

    def _save_timings(self, timings: dict):
        import json
        out_path = self.output_dir / "timings_naive.json"
        with open(out_path, "w") as f:
            json.dump({"mode": "naive", "timings": timings,
                       "total": sum(timings.values())}, f, indent=2)
        logger.info(f"Timings saved to {out_path}")
