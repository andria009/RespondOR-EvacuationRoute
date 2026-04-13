"""
Prepare GAMA simulation inputs from the extraction and optimization pipeline output.

Reads cached OSM + InaRISK data for a given scenario config, re-runs routing
(from cache — no new API calls), and writes all files needed to run
EvacuationModel.gaml in GAMA Platform.

Outputs (in <cfg.output_dir>/gama_inputs/):
  all_routes.csv   — all ranked routes per village → shelter, with waypoints_wkt
  villages.csv     — village agents: id, name, lat, lon, population, source, risk
  shelters.csv     — shelter agents: id, name, lat, lon, capacity, type, risk
  hazard_grid.csv  — InaRISK hazard scores over the region bbox
  region.shp       — region boundary polygon (for GAMA world shape)
  roads.shp        — road network (copied from gama_shp/ if run export_shp first)
  scenario.json    — disaster + simulation parameters for the GAML model

Usage:
  python -m experiments.prepare_gama_inputs --config configs/banjarnegara_landslide_2021.yaml
"""

import argparse
import csv
import json
import logging
import re
import shutil
import sys
from pathlib import Path

from src.utils.logging_setup import setup_logging as _setup_logging
_setup_logging("prepare_gama_inputs")
logger = logging.getLogger("prepare_gama_inputs")


# ── GAMA CSV sanitisation ──────────────────────────────────────────────────────
# GAMA's matrix(csv_file(...)) fails on two common issues:
#   1. CRLF line endings — always write LF-only (lineterminator='\n')
#   2. Commas inside quoted fields — OSM IDs like ('way', 304363066) contain a
#      comma; GAMA does not parse RFC-4180 quoting, so the row gets split into
#      too many columns and the whole matrix is returned as null.
# Fix: sanitise IDs and names before writing, and open csv writers with LF-only.

_OSM_ID_RE = re.compile(r"\(\s*'?(\w+)'?\s*,\s*(\d+)\s*\)")

def _gama_id(s: str) -> str:
    """Convert an OSM element ID to a GAMA-safe string (no commas/parens).

    ('way', 304363066)  →  way_304363066
    ('node', 12345678)  →  node_12345678
    shelter_cluster_1   →  shelter_cluster_1   (unchanged)
    """
    s = str(s).strip()
    m = _OSM_ID_RE.match(s)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    # Fallback: strip any remaining parens/quotes/commas
    s = re.sub(r"[()'\",]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _gama_name(s: str) -> str:
    """Strip commas and double-quotes from a display name for GAMA CSVs."""
    return str(s).replace(",", " ").replace('"', "'")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_app_config(config_path: Path):
    from src.config.config_loader import load_config
    return load_config(config_path)


def load_optimization_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _node_path_to_waypoints_wkt(node_path: list, node_coords: dict) -> str:
    """Convert a list of node IDs to a WKT LINESTRING (lon lat) for GAMA."""
    pts = [(node_coords[n][1], node_coords[n][0])
           for n in node_path if n in node_coords]
    if len(pts) < 2:
        return ""
    coords_str = ", ".join(f"{lon:.7f} {lat:.7f}" for lon, lat in pts)
    return f"LINESTRING ({coords_str})"


# ── Pipeline re-run (from cache — no new API calls) ───────────────────────────

def run_pipeline_for_inputs(cfg):
    """
    Run extraction + risk scoring + graph build + routing using cached data only.
    Returns (villages, shelters, G, routes_by_village, node_coords).
    No new API calls are made (use_cached_osm=True, use_cached_inarisk=True enforced).
    """
    import time
    from src.data.models import (
        DisasterInput, RegionOfInterest, RegionType, DisasterType, ExecutionMode
    )
    from src.data.osm_extractor import OSMExtractor
    from src.data.inarisk_client import InaRISKClient
    from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
    from src.graph.graph_builder import EvacuationGraphBuilder
    from src.routing.heuristic_optimizer import HeuristicOptimizer

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

    # ── Stage 1: Extract (from cache) ─────────────────────────────────────────
    extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)
    logger.info("Extracting road network (from cache) …")
    if cfg.preloaded_network_pycgr:
        nodes, edges = extractor.load_network_from_pycgr(cfg.preloaded_network_pycgr)
    elif cfg.preloaded_network_json:
        nodes, edges = extractor.load_network_from_json(cfg.preloaded_network_json)
    else:
        nodes, edges = extractor.extract_road_network(
            region,
            network_type=cfg.extraction.network_type,
            road_types=cfg.extraction.road_types,
            use_cache=True,         # always use cache here
        )

    if cfg.preloaded_poi_csv:
        villages, shelters = extractor.load_pois_from_csv(cfg.preloaded_poi_csv)
    else:
        if cfg.preloaded_villages_geojson:
            villages = extractor.load_villages_from_geojson(cfg.preloaded_villages_geojson)
        else:
            logger.info("Extracting villages (from cache) …")
            villages = extractor.extract_villages(
                region,
                admin_levels=cfg.extraction.village_admin_levels,
                population_density_per_km2=cfg.extraction.village_pop_density,
                max_population_per_village=cfg.extraction.village_max_pop,
                use_cache=True,
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
            logger.info("Extracting shelters (from cache) …")
            shelters = extractor.extract_shelters(
                region,
                shelter_tags=cfg.extraction.shelter_tags,
                min_area_m2=cfg.extraction.shelter_min_area_m2,
                m2_per_person=cfg.extraction.shelter_m2_per_person,
                use_cache=True,
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
        m2_per_person=cfg.extraction.m2_per_person,
    )
    logger.info(f"  Loaded: {len(nodes)}N {len(edges)}E {len(villages)}V {len(shelters)}S")

    # ── Stage 2: Risk scoring (from cache) ────────────────────────────────────
    if cfg.skip_inarisk:
        logger.warning("skip_inarisk=true — all risk scores 0.0")
        for v in villages:
            v.risk_scores["composite"] = 0.0
        for s in shelters:
            s.risk_scores["composite"] = 0.0
    else:
        logger.info("Loading InaRISK risk scores (from cache) …")
        inarisk = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )
        cache_path = Path(cfg.extraction.inarisk_cache_dir) / "poi_risk_cache.json"
        grid_cache_path = Path(cfg.extraction.inarisk_cache_dir) / "road_risk_cache.json"
        raw_hazard = cfg.routing.hazard_layers
        if raw_hazard:
            from src.data.models import DisasterType as _DT
            hazard_layers = {}
            for name, weight in raw_hazard.items():
                try:
                    hazard_layers[_DT(name)] = float(weight)
                except ValueError:
                    pass
            if hazard_layers:
                inarisk.enrich_villages_compound(
                    villages, hazard_layers, cfg.routing.hazard_aggregation,
                    cache_path=cache_path, use_cache=True, grid_cache_path=grid_cache_path,
                )
                inarisk.enrich_shelters_compound(
                    shelters, hazard_layers, cfg.routing.hazard_aggregation,
                    cache_path=cache_path, use_cache=True, grid_cache_path=grid_cache_path,
                )
            else:
                raw_hazard = None  # fall through
        if not raw_hazard:
            inarisk.enrich_villages_with_risk(
                villages, disaster.disaster_type,
                cache_path=cache_path, use_cache=True, grid_cache_path=grid_cache_path,
            )
            inarisk.enrich_shelters_with_risk(
                shelters, disaster.disaster_type,
                cache_path=cache_path, use_cache=True, grid_cache_path=grid_cache_path,
            )

    # ── Stage 3: Graph build ───────────────────────────────────────────────────
    logger.info("Building graph …")
    builder = EvacuationGraphBuilder()
    G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
    builder.attach_pois_to_graph(villages, shelters)
    if not cfg.skip_inarisk:
        raw_hazard = cfg.routing.hazard_layers
        if raw_hazard:
            from src.data.models import DisasterType as _DT
            hazard_layers = {_DT(k): float(v) for k, v in raw_hazard.items()
                             if _DT.__members__.get(k)}
        else:
            hazard_layers = {disaster.disaster_type: 1.0}
        inarisk2 = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )
        builder.apply_inarisk_to_edges(
            inarisk=inarisk2,
            hazard_layers=hazard_layers,
            aggregation=cfg.routing.hazard_aggregation,
            cache_path=Path(cfg.extraction.inarisk_cache_dir) / "road_risk_cache.json",
            use_cache=True,
        )
    builder.propagate_poi_risk_to_graph(villages, shelters)
    logger.info(f"  Graph: {G.number_of_nodes()}N {G.number_of_edges()}E")

    # Build node_coords from live graph
    node_coords = {n: (d["lat"], d["lon"])
                   for n, d in G.nodes(data=True) if "lat" in d and "lon" in d}

    # ── Stage 4: Routing ───────────────────────────────────────────────────────
    logger.info("Computing routes (all ranks, from cache) …")
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
    total_routes = sum(len(v) for v in routes_by_village.values())
    logger.info(f"  {total_routes} routes across {len(routes_by_village)} villages")

    return villages, shelters, G, routes_by_village, node_coords


# ── Writers ────────────────────────────────────────────────────────────────────

def write_all_routes_csv(routes_by_village, villages, shelters, node_coords, out_path: Path):
    """
    Write all_routes.csv — all ranked routes (rank 1 = primary, 2 = alt-1, 3 = alt-2, …)
    for each village × shelter pair.  Includes waypoints_wkt for GAMA road-following movement.
    """
    v_map = {v.village_id: v for v in villages}
    s_map = {s.shelter_id: s for s in shelters}

    fieldnames = [
        "village_id", "village_name", "population",
        "rank",
        "shelter_id", "shelter_name", "shelter_capacity",
        "distance_km", "travel_time_min", "avg_risk", "composite_score",
        "n_nodes", "waypoints_wkt",
        "village_lat", "village_lon",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|", lineterminator="\n")
        writer.writeheader()
        total_with_wkt = 0
        for vid, routes in routes_by_village.items():
            v = v_map.get(vid)
            for rank, route in enumerate(routes, start=1):
                s = s_map.get(route.shelter_id)
                wkt = _node_path_to_waypoints_wkt(route.node_path or [], node_coords)
                if wkt:
                    total_with_wkt += 1
                writer.writerow({
                    "village_id":      vid,
                    "village_name":    _gama_name(v.name) if v else "",
                    "population":      v.population if v else 0,
                    "rank":            rank,
                    "shelter_id":      _gama_id(route.shelter_id),
                    "shelter_name":    _gama_name(s.name) if s else "",
                    "shelter_capacity": s.capacity if s else 0,
                    "distance_km":     round(route.total_distance_km, 3),
                    "travel_time_min": round(route.total_time_min, 1),
                    "avg_risk":        round(route.avg_risk_score, 4),
                    "composite_score": round(route.composite_score, 4),
                    "n_nodes":         len(route.node_path) if route.node_path else 0,
                    "waypoints_wkt":   wkt,
                    "village_lat":     round(v.centroid_lat, 8) if v else 0.0,
                    "village_lon":     round(v.centroid_lon, 8) if v else 0.0,
                })
        total_rows = sum(len(r) for r in routes_by_village.values())
        logger.info(f"  all_routes.csv: {total_rows} routes "
                    f"({total_with_wkt} with waypoints) → {out_path}")


_SOURCE_LABEL = {11: "building_cluster", 10: "place_node", 9: "admin_l9", 8: "admin_l8"}


def _best_risk(risk_scores: dict) -> float:
    """Return the composite risk score, or the first available layer score."""
    if not risk_scores:
        return 0.0
    return risk_scores.get("composite", next(iter(risk_scores.values()), 0.0))


def write_villages_csv(villages, out_path: Path):
    """Write villages.csv for GAMA EvacueeAgent initialisation."""
    fieldnames = [
        "village_id", "name", "lat", "lon", "population",
        "area_m2", "admin_level", "source", "risk",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|", lineterminator="\n")
        writer.writeheader()
        written = 0
        for v in villages:
            if v.population <= 0:
                continue
            writer.writerow({
                "village_id":  v.village_id,
                "name":        _gama_name(v.name[:80]),
                "lat":         round(v.centroid_lat, 8),
                "lon":         round(v.centroid_lon, 8),
                "population":  v.population,
                "area_m2":     round(v.area_m2, 1),
                "admin_level": v.admin_level,
                "source":      _SOURCE_LABEL.get(v.admin_level, f"admin_l{v.admin_level}"),
                "risk":        round(_best_risk(v.risk_scores), 4),
            })
            written += 1
    logger.info(f"  villages.csv: {written} rows → {out_path}")


def write_shelters_csv(shelters, out_path: Path):
    """Write shelters.csv for GAMA ShelterAgent initialisation."""
    fieldnames = [
        "shelter_id", "name", "lat", "lon",
        "capacity", "shelter_type", "area_m2", "risk",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|", lineterminator="\n")
        writer.writeheader()
        written = 0
        for s in shelters:
            writer.writerow({
                "shelter_id":   _gama_id(s.shelter_id),
                "name":         _gama_name(s.name[:80]),
                "lat":          round(s.centroid_lat, 8),
                "lon":          round(s.centroid_lon, 8),
                "capacity":     s.capacity,
                "shelter_type": s.shelter_type,
                "area_m2":      round(s.area_m2, 1),
                "risk":         round(_best_risk(s.risk_scores), 4),
            })
            written += 1
    logger.info(f"  shelters.csv: {written} rows → {out_path}")


def write_hazard_grid_csv(cfg, out_path: Path):
    """
    Write hazard_grid.csv from the InaRISK grid cache, clipped to the scenario bbox.
    Format: lat, lon, hazard_type, score
    """
    from src.data.models import RegionOfInterest, RegionType
    cache_path = Path(cfg.extraction.inarisk_cache_dir) / "hazard_grid_cache.json"
    if not cache_path.exists():
        logger.warning(f"  hazard_grid_cache.json not found at {cache_path} — skipping hazard_grid.csv")
        return

    with open(cache_path) as f:
        full_cache = json.load(f)

    region = RegionOfInterest(
        region_type=RegionType(cfg.region.region_type),
        bbox=tuple(cfg.region.bbox) if cfg.region.bbox else None,
        center=tuple(cfg.region.center) if cfg.region.center else None,
        radius_km=cfg.region.radius_km,
    )
    south, west, north, east = region.to_bbox()

    layers = list(cfg.routing.hazard_layers.keys()) if cfg.routing.hazard_layers else [cfg.disaster.disaster_type]

    fieldnames = ["lat", "lon", "hazard_type", "score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="|", lineterminator="\n")
        writer.writeheader()
        total = 0
        for htype in layers:
            if htype not in full_cache:
                continue
            for key, score in full_cache[htype].items():
                try:
                    lat_s, lon_s = key.split(",")
                    lat, lon = float(lat_s), float(lon_s)
                except ValueError:
                    continue
                if not (south <= lat <= north and west <= lon <= east):
                    continue
                writer.writerow({
                    "lat":         lat,
                    "lon":         lon,
                    "hazard_type": htype,
                    "score":       round(score, 4),
                })
                total += 1
    logger.info(f"  hazard_grid.csv: {total} grid points → {out_path}")


def write_region_shp(cfg, out_path: Path):
    """
    Write region.shp — the circular region boundary polygon (WGS84).
    GAMA will use this as the world shape to set up geographic CRS.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        import pyproj
        from shapely.ops import transform as _shapely_transform

        lat = cfg.region.center[0] if cfg.region.center else (
            (cfg.region.bbox[0] + cfg.region.bbox[2]) / 2 if cfg.region.bbox else cfg.disaster.lat
        )
        lon = cfg.region.center[1] if cfg.region.center else (
            (cfg.region.bbox[1] + cfg.region.bbox[3]) / 2 if cfg.region.bbox else cfg.disaster.lon
        )
        radius_km = cfg.region.radius_km or 10.0

        # Project to UTM, buffer, reproject to WGS84
        utm_crs = f"+proj=utm +zone={int((lon + 180) / 6) + 1} +{'south' if lat < 0 else 'north'} +datum=WGS84"
        proj_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
        proj_to_wgs = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform

        center_utm = _shapely_transform(proj_to_utm, Point(lon, lat))
        circle_utm = center_utm.buffer(radius_km * 1000, resolution=64)
        circle_wgs = _shapely_transform(proj_to_wgs, circle_utm)

        gdf = gpd.GeoDataFrame(
            [{
                "scenario_id": cfg.scenario_id,
                "disaster":    cfg.disaster.name,
                "radius_km":   radius_km,
                "center_lat":  lat,
                "center_lon":  lon,
            }],
            geometry=[circle_wgs],
            crs="EPSG:4326",
        )
        gdf.to_file(out_path, driver="ESRI Shapefile")
        logger.info(f"  region.shp → {out_path}")

    except Exception as e:
        logger.warning(f"  Could not write region.shp: {e}")


def copy_gama_shp(out_dir: Path, cfg_output_dir: Path):
    """
    Copy roads.shp, villages.shp, and shelters.shp (+ companion files)
    from gama_shp/ into gama_inputs/ so GAMA can load them as map layers.
    Run 'python -m experiments.export_shp' first to generate gama_shp/.
    """
    shp_dir = cfg_output_dir / "gama_shp"
    if not shp_dir.exists():
        logger.warning("  gama_shp/ not found — run 'python -m experiments.export_shp' first")
        logger.warning("  GAMA will use straight-line movement; village/shelter polygons will be missing")
        return

    for layer in ["roads", "villages", "shelters"]:
        copied = []
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            src = shp_dir / f"{layer}{ext}"
            if src.exists():
                dst = out_dir / f"{layer}{ext}"
                shutil.copy2(src, dst)
                copied.append(dst.name)
        if copied:
            logger.info(f"  {layer}.shp: copied {len(copied)} files → {out_dir}")
        else:
            logger.warning(f"  {layer}.shp not found in gama_shp/ — run export_shp first")


def write_scenario_json(cfg, summary: dict, out_path: Path):
    """Write scenario.json with all parameters GAMA needs at init."""
    scenario = {
        # Disaster
        "scenario_id":              cfg.scenario_id,
        "disaster_name":            cfg.disaster.name,
        "disaster_type":            cfg.disaster.disaster_type,
        "disaster_lat":             cfg.disaster.lat,
        "disaster_lon":             cfg.disaster.lon,
        "disaster_severity":        cfg.disaster.severity,
        "region_radius_km":         cfg.region.radius_km,

        # BPR congestion (used for road speed under load)
        "bpr_alpha":                cfg.routing.bpr_alpha,
        "bpr_beta":                 cfg.routing.bpr_beta,

        # Hazard propagation defaults (user can override in GAMA experiment)
        "hazard_speed_kmh":         2.0,        # km/h; tune per disaster type
        "hazard_propagation_delay_steps": 0,    # steps before hazard starts moving

        # Simulation defaults
        "suggested_max_steps":      1440,       # 24 hours at 1-min steps
        "suggested_time_step_min":  1.0,
        "suggested_n_runs":         5,

        # Available simulation modes (for GAMA experiment parameters)
        "simulation_modes": {
            "1": "random_route",
            "2": "nearest_shelter",
            "3": "nearest_shelter_away_from_hazard",
            "4": "primary_route_rank1",
            "5": "alternative_route_rank2",
            "6": "alternative_route_rank3",
            "7": "random_ranked_route",
        },

        # Routing weights (for reference; used when computing composite scores)
        "weight_distance":          cfg.routing.weight_distance,
        "weight_risk":              cfg.routing.weight_risk,
        "weight_road_quality":      cfg.routing.weight_road_quality,
        "weight_time":              cfg.routing.weight_time,
        "weight_disaster_distance": cfg.routing.weight_disaster_distance,

        # Optimization summary (for reference)
        "total_population":         summary.get("total_population", 0),
        "total_evacuated":          summary.get("total_evacuated", 0),
        "evacuation_ratio":         summary.get("evacuation_ratio", 0.0),
        "avg_route_distance_km":    summary.get("avg_route_distance_km", 0.0),
        "avg_route_time_min":       summary.get("avg_route_time_min", 0.0),
    }
    with open(out_path, "w") as f:
        json.dump(scenario, f, indent=2)
    logger.info(f"  scenario.json → {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare GAMA simulation inputs from pipeline optimization output."
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to scenario YAML config")
    parser.add_argument("--output-dir", "-o",
                        help="Override output directory (default: <cfg.output_dir>/gama_inputs/)")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = load_app_config(config_path)
    cfg_out = Path(cfg.output_dir)
    out_dir = Path(args.output_dir) if args.output_dir else cfg_out / "gama_inputs"

    summary_json = cfg_out / "optimization_summary.json"
    if not summary_json.exists():
        logger.error(f"optimization_summary.json not found at {summary_json}")
        logger.error(f"Run the main pipeline first: python -m src.main --config {config_path}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"RespondOR — Prepare GAMA Inputs: {cfg.disaster.name}")
    logger.info(f"Config     : {config_path}")
    logger.info(f"Output dir : {out_dir}")
    logger.info("=" * 60)

    # ── Run pipeline from cache ────────────────────────────────────────────────
    logger.info("Running extraction + routing from cache (no new API calls) …")
    villages, shelters, G, routes_by_village, node_coords = run_pipeline_for_inputs(cfg)

    # Load optimization summary for stats
    summary = load_optimization_summary(summary_json)

    # ── Write outputs ──────────────────────────────────────────────────────────
    logger.info("Writing GAMA input files …")
    write_all_routes_csv(routes_by_village, villages, shelters, node_coords,
                         out_dir / "all_routes.csv")
    write_villages_csv(villages, out_dir / "villages.csv")
    write_shelters_csv(shelters, out_dir / "shelters.csv")
    write_hazard_grid_csv(cfg, out_dir / "hazard_grid.csv")
    write_region_shp(cfg, out_dir / "region.shp")
    copy_gama_shp(out_dir, cfg_out)
    write_scenario_json(cfg, summary, out_dir / "scenario.json")

    # ── Summary ────────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Done. GAMA input files:")
    for fname in ["all_routes.csv", "villages.csv", "shelters.csv",
                  "hazard_grid.csv", "region.shp",
                  "roads.shp", "villages.shp", "shelters.shp",
                  "scenario.json"]:
        p = out_dir / fname
        if p.exists():
            kb = p.stat().st_size / 1024
            logger.info(f"  {p}  ({kb:,.1f} KB)")

    logger.info("")
    logger.info("To run the simulation in GAMA Platform:")
    logger.info("  1. Open GAMA Platform")
    logger.info("  2. Import project: simulation/models/EvacuationModel.gaml")
    logger.info("  3. Set experiment parameter 'inputs_dir' to:")
    logger.info(f"       {out_dir.resolve()}")
    logger.info("  4. Set 'simulation_mode' (1–7) to select evacuation strategy")
    logger.info("  5. Run the experiment")


if __name__ == "__main__":
    main()
