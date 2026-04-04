"""
Prepare GAMA simulation inputs from the extraction and optimization pipeline output.

Reads the optimization result (evacuation_results.csv + optimization_summary.json)
and the OSM-extracted data (villages GeoJSON, shelters GeoJSON, road network JSON)
for a given scenario config, then writes all files needed to run EvacuationModel.gaml
in GAMA Platform.

Outputs (in <cfg.output_dir>/gama_inputs/):
  villages.csv        — village agents: id, name, lat, lon, population, risk
  shelters.csv        — shelter agents: id, name, lat, lon, capacity, type, risk
  routes.csv          — assigned routes: village_id, shelter_id, distance_km,
                          travel_time_min, assigned_population, avg_risk, waypoints_wkt
  roads.shp           — road network shapefile (symlinked/copied from gama_shp/ if available)
  scenario.json       — disaster + simulation parameters for the GAML model

After running this script, open GAMA Platform and load:
  simulation/models/EvacuationModel.gaml
Set the experiment parameter "inputs_dir" to the path printed by this script.

Usage:
  python -m experiments.prepare_gama_inputs --config configs/demak_flood_2024.yaml
  python -m experiments.prepare_gama_inputs --config configs/demak_flood_2024.yaml \\
      --results output/<scenario_id>/evacuation_results.csv
"""

import argparse
import csv
import json
import logging
import math
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_gama_inputs")


# ── Config ─────────────────────────────────────────────────────────────────────

def load_app_config(config_path: Path):
    from src.config.config_loader import load_config
    return load_config(config_path)


def _latest_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        logger.error(f"No '{pattern}' found in {directory}.")
        sys.exit(1)
    return candidates[-1]


# ── Load optimization results ──────────────────────────────────────────────────

def load_evacuation_results(results_csv: Path) -> list:
    """
    Load assignment results from evacuation_results.csv.
    Returns list of dicts with keys:
      village_id, village_name, population, shelter_id, shelter_name,
      shelter_capacity, assigned_population, fraction,
      distance_km, travel_time_min, avg_risk, composite_score
    """
    rows = []
    with open(results_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    logger.info(f"Loaded {len(rows)} assignment rows from {results_csv}")
    return rows


def load_optimization_summary(summary_json: Path) -> dict:
    with open(summary_json) as f:
        return json.load(f)


# ── Load spatial data ──────────────────────────────────────────────────────────

_SOURCE_LABEL = {
    11: "building_cluster",
    10: "place_node",
    9:  "admin_l9",
    8:  "admin_l8",
    7:  "admin_l7",
}


def load_village_geoms(cache_dir: Path) -> dict:
    """Return {village_id: {name, lat, lon, population, area_m2, admin_level, source, risk}} from GeoJSON."""
    from shapely.geometry import shape as _shape
    path = _latest_file(cache_dir, "villages_*.geojson")
    with open(path) as f:
        features = json.load(f).get("features", [])
    out = {}
    for feat in features:
        props = feat.get("properties", {})
        vid = str(props.get("village_id", ""))
        if not vid:
            continue
        try:
            geom = _shape(feat["geometry"])
            c = geom.centroid
            admin_level = int(props.get("admin_level", 9))
            out[vid] = {
                "village_id":  vid,
                "name":        str(props.get("name", ""))[:80],
                "lat":         round(c.y, 8),
                "lon":         round(c.x, 8),
                "population":  int(props.get("population", 0)),
                "area_m2":     float(props.get("area_m2", 0.0)),
                "admin_level": admin_level,
                "source":      _SOURCE_LABEL.get(admin_level, f"admin_l{admin_level}"),
                "risk":        0.0,   # filled later from InaRISK cache if present
            }
        except Exception:
            continue
    logger.info(f"  {len(out)} village geometries from {path.name}")
    return out


def load_shelter_geoms(cache_dir: Path) -> dict:
    """Return {shelter_id: {name, lat, lon, capacity, shelter_type, area_m2, risk}}."""
    from shapely.geometry import shape as _shape
    path = _latest_file(cache_dir, "shelters_*.geojson")
    with open(path) as f:
        features = json.load(f).get("features", [])
    out = {}
    for feat in features:
        props = feat.get("properties", {})
        sid = str(props.get("shelter_id", ""))
        if not sid:
            continue
        try:
            geom = _shape(feat["geometry"])
            c = geom.centroid
            out[sid] = {
                "shelter_id":   sid,
                "name":         str(props.get("name", ""))[:80],
                "lat":          round(c.y, 8),
                "lon":          round(c.x, 8),
                "capacity":     int(props.get("capacity", 0)),
                "shelter_type": str(props.get("shelter_type", "shelter")),
                "area_m2":      float(props.get("area_m2", 0.0)),
                "risk":         0.0,
            }
        except Exception:
            continue
    logger.info(f"  {len(out)} shelter geometries from {path.name}")
    return out


def load_node_coords(cache_dir: Path) -> dict:
    """Return {node_id: (lat, lon)} from network JSON."""
    path = _latest_file(cache_dir, "network_*.json")
    with open(path) as f:
        data = json.load(f)
    coords = {n["id"]: (n["lat"], n["lon"]) for n in data.get("nodes", [])}
    logger.info(f"  {len(coords):,} nodes from {path.name}")
    return coords


def _enrich_with_risk(village_dict: dict, shelter_dict: dict, out_dir: Path, disaster_type: str):
    """
    Pull InaRISK risk scores from the gama_shp risk_cache.json if it exists.
    Fills the 'risk' field in village_dict and shelter_dict in-place.
    """
    risk_cache_path = out_dir.parent / "gama_shp" / "risk_cache.json"
    if not risk_cache_path.exists():
        return

    with open(risk_cache_path) as f:
        cache = json.load(f)

    road_cache    = cache.get(f"roads_{disaster_type}", {})
    shelter_cache = cache.get(f"shelters_{disaster_type}", {})

    def _nearest_cache(lat, lon, lookup):
        """Find closest cached grid point within ~5 km."""
        best_key, best_d = None, 999.0
        lat_r, lon_r = round(lat, 2), round(lon, 2)
        for k in lookup:
            kl, kn = map(float, k.split(","))
            d = math.hypot(kl - lat_r, kn - lon_r)
            if d < best_d:
                best_d, best_key = d, k
        return lookup.get(best_key, 0.0) if best_d < 0.1 else 0.0

    for vid, v in village_dict.items():
        v["risk"] = round(_nearest_cache(v["lat"], v["lon"], road_cache), 4)

    for sid, s in shelter_dict.items():
        key = f"{s['lat']:.6f},{s['lon']:.6f}"
        s["risk"] = round(shelter_cache.get(key, 0.0), 4)


def _node_path_to_waypoints_wkt(node_path: list, node_coords: dict) -> str:
    """
    Convert a list of node IDs to a WKT LINESTRING of (lon lat) pairs.
    GAMA expects lon before lat in geographic coordinates.
    Returns empty string if fewer than 2 valid nodes.
    """
    pts = [(node_coords[n][1], node_coords[n][0])
           for n in node_path if n in node_coords]
    if len(pts) < 2:
        return ""
    coords_str = ", ".join(f"{lon} {lat}" for lon, lat in pts)
    return f"LINESTRING ({coords_str})"


# ── Load node_path from optimization_summary (not stored in CSV) ──────────────

def load_routes_with_paths(cfg, results_rows: list, node_coords: dict) -> list:
    """
    Re-run the routing stage (read-only, uses cache) to recover node_path for each
    assigned route, then merge with assignment rows.

    Returns enriched rows with an added 'waypoints_wkt' column.
    """
    from pathlib import Path as _Path

    # Build index of results rows
    assignment_index = {}  # (village_id, shelter_id) -> row
    for row in results_rows:
        key = (row["village_id"], row["shelter_id"])
        assignment_index[key] = row

    # Try to load routes from the pipeline
    try:
        from src.config.config_loader import load_config
        from src.data.models import (DisasterInput, RegionOfInterest, RegionType,
                                     DisasterType, ExecutionMode)
        from src.data.osm_extractor import OSMExtractor
        from src.data.inarisk_client import InaRISKClient
        from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
        from src.graph.graph_builder import EvacuationGraphBuilder
        from src.routing.heuristic_optimizer import HeuristicOptimizer

        extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)
        region = RegionOfInterest(
            region_type=RegionType(cfg.region.region_type),
            bbox=tuple(cfg.region.bbox) if cfg.region.bbox else None,
            center=tuple(cfg.region.center) if cfg.region.center else None,
            radius_km=cfg.region.radius_km,
        )

        if cfg.preloaded_network_pycgr:
            nodes, edges = extractor.load_network_from_pycgr(cfg.preloaded_network_pycgr)
        elif cfg.preloaded_network_json:
            nodes, edges = extractor.load_network_from_json(cfg.preloaded_network_json)
        else:
            nodes, edges = extractor.extract_road_network(
                region, network_type=cfg.extraction.network_type,
                road_types=cfg.extraction.road_types, use_cache=True,
            )

        if cfg.preloaded_poi_csv:
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
                    use_cache=True,
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
                    region, shelter_tags=cfg.extraction.shelter_tags,
                    min_area_m2=cfg.extraction.shelter_min_area_m2,
                    m2_per_person=cfg.extraction.shelter_m2_per_person, use_cache=True,
                )

        PopulationLoader().apply_population(villages,
            population_csv=cfg.extraction.population_csv,
            density_per_km2=cfg.extraction.village_pop_density)
        ShelterCapacityLoader().apply_capacity(shelters,
            capacity_csv=cfg.extraction.shelter_capacity_csv,
            m2_per_person=cfg.extraction.m2_per_person)

        disaster = DisasterInput(
            location=(cfg.disaster.lat, cfg.disaster.lon),
            disaster_type=DisasterType(cfg.disaster.disaster_type),
            name=cfg.disaster.name, severity=cfg.disaster.severity,
        )
        InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        ).enrich_villages_with_risk(villages, disaster.disaster_type)
        InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        ).enrich_shelters_with_risk(shelters, disaster.disaster_type)

        builder = EvacuationGraphBuilder()
        G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
        builder.attach_pois_to_graph(villages, shelters)
        node_coords.update(builder._node_coords)

        optimizer = HeuristicOptimizer(
            weight_distance=cfg.routing.weight_distance,
            weight_risk=cfg.routing.weight_risk,
            weight_road_quality=cfg.routing.weight_road_quality,
            weight_time=cfg.routing.weight_time,
            max_routes_per_village=cfg.routing.max_routes_per_village,
            max_risk_threshold=cfg.routing.max_route_risk_threshold,
        )
        routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)

        # Build route lookup keyed by (village_id, shelter_id)
        route_lookup = {}
        for route in routes:
            key = (route.village_id, route.shelter_id)
            # Keep the one with lowest composite score (= best route)
            if key not in route_lookup or route.composite_score < route_lookup[key].composite_score:
                route_lookup[key] = route

        for row in results_rows:
            key = (row["village_id"], row["shelter_id"])
            route = route_lookup.get(key)
            if route and route.node_path:
                row["waypoints_wkt"] = _node_path_to_waypoints_wkt(route.node_path, node_coords)
            else:
                row["waypoints_wkt"] = ""

        logger.info(f"  Route paths resolved: "
                    f"{sum(1 for r in results_rows if r.get('waypoints_wkt'))} / {len(results_rows)}")

    except Exception as e:
        logger.warning(f"Could not recover route node_paths: {e}")
        logger.warning("Routes will be written without waypoints (straight-line fallback in GAMA).")
        for row in results_rows:
            row["waypoints_wkt"] = ""

    return results_rows


# ── Writers ────────────────────────────────────────────────────────────────────

def write_villages_csv(village_dict: dict, results_rows: list, cfg, out_path: Path):
    """
    Write villages.csv for GAMA EvacueeAgent initialisation.
    Population is taken from results_rows (sum of assigned_population per village)
    or from the GeoJSON cache. Population fallback is source-aware:
      admin_boundary → village_pop_density
      place_node     → per-tag density from village_place_settings
      building_cluster → village_persons_per_dwelling × estimated building count
    Columns include admin_level and source so GAMA can filter/colour by origin.
    """
    pop_density      = cfg.extraction.village_pop_density
    place_settings   = cfg.extraction.village_place_settings
    per_dwelling     = cfg.extraction.village_persons_per_dwelling
    max_pop          = cfg.extraction.village_max_pop

    # Average place_node density across configured tags (used as fallback)
    _place_densities = [v.get("pop_density", pop_density) for v in place_settings.values()]
    avg_place_density = sum(_place_densities) / len(_place_densities) if _place_densities else pop_density

    # Aggregate assigned population per village from results
    pop_assigned = {}
    for row in results_rows:
        vid = row["village_id"]
        pop_assigned[vid] = pop_assigned.get(vid, 0) + int(float(row.get("assigned_population", 0)))

    fieldnames = ["village_id", "name", "lat", "lon", "population", "admin_level", "source", "risk"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        written = 0
        for vid, v in village_dict.items():
            pop = pop_assigned.get(vid, 0)
            if pop == 0:
                pop = v.get("population", 0)
            if pop == 0 and v.get("area_m2", 0) > 0:
                admin_level = v.get("admin_level", 9)
                if admin_level == 10:       # place_node
                    density = avg_place_density
                elif admin_level == 11:     # building_cluster
                    density = per_dwelling * 200  # ~200 m² per dwelling footprint
                else:                       # admin_boundary
                    density = pop_density
                pop = max(1, min(int(v["area_m2"] / 1e6 * density), max_pop))
            if pop == 0:
                continue   # skip villages with no population
            writer.writerow({
                "village_id":  vid,
                "name":        v["name"],
                "lat":         v["lat"],
                "lon":         v["lon"],
                "population":  pop,
                "admin_level": v.get("admin_level", 9),
                "source":      v.get("source", "admin_boundary"),
                "risk":        v.get("risk", 0.0),
            })
            written += 1
    logger.info(f"  Villages CSV: {written} rows → {out_path}")


def write_shelters_csv(shelter_dict: dict, results_rows: list, out_path: Path):
    """
    Write shelters.csv for GAMA ShelterAgent initialisation.
    """
    # Compute assigned load per shelter from results
    assigned_load = {}
    for row in results_rows:
        sid = row["shelter_id"]
        assigned_load[sid] = assigned_load.get(sid, 0) + int(float(row.get("assigned_population", 0)))

    # Only write shelters that appear in assignments
    active_sids = {row["shelter_id"] for row in results_rows}

    fieldnames = ["shelter_id", "name", "lat", "lon",
                  "capacity", "shelter_type", "assigned_load", "risk"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        written = 0
        for sid in active_sids:
            s = shelter_dict.get(sid, {})
            if not s:
                continue
            writer.writerow({
                "shelter_id":   sid,
                "name":         s.get("name", sid),
                "lat":          s.get("lat", 0.0),
                "lon":          s.get("lon", 0.0),
                "capacity":     s.get("capacity", 0),
                "shelter_type": s.get("shelter_type", "shelter"),
                "assigned_load": assigned_load.get(sid, 0),
                "risk":         s.get("risk", 0.0),
            })
            written += 1
    logger.info(f"  Shelters CSV: {written} rows → {out_path}")


def write_routes_csv(results_rows: list, out_path: Path):
    """
    Write routes.csv for GAMA route assignment.
    Each row represents one village → shelter assignment with road path as WKT.
    """
    fieldnames = [
        "village_id", "shelter_id",
        "distance_km", "travel_time_min",
        "assigned_population", "avg_risk",
        "composite_score", "waypoints_wkt",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_rows:
            if int(float(row.get("assigned_population", 0))) <= 0:
                continue
            writer.writerow({
                "village_id":          row["village_id"],
                "shelter_id":          row["shelter_id"],
                "distance_km":         row.get("distance_km", 0),
                "travel_time_min":     row.get("travel_time_min", 0),
                "assigned_population": row.get("assigned_population", 0),
                "avg_risk":            row.get("avg_risk", 0),
                "composite_score":     row.get("composite_score", 0),
                "waypoints_wkt":       row.get("waypoints_wkt", ""),
            })
    logger.info(f"  Routes CSV: {sum(1 for r in results_rows if int(float(r.get('assigned_population',0)))>0)}"
                f" active rows → {out_path}")


def write_scenario_json(cfg, summary: dict, out_path: Path):
    """
    Write scenario.json with disaster parameters and simulation settings
    for the GAML model to read at init time.
    """
    scenario = {
        "scenario_id":           cfg.scenario_id,
        "disaster_name":         cfg.disaster.name,
        "disaster_type":         cfg.disaster.disaster_type,
        "disaster_lat":          cfg.disaster.lat,
        "disaster_lon":          cfg.disaster.lon,
        "disaster_severity":     cfg.disaster.severity,
        "region_radius_km":      cfg.region.radius_km,

        # Routing params (re-used by GAMA BPR model)
        "bpr_alpha":             cfg.routing.bpr_alpha,
        "bpr_beta":              cfg.routing.bpr_beta,

        # Simulation hints (GAMA can read but user sets in experiment)
        "suggested_max_steps":   500,
        "suggested_time_step_min": 1.0,
        "suggested_n_runs":      5,

        # Village extraction sources used
        "village_sources":       cfg.extraction.village_sources,

        # Summary from optimization
        "total_population":      summary.get("total_population", 0),
        "total_evacuated":       summary.get("total_evacuated", 0),
        "evacuation_ratio":      summary.get("evacuation_ratio", 0.0),
        "avg_route_distance_km": summary.get("avg_route_distance_km", 0.0),
        "avg_route_time_min":    summary.get("avg_route_time_min", 0.0),
    }
    with open(out_path, "w") as f:
        json.dump(scenario, f, indent=2)
    logger.info(f"  Scenario JSON → {out_path}")


def copy_roads_shp(out_dir: Path, cfg_output_dir: Path):
    """
    Copy roads.shp (and companion files) from gama_shp/ into gama_inputs/
    so all GAMA inputs are in one directory.
    """
    shp_dir = cfg_output_dir / "gama_shp"
    if not shp_dir.exists():
        logger.warning("  gama_shp/ not found — run 'python -m experiments.export_shp' first")
        logger.warning("  GAMA will use straight-line movement without road network")
        return None

    copied = []
    for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
        src = shp_dir / f"roads{ext}"
        if src.exists():
            dst = out_dir / f"roads{ext}"
            shutil.copy2(src, dst)
            copied.append(dst.name)

    if copied:
        logger.info(f"  Roads SHP: {', '.join(copied)} → {out_dir}")
        return out_dir / "roads.shp"
    else:
        logger.warning("  roads.shp not found in gama_shp/ — run export_shp first")
        return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prepare GAMA simulation inputs from pipeline optimization output."
    )
    parser.add_argument(
        "--config", default="configs/demak_flood_2024.yaml",
        help="Path to scenario YAML config (default: configs/demak_flood_2024.yaml)",
    )
    parser.add_argument(
        "--results",
        help="Path to evacuation_results.csv (default: <cfg.output_dir>/evacuation_results.csv)",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory (default: <cfg.output_dir>/gama_inputs/)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = load_app_config(config_path)

    cfg_out = Path(cfg.output_dir)
    results_csv  = Path(args.results) if args.results else cfg_out / "evacuation_results.csv"
    summary_json = cfg_out / "optimization_summary.json"
    out_dir      = Path(args.output_dir) if args.output_dir else cfg_out / "gama_inputs"
    cache_dir    = Path(cfg.extraction.osm_cache_dir)

    for p, label in [(results_csv, "evacuation_results.csv"),
                     (summary_json, "optimization_summary.json")]:
        if not p.exists():
            logger.error(f"{label} not found at {p}. Run the main pipeline first:")
            logger.error(f"  python -m src.main --config {config_path}")
            sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"RespondOR — Prepare GAMA Inputs: {cfg.disaster.name}")
    logger.info(f"Config     : {config_path}")
    logger.info(f"Results    : {results_csv}")
    logger.info(f"Output dir : {out_dir}")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading spatial data …")
    village_dict = load_village_geoms(cache_dir)
    shelter_dict = load_shelter_geoms(cache_dir)
    node_coords  = load_node_coords(cache_dir)
    results_rows = load_evacuation_results(results_csv)
    summary      = load_optimization_summary(summary_json)

    # Enrich village/shelter risk from existing InaRISK cache
    disaster_type = cfg.disaster.disaster_type
    _enrich_with_risk(village_dict, shelter_dict, out_dir, disaster_type)

    # ── Recover route node_paths and add waypoints_wkt ────────────────────────
    logger.info("Recovering route paths from pipeline (uses cache — no new API calls) …")
    results_rows = load_routes_with_paths(cfg, results_rows, node_coords)

    # ── Write outputs ─────────────────────────────────────────────────────────
    logger.info("Writing GAMA input files …")
    write_villages_csv(village_dict, results_rows, cfg, out_dir / "villages.csv")
    write_shelters_csv(shelter_dict, results_rows, out_dir / "shelters.csv")
    write_routes_csv(results_rows, out_dir / "routes.csv")
    write_scenario_json(cfg, summary, out_dir / "scenario.json")
    roads_shp = copy_roads_shp(out_dir, cfg_out)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Done. GAMA input files:")
    for f in ["villages.csv", "shelters.csv", "routes.csv", "scenario.json"]:
        p = out_dir / f
        if p.exists():
            kb = p.stat().st_size / 1024
            logger.info(f"  {p}  ({kb:,.0f} KB)")
    if roads_shp and roads_shp.exists():
        logger.info(f"  {roads_shp}  (road network)")

    logger.info("")
    logger.info("To run the simulation in GAMA Platform:")
    logger.info("  1. Open GAMA Platform")
    logger.info(f"  2. Import project: simulation/models/EvacuationModel.gaml")
    logger.info(f'  3. Set experiment parameter "inputs_dir" to:')
    logger.info(f"       {out_dir.resolve()}")
    logger.info("  4. Run experiment: EvacuationExperiment (GUI) or BatchHeadless")


if __name__ == "__main__":
    main()
