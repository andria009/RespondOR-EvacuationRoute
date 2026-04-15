"""
Build legacy / preloaded input files from a fully-extracted scenario cache.

Calls OSMExtractor with the same parameters as the main pipeline so the
exported files exactly match what the main pipeline would produce — same road
network, same village sources/clustering, same shelter clustering.

Outputs (in <output_dir>/legacy_input/):
  network.json              — road network (RespondOR JSON format)
  network.pycgr             — road network (OsmToRoadGraph PYCGR format)
  villages.geojson          — village polygons (same sources + clustering as pipeline)
  shelters.geojson          — shelter polygons (same clustering as pipeline)
  poi.csv                   — combined POI CSV (villages + shelters)
  scenario_preloaded.yaml   — ready-to-use config with all preloaded_* fields filled in

Usage:
  python -m experiments.build_legacy_input --config configs/banjarnegara_landslide_2021.yaml
"""

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

from src.utils.logging_setup import setup_logging as _setup_logging
_setup_logging("build_legacy_input")
logger = logging.getLogger("build_legacy_input")

_SOURCE_LABEL = {
    11: "building_cluster",
    10: "place_node",
    9:  "admin_l9",
    8:  "admin_l8",
    7:  "admin_l7",
}


# ── Config ────────────────────────────────────────────────────────────────────

def _load_config(config_path: Path):
    from src.config.config_loader import load_config
    return load_config(config_path)


def _build_region(cfg):
    from src.data.models import RegionOfInterest, RegionType
    return RegionOfInterest(
        region_type=RegionType(cfg.region.region_type),
        bbox=tuple(cfg.region.bbox) if cfg.region.bbox else None,
        center=tuple(cfg.region.center) if cfg.region.center else None,
        radius_km=cfg.region.radius_km,
    )


# ── Extraction (mirrors main pipeline) ────────────────────────────────────────

def extract_all(cfg, region):
    """
    Run OSMExtractor with the same parameters as the main pipeline.
    All steps use use_cache=True so no new API/HTTP calls are made.
    Returns (nodes, edges, villages, shelters).
    """
    from src.data.osm_extractor import OSMExtractor
    from src.data.population_loader import PopulationLoader, ShelterCapacityLoader

    extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)

    # Road network
    nodes, edges = extractor.extract_road_network(
        region,
        network_type=cfg.extraction.network_type,
        road_types=cfg.extraction.road_types,
        use_cache=cfg.extraction.use_cached_osm,
    )
    logger.info(f"Network: {len(nodes):,} nodes, {len(edges):,} edges")

    # Villages — same sources + clustering as main pipeline
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

    # Shelters — same clustering as main pipeline
    shelters = extractor.extract_shelters(
        region,
        shelter_tags=cfg.extraction.shelter_tags,
        min_area_m2=cfg.extraction.shelter_min_area_m2,
        m2_per_person=cfg.extraction.shelter_m2_per_person,
        use_cache=cfg.extraction.use_cached_osm,
        cluster_eps_m=cfg.extraction.shelter_cluster_eps_m,
        cluster_min_shelters=cfg.extraction.shelter_cluster_min_shelters,
    )

    # Apply population / capacity (same as main pipeline)
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

    # Count villages by source
    from collections import Counter
    src_counts = Counter(_SOURCE_LABEL.get(v.admin_level, f"admin_l{v.admin_level}")
                         for v in villages)
    logger.info(f"Villages: {len(villages):,} — " +
                ", ".join(f"{n} {s}" for s, n in sorted(src_counts.items())))
    logger.info(f"Shelters: {len(shelters):,} (clustered)")

    return nodes, edges, villages, shelters


# ── Network serialisation ─────────────────────────────────────────────────────

def write_network_json(nodes, edges, out_path: Path):
    """Serialise nodes/edges to RespondOR JSON format."""
    data = {
        "nodes": [{"id": n.node_id, "lat": n.lat, "lon": n.lon} for n in nodes],
        "edges": [
            {
                "src":    e.source_id,
                "tgt":    e.target_id,
                "len":    round(e.length_m, 2),
                "hw":     e.highway_type,
                "spd":    round(e.max_speed_kmh, 1),
                "bi":     e.bidirectional,
                "lanes":  e.lanes,
            }
            for e in edges
        ],
    }
    with open(out_path, "w") as f:
        json.dump(data, f)
    logger.info(f"Network JSON: {out_path}  ({len(nodes):,}N, {len(edges):,}E)")


def write_network_pycgr(nodes, edges, out_path: Path):
    """Serialise nodes/edges to OsmToRoadGraph PYCGR format."""
    with open(out_path, "w") as f:
        f.write("# RespondOR-EvacuationRoute — PYCGR network export\n")
        f.write("# node_id lat lon\n")
        f.write("# src_id tgt_id length_m highway_type max_speed_kmh bidirectional\n")
        f.write(f"{len(nodes)}\n")
        f.write(f"{len(edges)}\n")
        for n in nodes:
            f.write(f"{n.node_id} {n.lat:.8f} {n.lon:.8f}\n")
        for e in edges:
            bi = 1 if e.bidirectional else 0
            f.write(f"{e.source_id} {e.target_id} {e.length_m:.2f} "
                    f"{e.highway_type} {e.max_speed_kmh:.1f} {bi}\n")
    logger.info(f"Network PYCGR: {out_path}  ({len(nodes):,}N, {len(edges):,}E)")


# ── POI serialisation ─────────────────────────────────────────────────────────

def _wkt_to_geojson_geom(wkt: str):
    from shapely import wkt as shapely_wkt
    from shapely.geometry import mapping
    try:
        return mapping(shapely_wkt.loads(wkt))
    except Exception:
        return None


def _village_to_feature(v) -> dict:
    geom = _wkt_to_geojson_geom(v.geometry_wkt) if v.geometry_wkt else None
    return {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "village_id":   str(v.village_id),
            "name":         v.name or "",
            "population":   v.population,
            "area_m2":      round(v.area_m2, 1),
            "admin_level":  v.admin_level,
            "centroid_lat": round(v.centroid_lat, 8),
            "centroid_lon": round(v.centroid_lon, 8),
        },
    }


def _shelter_to_feature(s) -> dict:
    geom = _wkt_to_geojson_geom(s.geometry_wkt) if s.geometry_wkt else None
    return {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "shelter_id":   str(s.shelter_id),
            "name":         s.name or "",
            "capacity":     s.capacity,
            "shelter_type": s.shelter_type or "shelter",
            "area_m2":      round(s.area_m2, 1),
            "centroid_lat": round(s.centroid_lat, 8),
            "centroid_lon": round(s.centroid_lon, 8),
        },
    }


def write_villages_geojson(villages, out_path: Path):
    fc = {"type": "FeatureCollection", "features": [_village_to_feature(v) for v in villages]}
    with open(out_path, "w") as f:
        json.dump(fc, f)
    logger.info(f"Villages GeoJSON: {out_path}  ({len(villages):,} features)")


def write_shelters_geojson(shelters, out_path: Path):
    fc = {"type": "FeatureCollection", "features": [_shelter_to_feature(s) for s in shelters]}
    with open(out_path, "w") as f:
        json.dump(fc, f)
    logger.info(f"Shelters GeoJSON: {out_path}  ({len(shelters):,} features)")


def write_poi_csv(villages, shelters, out_path: Path):
    """Combined POI CSV (villages + shelters) in legacy RespondOR v1 format."""
    fieldnames = ["id", "name", "type", "latitude", "longitude",
                  "population", "capacity", "area_m2", "admin_level", "source"]
    rows = []

    for v in villages:
        rows.append({
            "id":          str(v.village_id),
            "name":        (v.name or "village")[:80],
            "type":        "village",
            "latitude":    round(v.centroid_lat, 8),
            "longitude":   round(v.centroid_lon, 8),
            "population":  v.population,
            "capacity":    0,
            "area_m2":     round(v.area_m2, 1),
            "admin_level": v.admin_level,
            "source":      _SOURCE_LABEL.get(v.admin_level, f"admin_l{v.admin_level}"),
        })

    for s in shelters:
        rows.append({
            "id":          str(s.shelter_id),
            "name":        (s.name or "shelter")[:80],
            "type":        s.shelter_type or "shelter",
            "latitude":    round(s.centroid_lat, 8),
            "longitude":   round(s.centroid_lon, 8),
            "population":  0,
            "capacity":    s.capacity,
            "area_m2":     round(s.area_m2, 1),
            "admin_level": "",
            "source":      "osm_polygon",
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_v = len(villages)
    n_s = len(shelters)
    logger.info(f"POI CSV: {out_path}  ({n_v} villages, {n_s} shelters)")


# ── Preloaded config ──────────────────────────────────────────────────────────

def write_preloaded_config(cfg, config_path: Path, out_dir: Path, files: dict):
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — skipping scenario_preloaded.yaml")
        return

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    raw["preloaded_network_json"]     = str(files["network_json"])
    raw["preloaded_network_pycgr"]    = str(files["network_pycgr"])
    raw["preloaded_villages_geojson"] = str(files["villages_geojson"])
    raw["preloaded_shelters_geojson"] = str(files["shelters_geojson"])
    raw["preloaded_poi_csv"]          = str(files["poi_csv"])

    if "extraction" not in raw:
        raw["extraction"] = {}
    raw["extraction"]["use_cached_osm"] = True

    out_path = out_dir / "scenario_preloaded.yaml"
    with open(out_path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    logger.info(f"Preloaded config: {out_path}")
    logger.info(f"  Use with: python -m src.main --config {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build legacy/preloaded input files from a scenario OSM cache."
    )
    parser.add_argument("--config", required=True, help="Path to scenario YAML config")
    parser.add_argument("--output-dir", help="Override output directory")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = _load_config(config_path)
    region = _build_region(cfg)

    out_dir = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir) / "legacy_input"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"RespondOR — Build Legacy Input: {cfg.disaster.name}")
    logger.info(f"Config    : {config_path}")
    logger.info(f"Cache dir : {cfg.extraction.osm_cache_dir}")
    logger.info(f"Output    : {out_dir}")
    logger.info(f"Sources   : {cfg.extraction.village_sources}")
    logger.info("=" * 60)

    nodes, edges, villages, shelters = extract_all(cfg, region)

    files = {
        "network_json":     out_dir / "network.json",
        "network_pycgr":    out_dir / "network.pycgr",
        "villages_geojson": out_dir / "villages.geojson",
        "shelters_geojson": out_dir / "shelters.geojson",
        "poi_csv":          out_dir / "poi.csv",
    }

    write_network_json(nodes, edges, files["network_json"])
    write_network_pycgr(nodes, edges, files["network_pycgr"])
    write_villages_geojson(villages, files["villages_geojson"])
    write_shelters_geojson(shelters, files["shelters_geojson"])
    write_poi_csv(villages, shelters, files["poi_csv"])
    write_preloaded_config(cfg, config_path, out_dir, files)

    logger.info("")
    logger.info("Done. Generated files:")
    for path in files.values():
        if path.exists():
            logger.info(f"  {path}  ({path.stat().st_size / 1024:,.0f} KB)")
    logger.info(f"  {out_dir / 'scenario_preloaded.yaml'}")
    logger.info("")
    logger.info("To run with preloaded inputs (skips OSM + InaRISK extraction):")
    logger.info(f"  python -m src.main --config {out_dir}/scenario_preloaded.yaml")


if __name__ == "__main__":
    main()
