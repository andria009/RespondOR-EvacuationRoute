"""
Build legacy / preloaded input files from a fully-extracted scenario cache.

Reads the OSM extraction cache (villages GeoJSON, shelters GeoJSON, road network JSON)
produced by running the main pipeline, then writes all supported preloaded formats so
that subsequent runs can skip the OSM + InaRISK extraction stages entirely.

Village sources supported: admin_boundary, place_nodes, building_clusters.
The admin_level field in the village GeoJSON encodes the origin:
  9 / 8 / 7 = admin boundary polygon
  10        = place_nodes (synthetic circle)
  11        = building_clusters (convex hull)

Outputs (in <output_dir>/legacy_input/):
  network.json              — pre-extracted road network (RespondOR JSON format)
  network.pycgr             — pre-extracted road network (OsmToRoadGraph PYCGR format)
  villages.geojson          — pre-extracted village polygons (all sources merged)
  shelters.geojson          — pre-extracted shelter polygons
  poi.csv                   — legacy RespondOR v1 combined POI CSV (villages + shelters)
  scenario_preloaded.yaml   — ready-to-use config with all preloaded_* fields filled in

Usage:
  python -m experiments.build_legacy_input --config configs/demak_flood_2024.yaml
  python -m experiments.build_legacy_input --config configs/my_scenario.yaml [--output-dir path/to/out]

After running, point a new scenario config at the generated files:
  preloaded_network_json:     output/.../legacy_input/network.json
  preloaded_network_pycgr:    output/.../legacy_input/network.pycgr
  preloaded_villages_geojson: output/.../legacy_input/villages.geojson
  preloaded_shelters_geojson: output/.../legacy_input/shelters.geojson
  preloaded_poi_csv:          output/.../legacy_input/poi.csv
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_legacy_input")

# admin_level sentinel values written by OSMExtractor
_SOURCE_LABEL = {
    11: "building_cluster",
    10: "place_node",
    9:  "admin_l9",
    8:  "admin_l8",
    7:  "admin_l7",
}


# ── Config ────────────────────────────────────────────────────────────────────

def load_app_config(config_path: Path):
    from src.config.config_loader import load_config
    return load_config(config_path)


def _latest_file(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        logger.error(f"No '{pattern}' found in {directory}. Run the main pipeline first.")
        sys.exit(1)
    return candidates[-1]


# ── Network formats ───────────────────────────────────────────────────────────

def build_network_json(cache_dir: Path, out_path: Path):
    """
    Copy the network JSON cache as-is — it is already in the correct format:
      {nodes: [{id, lat, lon}], edges: [{src, tgt, len, hw, spd, bi, lanes}]}
    """
    src = _latest_file(cache_dir, "network_*.json")
    logger.info(f"Network JSON: {src} → {out_path}")
    shutil.copy2(src, out_path)
    with open(out_path) as f:
        d = json.load(f)
    logger.info(f"  {len(d['nodes']):,} nodes, {len(d['edges']):,} edges")


def build_network_pycgr(cache_dir: Path, out_path: Path):
    """
    Write a PYCGR file from the network JSON cache.

    Format (OsmToRoadGraph compatible):
      # comment lines …
      <n_nodes>
      <n_edges>
      <node_id> <lat> <lon>   (one per line)
      <src_id> <tgt_id> <length_m> <highway_type> <max_speed_kmh> <bidirectional(0/1)>
    """
    src = _latest_file(cache_dir, "network_*.json")
    with open(src) as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    logger.info(f"Network PYCGR: {src} → {out_path}")
    with open(out_path, "w") as f:
        f.write("# RespondOR-EvacuationRoute — PYCGR network export\n")
        f.write("# node_id lat lon\n")
        f.write("# src_id tgt_id length_m highway_type max_speed_kmh bidirectional\n")
        f.write(f"{len(nodes)}\n")
        f.write(f"{len(edges)}\n")
        for n in nodes:
            f.write(f"{n['id']} {n['lat']:.8f} {n['lon']:.8f}\n")
        for e in edges:
            bi  = 1 if e.get("bi", True) else 0
            hw  = e.get("hw", "residential")
            spd = e.get("spd", 30.0)
            f.write(f"{e['src']} {e['tgt']} {e['len']:.2f} {hw} {spd:.1f} {bi}\n")

    logger.info(f"  {len(nodes):,} nodes, {len(edges):,} edges")


# ── POI formats ───────────────────────────────────────────────────────────────

def build_villages_geojson(cache_dir: Path, out_path: Path):
    """
    Copy the most recent villages GeoJSON cache.
    Contains all villages from all configured sources (admin_boundary,
    place_nodes, building_clusters) merged and deduplicated.
    """
    src = _latest_file(cache_dir, "villages_*.geojson")
    logger.info(f"Villages GeoJSON: {src} → {out_path}")
    shutil.copy2(src, out_path)
    with open(out_path) as f:
        d = json.load(f)
    features = d.get("features", [])
    # Count by source
    source_counts: dict = {}
    for feat in features:
        al = feat.get("properties", {}).get("admin_level", "?")
        label = _SOURCE_LABEL.get(int(al) if str(al).lstrip("-").isdigit() else -1, f"admin_l{al}")
        source_counts[label] = source_counts.get(label, 0) + 1
    logger.info(f"  {len(features)} villages: " +
                ", ".join(f"{v} {k}" for k, v in sorted(source_counts.items())))


def build_shelters_geojson(cache_dir: Path, out_path: Path):
    """Copy the most recent shelters GeoJSON cache."""
    src = _latest_file(cache_dir, "shelters_*.geojson")
    logger.info(f"Shelters GeoJSON: {src} → {out_path}")
    shutil.copy2(src, out_path)
    with open(out_path) as f:
        d = json.load(f)
    n = len(d.get("features", []))
    logger.info(f"  {n} shelter features")


def build_poi_csv(cache_dir: Path, out_path: Path, cfg):
    """
    Build a legacy RespondOR v1 POI CSV from the village and shelter GeoJSON caches.

    Output columns:
      name, type, latitude, longitude, population, capacity, area_m2, admin_level, source, id

    Population for villages is taken from the cache (already computed using
    per-place-tag density / per-building-type occupancy during extraction).
    Fallback uses place_settings density for place_nodes villages (admin_level=10),
    building_persons default for building_cluster villages (admin_level=11),
    and village_pop_density for admin boundary villages.

    Villages use type='village'; shelters use their shelter_type field.
    """
    from shapely.geometry import shape as _shape

    vsrc = _latest_file(cache_dir, "villages_*.geojson")
    with open(vsrc) as f:
        vdata = json.load(f)

    ssrc = _latest_file(cache_dir, "shelters_*.geojson")
    with open(ssrc) as f:
        sdata = json.load(f)

    place_settings       = cfg.extraction.village_place_settings      # {tag: {radius_m, pop_density}}
    default_pop_density  = cfg.extraction.village_pop_density
    persons_per_dwelling = cfg.extraction.village_persons_per_dwelling
    max_pop              = cfg.extraction.village_max_pop
    m2_per_person        = cfg.extraction.shelter_m2_per_person

    rows = []

    for feat in vdata.get("features", []):
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        try:
            g = _shape(geom)
            lat, lon = g.centroid.y, g.centroid.x
        except Exception:
            continue

        area_m2    = float(props.get("area_m2", 0.0))
        admin_level = int(props.get("admin_level", 9))
        pop        = int(props.get("population", 0))
        source     = _SOURCE_LABEL.get(admin_level, f"admin_l{admin_level}")

        # Fallback population when cache value is 0 (shouldn't normally happen)
        if pop == 0 and area_m2 > 0:
            if admin_level == 10:
                # place_node: use average density across configured place tags
                densities = [v.get("pop_density", default_pop_density)
                             for v in place_settings.values()]
                density = sum(densities) / len(densities) if densities else default_pop_density
            elif admin_level == 11:
                # building_cluster: area × persons_per_dwelling / avg_building_area
                density = persons_per_dwelling * 200  # ~200 m² per dwelling footprint
            else:
                density = default_pop_density
            pop = max(1, min(int((area_m2 / 1e6) * density), max_pop))
        if pop == 0:
            pop = 100

        rows.append({
            "id":          str(props.get("village_id", "")),
            "name":        str(props.get("name", "village"))[:80],
            "type":        "village",
            "latitude":    round(lat, 8),
            "longitude":   round(lon, 8),
            "population":  pop,
            "capacity":    0,
            "area_m2":     round(area_m2, 1),
            "admin_level": admin_level,
            "source":      source,
        })

    for feat in sdata.get("features", []):
        props = feat.get("properties", {})
        geom  = feat.get("geometry", {})
        try:
            g = _shape(geom)
            lat, lon = g.centroid.y, g.centroid.x
        except Exception:
            continue

        area_m2 = float(props.get("area_m2", 0.0))
        cap     = int(props.get("capacity", 0))
        if cap == 0 and area_m2 > 0:
            cap = max(10, int(area_m2 / m2_per_person))
        if cap == 0:
            cap = 200

        rows.append({
            "id":          str(props.get("shelter_id", "")),
            "name":        str(props.get("name", "shelter"))[:80],
            "type":        str(props.get("shelter_type", "shelter")),
            "latitude":    round(lat, 8),
            "longitude":   round(lon, 8),
            "population":  0,
            "capacity":    cap,
            "area_m2":     round(area_m2, 1),
            "admin_level": "",
            "source":      "osm_polygon",
        })

    import csv
    fieldnames = ["id", "name", "type", "latitude", "longitude",
                  "population", "capacity", "area_m2", "admin_level", "source"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_v = sum(1 for r in rows if r["type"] == "village")
    n_s = len(rows) - n_v
    # Count by source
    src_counts: dict = {}
    for r in rows:
        if r["type"] == "village":
            src_counts[r["source"]] = src_counts.get(r["source"], 0) + 1
    logger.info(f"POI CSV: {out_path}  ({n_v} villages, {n_s} shelters)")
    logger.info(f"  Village sources: " + ", ".join(f"{v} {k}" for k, v in sorted(src_counts.items())))


# ── Preloaded config ──────────────────────────────────────────────────────────

def build_preloaded_config(cfg, out_dir: Path, config_path: Path, files: dict):
    """
    Write a scenario_preloaded.yaml that is identical to the source config
    but with all preloaded_* fields set to the generated files,
    use_cached_osm set to true, and all village_sources / place_settings /
    building_persons fields preserved so re-runs are consistent.
    """
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not installed — skipping scenario_preloaded.yaml")
        return

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Inject preloaded paths (relative to project root)
    raw["preloaded_network_json"]     = str(files["network_json"])
    raw["preloaded_network_pycgr"]    = str(files["network_pycgr"])
    raw["preloaded_villages_geojson"] = str(files["villages_geojson"])
    raw["preloaded_shelters_geojson"] = str(files["shelters_geojson"])
    raw["preloaded_poi_csv"]          = str(files["poi_csv"])

    # Force cache usage — no re-extraction needed
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
    parser.add_argument(
        "--config", default="configs/demak_flood_2024.yaml",
        help="Path to scenario YAML/JSON config (default: configs/demak_flood_2024.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory (default: <cfg.output_dir>/legacy_input/)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = load_app_config(config_path)

    out_dir   = Path(args.output_dir) if args.output_dir else Path(cfg.output_dir) / "legacy_input"
    cache_dir = Path(cfg.extraction.osm_cache_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"RespondOR — Build Legacy Input: {cfg.disaster.name}")
    logger.info(f"Config    : {config_path}")
    logger.info(f"Cache dir : {cache_dir}")
    logger.info(f"Output    : {out_dir}")
    logger.info(f"Sources   : {cfg.extraction.village_sources}")
    logger.info("=" * 60)

    files = {
        "network_json":      out_dir / "network.json",
        "network_pycgr":     out_dir / "network.pycgr",
        "villages_geojson":  out_dir / "villages.geojson",
        "shelters_geojson":  out_dir / "shelters.geojson",
        "poi_csv":           out_dir / "poi.csv",
    }

    build_network_json(cache_dir, files["network_json"])
    build_network_pycgr(cache_dir, files["network_pycgr"])
    build_villages_geojson(cache_dir, files["villages_geojson"])
    build_shelters_geojson(cache_dir, files["shelters_geojson"])
    build_poi_csv(cache_dir, files["poi_csv"], cfg)
    build_preloaded_config(cfg, out_dir, config_path, files)

    logger.info("")
    logger.info("Done. Generated files:")
    for label, path in files.items():
        if path.exists():
            size_kb = path.stat().st_size / 1024
            logger.info(f"  {path}  ({size_kb:,.0f} KB)")
    preloaded_yaml = out_dir / "scenario_preloaded.yaml"
    if preloaded_yaml.exists():
        logger.info(f"  {preloaded_yaml}")
    logger.info("")
    logger.info("To run with preloaded inputs (skips OSM + InaRISK extraction):")
    logger.info(f"  python -m src.main --config {out_dir}/scenario_preloaded.yaml")
    logger.info("")
    logger.info("Or set individual fields in any config:")
    for key, path in [
        ("preloaded_network_json",     files["network_json"]),
        ("preloaded_network_pycgr",    files["network_pycgr"]),
        ("preloaded_villages_geojson", files["villages_geojson"]),
        ("preloaded_shelters_geojson", files["shelters_geojson"]),
        ("preloaded_poi_csv",          files["poi_csv"]),
    ]:
        logger.info(f"  {key}: {path}")


if __name__ == "__main__":
    main()
