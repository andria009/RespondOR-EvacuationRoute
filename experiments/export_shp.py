"""
Export disaster scenario data to Shapefiles for GAMA simulation.

Reads all parameters from a scenario YAML/JSON config file.

Outputs (in <output_dir>/gama_shp/):
  villages.shp  — polygons: village_id, name, population, area_m2, admin_lvl, pop_class
  shelters.shp  — polygons: shelter_id, name, capacity, area_m2, shlt_type, cap_class, risk, risk_cls
  roads.shp     — lines:    u, v, name, highway, speed_kmh, capacity, length_m, road_class, oneway, risk, risk_cls
  preview.html  — interactive Folium map with toggleable layers

pop_class / cap_class / road_class: 1–5 (quintile / highway rank)
risk / risk_cls: InaRISK hazard score [0–1] and 1–5 class

Usage:
  python -m experiments.export_shp --config configs/demak_flood_2024.yaml
  python -m experiments.export_shp --config configs/my_scenario.yaml
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, shape

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("export_shp")

CRS = "EPSG:4326"  # WGS84 — standard for GAMA

# ── Road highway rank (1=best, 5=local) ──────────────────────────────────────
ROAD_RANK = {
    "motorway": 1, "trunk": 1,
    "primary": 2,
    "secondary": 3, "tertiary": 3,
    "unclassified": 4, "residential": 4, "service": 4, "living_street": 4,
    "footway": 5, "path": 5, "track": 5, "pedestrian": 5,
}
ROAD_WEIGHTS = {1: 3.5, 2: 2.5, 3: 1.8, 4: 1.2, 5: 0.7}

# ── Colour palettes ───────────────────────────────────────────────────────────
POP_COLORS  = ["#c6dbef", "#6baed6", "#2171b5", "#084594", "#08306b"]  # blue quintiles
CAP_COLORS  = ["#c7e9c0", "#74c476", "#238b45", "#005a32", "#00441b"]  # green quintiles
ROAD_COLORS = {1: "#b22222", 2: "#e06000", 3: "#d4a800", 4: "#888888", 5: "#cccccc"}
RISK_COLORS = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"]  # low→high red

GRID_PRECISION = 2  # decimal degrees for road midpoint dedup (~1.1 km grid)


# ── Config loading ────────────────────────────────────────────────────────────

def load_app_config(config_path: Path):
    from src.config.config_loader import load_config
    return load_config(config_path)


def _region_bbox(cfg) -> tuple:
    """Return (west, south, east, north) from AppConfig region."""
    region = cfg.region
    if region.region_type == "bbox":
        south, west, north, east = region.bbox
    else:
        lat, lon = region.center
        r = region.radius_km
        dlat = r / 111.0
        dlon = r / (111.0 * math.cos(math.radians(lat)))
        south, north = lat - dlat, lat + dlat
        west,  east  = lon - dlon, lon + dlon
    return west, south, east, north


def _disaster_center(cfg) -> tuple:
    """Return (lat, lon) of the disaster event."""
    return cfg.disaster.lat, cfg.disaster.lon


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quintile_class(series: pd.Series) -> pd.Series:
    """Assign 1–5 quintile class; zeros get class 1."""
    out = pd.Series(1, index=series.index)
    nonzero = series[series > 0]
    if nonzero.empty:
        return out
    q = np.percentile(nonzero, [20, 40, 60, 80])
    for i, v in series.items():
        if v <= 0:         out[i] = 1
        elif v <= q[0]:    out[i] = 1
        elif v <= q[1]:    out[i] = 2
        elif v <= q[2]:    out[i] = 3
        elif v <= q[3]:    out[i] = 4
        else:              out[i] = 5
    return out


def _risk_color(score: float) -> str:
    return RISK_COLORS[min(4, int(float(score) * 5))]


def _risk_class(score: float) -> int:
    return min(5, int(float(score) * 5) + 1)


def _parse_speed(raw, highway_type: str, road_types: dict) -> float:
    """Parse OSM maxspeed tag, falling back to road_types config default."""
    if road_types and highway_type in road_types:
        default = float(road_types[highway_type].get("speed_kmh", 30))
    else:
        defaults = {
            "motorway": 100, "trunk": 80, "primary": 60, "secondary": 50,
            "tertiary": 40, "unclassified": 30, "residential": 30,
            "service": 20, "living_street": 15, "footway": 8, "path": 8, "track": 15,
        }
        default = float(defaults.get(highway_type, 30))
    if raw is None or (isinstance(raw, float) and raw != raw):
        return default
    try:
        s = str(raw).replace(" mph", "").replace(" kmh", "").replace("mph", "").strip()
        val = float(s.split(";")[0].split("|")[0])
        return val if val > 0 else default
    except Exception:
        return default


def _latest_geojson(cache_dir: Path, pattern: str) -> Path:
    """Return the most recently modified file matching pattern, or exit."""
    candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not candidates:
        logger.error(f"No {pattern} found in {cache_dir}. Run the main pipeline first.")
        sys.exit(1)
    return candidates[-1]


def _load_geojson_features(path: Path) -> list:
    with open(path) as f:
        return json.load(f).get("features", [])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_villages(cache_dir: Path, pop_density: float) -> gpd.GeoDataFrame:
    logger.info("Loading villages from cache …")
    path = _latest_geojson(cache_dir, "villages_*.geojson")
    logger.info(f"  {path}")

    rows = []
    for feat in _load_geojson_features(path):
        props = feat.get("properties", {})
        geom_raw = feat.get("geometry")
        if not geom_raw:
            continue
        try:
            geom = shape(geom_raw)
        except Exception:
            continue
        rows.append({
            "village_id": str(props.get("village_id", "")),
            "name":       str(props.get("name", ""))[:80],
            "population": int(props.get("population", 0)),
            "area_m2":    float(props.get("area_m2", 0.0)),
            "admin_lvl":  int(props.get("admin_level", 9)),
            "geometry":   geom,
        })

    gdf = gpd.GeoDataFrame(rows, crs=CRS)

    # Fill missing population from area
    zero = gdf["population"] == 0
    if zero.any():
        estimated = ((gdf.loc[zero, "area_m2"] / 1e6) * pop_density).clip(upper=50000).astype(int)
        gdf.loc[zero, "population"] = estimated.where(estimated > 0, 500)
    gdf.loc[gdf["population"] == 0, "population"] = 500

    gdf["pop_class"] = _quintile_class(gdf["population"]).astype(int)
    logger.info(f"  {len(gdf)} villages | pop {gdf['population'].min():,}–{gdf['population'].max():,}")
    return gdf


def load_shelters(cache_dir: Path, m2_per_person: float) -> gpd.GeoDataFrame:
    logger.info("Loading shelters from cache …")
    path = _latest_geojson(cache_dir, "shelters_*.geojson")
    logger.info(f"  {path}")

    rows = []
    for feat in _load_geojson_features(path):
        props = feat.get("properties", {})
        geom_raw = feat.get("geometry")
        if not geom_raw:
            continue
        try:
            geom = shape(geom_raw)
        except Exception:
            continue
        rows.append({
            "shelter_id": str(props.get("shelter_id", "")),
            "name":       str(props.get("name", ""))[:80],
            "capacity":   int(props.get("capacity", 0)),
            "area_m2":    float(props.get("area_m2", 0.0)),
            "shlt_type":  str(props.get("shelter_type", "shelter"))[:30],
            "geometry":   geom,
        })

    gdf = gpd.GeoDataFrame(rows, crs=CRS)

    # Fill missing capacity from area
    zero = gdf["capacity"] == 0
    if zero.any():
        estimated = (gdf.loc[zero, "area_m2"] / m2_per_person).clip(lower=10).astype(int)
        gdf.loc[zero, "capacity"] = estimated.where(estimated > 10, 200)
    gdf.loc[gdf["capacity"] == 0, "capacity"] = 200

    gdf["cap_class"] = _quintile_class(gdf["capacity"]).astype(int)
    logger.info(f"  {len(gdf)} shelters | capacity {gdf['capacity'].min():,}–{gdf['capacity'].max():,}")
    return gdf


def load_roads(bbox: tuple, osm_cache: Path, road_types: dict,
               network_type: str = "all") -> gpd.GeoDataFrame:
    """Load and filter road network from OSM (uses HTTP cache)."""
    logger.info("Loading road network via osmnx …")
    try:
        import osmnx as ox
        ox.settings.use_cache = True
        ox.settings.cache_folder = str(osm_cache)
        ox.settings.log_console = False
    except ImportError:
        logger.error("osmnx not installed: pip install osmnx")
        sys.exit(1)

    west, south, east, north = bbox
    G = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type=network_type,
        retain_all=True,
        simplify=True,
    )
    _, edges_gdf = ox.graph_to_gdfs(G)
    edges_gdf = edges_gdf.reset_index()

    rows, skipped = [], 0
    for _, row in edges_gdf.iterrows():
        hw = row.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        hw = str(hw).strip()

        if road_types and hw not in road_types:
            skipped += 1
            continue

        spd = _parse_speed(row.get("maxspeed"), hw, road_types)
        cap = int(road_types[hw].get("capacity_veh_h", 0)) if road_types and hw in road_types else 0

        rows.append({
            "u":          int(row["u"]),
            "v":          int(row["v"]),
            "name":       str(row.get("name", ""))[:80] if row.get("name") else "",
            "highway":    hw[:30],
            "speed_kmh":  spd,
            "capacity":   cap,
            "length_m":   round(float(row.get("length", 0.0)), 1),
            "road_class": ROAD_RANK.get(hw, 4),
            "oneway":     bool(row.get("oneway", False)),
            "geometry":   row.geometry,
        })

    gdf = gpd.GeoDataFrame(rows, crs=CRS)
    if skipped:
        logger.info(f"  Skipped {skipped} segments (excluded highway types)")
    hw_types = sorted(set(gdf["highway"]))
    logger.info(f"  {len(gdf):,} segments | {len(hw_types)} types: {hw_types}")
    return gdf


# ── InaRISK enrichment ────────────────────────────────────────────────────────

def enrich_roads_with_risk(roads: gpd.GeoDataFrame, disaster_type: str,
                            batch_size: int, rate_limit_s: float,
                            risk_cache_file: Path) -> gpd.GeoDataFrame:
    """
    Assign InaRISK risk score to each road segment.
    Samples segment midpoints on a ~1.1 km grid to minimise API calls.
    Results are persisted in risk_cache_file.
    """
    try:
        from src.data.inarisk_client import InaRISKClient
        from src.data.models import DisasterType
        dt = DisasterType(disaster_type)
    except Exception as e:
        logger.warning(f"InaRISK unavailable — skipping road risk: {e}")
        roads["risk"] = 0.0
        roads["risk_cls"] = 1
        return roads

    cache = _load_risk_cache(risk_cache_file)
    cache_key = f"roads_{disaster_type}"
    cached = cache.get(cache_key, {})

    grid_keys = []
    for _, row in roads.iterrows():
        mid = row.geometry.interpolate(0.5, normalized=True)
        grid_keys.append(f"{round(mid.y, GRID_PRECISION)},{round(mid.x, GRID_PRECISION)}")

    missing = [k for k in set(grid_keys) if k not in cached]
    logger.info(f"  Road risk: {len(set(grid_keys))} grid cells, {len(missing)} to query …")

    if missing:
        pts = [(float(k.split(",")[0]), float(k.split(",")[1])) for k in missing]
        scores = InaRISKClient(batch_size=batch_size, rate_limit_s=rate_limit_s) \
                     .get_risk_scores(pts, dt)
        cached.update(dict(zip(missing, scores)))
        cache[cache_key] = cached
        _save_risk_cache(risk_cache_file, cache)

    vals = [cached.get(k, 0.0) for k in grid_keys]
    roads["risk"]     = vals
    roads["risk_cls"] = [_risk_class(v) for v in vals]
    logger.info(f"  Road risk range: {min(vals):.3f}–{max(vals):.3f}")
    return roads


def enrich_shelters_with_risk(shelters: gpd.GeoDataFrame, disaster_type: str,
                               batch_size: int, rate_limit_s: float,
                               risk_cache_file: Path) -> gpd.GeoDataFrame:
    """
    Assign InaRISK risk score to each shelter.
    Samples centroid + up to 4 interior points for large shelters, then averages.
    Results are persisted in risk_cache_file.
    """
    try:
        from src.data.inarisk_client import InaRISKClient
        from src.data.models import DisasterType
        dt = DisasterType(disaster_type)
    except Exception as e:
        logger.warning(f"InaRISK unavailable — skipping shelter risk: {e}")
        shelters["risk"] = 0.0
        shelters["risk_cls"] = 1
        return shelters

    cache = _load_risk_cache(risk_cache_file)
    cache_key = f"shelters_{disaster_type}"
    cached = cache.get(cache_key, {})

    point_groups = []
    all_keys = []
    for _, row in shelters.iterrows():
        geom = row.geometry
        c = geom.centroid
        pts = [(c.y, c.x)]
        if row.get("area_m2", 0) > 5000:
            minx, miny, maxx, maxy = geom.bounds
            off_lat = (maxy - miny) * 0.4
            off_lon = (maxx - minx) * 0.4
            interior = [(c.y + off_lat, c.x), (c.y - off_lat, c.x),
                        (c.y, c.x + off_lon), (c.y, c.x - off_lon)]
            pts += [p for p in interior if geom.contains(Point(p[1], p[0]))]
        point_groups.append(pts)
        all_keys += [f"{p[0]:.6f},{p[1]:.6f}" for p in pts]

    missing = [k for k in set(all_keys) if k not in cached]
    logger.info(f"  Shelter risk: {len(shelters)} shelters, {len(missing)} points to query …")

    if missing:
        pts = [(float(k.split(",")[0]), float(k.split(",")[1])) for k in missing]
        scores = InaRISKClient(batch_size=batch_size, rate_limit_s=rate_limit_s) \
                     .get_risk_scores(pts, dt)
        cached.update(dict(zip(missing, scores)))
        cache[cache_key] = cached
        _save_risk_cache(risk_cache_file, cache)

    risks = []
    for pts in point_groups:
        vals = [cached.get(f"{p[0]:.6f},{p[1]:.6f}", 0.0) for p in pts]
        risks.append(sum(vals) / len(vals) if vals else 0.0)

    shelters["risk"]     = risks
    shelters["risk_cls"] = [_risk_class(v) for v in risks]
    logger.info(f"  Shelter risk range: {min(risks):.3f}–{max(risks):.3f}")
    return shelters


def _load_risk_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_risk_cache(path: Path, cache: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, indent=2)
    logger.info(f"  Risk cache → {path}")


# ── Output ────────────────────────────────────────────────────────────────────

def write_shapefiles(villages: gpd.GeoDataFrame, shelters: gpd.GeoDataFrame,
                     roads: gpd.GeoDataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    vcols = ["village_id", "name", "population", "area_m2", "admin_lvl", "pop_class"]
    scols = ["shelter_id", "name", "capacity", "area_m2", "shlt_type", "cap_class"]
    rcols = ["u", "v", "name", "highway", "speed_kmh", "capacity", "length_m", "road_class", "oneway"]
    if "risk" in shelters.columns:
        scols += ["risk", "risk_cls"]
    if "risk" in roads.columns:
        rcols += ["risk", "risk_cls"]

    for label, gdf, cols, fname in [
        ("villages", villages, vcols, "villages.shp"),
        ("shelters", shelters, scols, "shelters.shp"),
        ("roads",    roads,    rcols, "roads.shp"),
    ]:
        p = out_dir / fname
        logger.info(f"Writing {p} …")
        gdf[cols + ["geometry"]].to_file(p)

    logger.info("Shapefiles written.")


def write_preview(villages: gpd.GeoDataFrame, shelters: gpd.GeoDataFrame,
                  roads: gpd.GeoDataFrame, out_dir: Path,
                  center_lat: float, center_lon: float, disaster_name: str):
    try:
        import folium
    except ImportError:
        logger.warning("folium not installed — skipping preview map")
        return

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11,
                   tiles="CartoDB positron")

    # ── Roads – Highway Type ──────────────────────────────────────────────────
    road_layer = folium.FeatureGroup(name="Roads – Highway Type", show=True)
    for _, row in roads.iterrows():
        rc = int(row.get("road_class", 4))
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=ROAD_COLORS[rc], w=ROAD_WEIGHTS[rc]: {
                "color": c, "weight": w, "opacity": 0.75,
            },
            tooltip=f"{row['highway']} · {row['speed_kmh']:.0f} km/h · {row['length_m']:.0f} m",
        ).add_to(road_layer)
    road_layer.add_to(m)

    # ── Villages ──────────────────────────────────────────────────────────────
    village_layer = folium.FeatureGroup(name="Villages", show=True)
    for _, row in villages.iterrows():
        color = POP_COLORS[int(row.get("pop_class", 1)) - 1]
        geom  = row.geometry
        tip   = folium.Tooltip(f"<b>{row['name']}</b><br>Population: {row['population']:,}")
        if geom.geom_type == "Point":
            folium.CircleMarker(
                location=[geom.y, geom.x], radius=6,
                color="#1a4e8a", fill=True, fill_color=color, fill_opacity=0.75,
                tooltip=tip,
            ).add_to(village_layer)
        else:
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda _, c=color: {
                    "fillColor": c, "color": "#1a4e8a", "weight": 1.5, "fillOpacity": 0.55,
                },
                tooltip=tip,
            ).add_to(village_layer)
    village_layer.add_to(m)

    # ── Shelters – Capacity ───────────────────────────────────────────────────
    shelter_layer = folium.FeatureGroup(name="Shelters – Capacity", show=True)
    for _, row in shelters.iterrows():
        color = CAP_COLORS[int(row.get("cap_class", 1)) - 1]
        geom  = row.geometry
        tip   = folium.Tooltip(
            f"<b>{row['name']}</b><br>Type: {row['shlt_type']}<br>Capacity: {row['capacity']:,}"
        )
        if geom.geom_type == "Point":
            folium.CircleMarker(
                location=[geom.y, geom.x], radius=5,
                color="#1a5e2a", fill=True, fill_color=color, fill_opacity=0.8,
                tooltip=tip,
            ).add_to(shelter_layer)
        else:
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda _, c=color: {
                    "fillColor": c, "color": "#1a5e2a", "weight": 1.5, "fillOpacity": 0.65,
                },
                tooltip=tip,
            ).add_to(shelter_layer)
    shelter_layer.add_to(m)

    # ── Roads – Risk (off by default) ─────────────────────────────────────────
    if "risk" in roads.columns:
        risk_road_layer = folium.FeatureGroup(name="Roads – Risk", show=False)
        for _, row in roads.iterrows():
            rc   = int(row.get("road_class", 4))
            risk = float(row.get("risk", 0.0))
            col  = _risk_color(risk)
            folium.GeoJson(
                row.geometry.__geo_interface__,
                style_function=lambda _, c=col, w=ROAD_WEIGHTS[rc]: {
                    "color": c, "weight": w, "opacity": 0.85,
                },
                tooltip=(f"{row['highway']} · {row['speed_kmh']:.0f} km/h "
                         f"· {row['length_m']:.0f} m · risk: {risk:.2f}"),
            ).add_to(risk_road_layer)
        risk_road_layer.add_to(m)

    # ── Shelters – Risk (off by default) ──────────────────────────────────────
    if "risk" in shelters.columns:
        risk_shelter_layer = folium.FeatureGroup(name="Shelters – Risk", show=False)
        for _, row in shelters.iterrows():
            risk = float(row.get("risk", 0.0))
            col  = _risk_color(risk)
            geom = row.geometry
            tip  = folium.Tooltip(
                f"<b>{row['name']}</b><br>Type: {row['shlt_type']}<br>"
                f"Capacity: {row['capacity']:,}<br>Risk: {risk:.2f}"
            )
            if geom.geom_type == "Point":
                folium.CircleMarker(
                    location=[geom.y, geom.x], radius=8,
                    color="#800000", fill=True, fill_color=col, fill_opacity=0.85,
                    tooltip=tip,
                ).add_to(risk_shelter_layer)
            else:
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda _, c=col: {
                        "fillColor": c, "color": "#800000", "weight": 2.0, "fillOpacity": 0.75,
                    },
                    tooltip=tip,
                ).add_to(risk_shelter_layer)
        risk_shelter_layer.add_to(m)

    # ── Disaster epicentre marker ─────────────────────────────────────────────
    folium.Marker(
        location=[center_lat, center_lon],
        tooltip=disaster_name,
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)

    # ── Legend ────────────────────────────────────────────────────────────────
    has_risk = "risk" in roads.columns or "risk" in shelters.columns
    risk_legend = (
        "<br><b>Risk (InaRISK)</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:14px;height:10px;'
            f'margin-right:5px;border:1px solid #555;"></span>'
            f' Class {i+1}{"  (low)" if i==0 else "  (high)" if i==4 else ""}<br>'
            for i, c in enumerate(RISK_COLORS)
        )
    ) if has_risk else ""

    legend_html = (
        '<div style="position:fixed;bottom:40px;left:40px;z-index:9999;background:white;'
        'padding:12px 16px;border-radius:8px;border:1px solid #aaa;font-size:13px;">'
        "<b>Villages (population)</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:14px;height:14px;'
            f'margin-right:5px;border:1px solid #555;"></span> Class {i+1}<br>'
            for i, c in enumerate(POP_COLORS)
        )
        + "<br><b>Shelters (capacity)</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:14px;height:14px;'
            f'margin-right:5px;border:1px solid #555;"></span> Class {i+1}<br>'
            for i, c in enumerate(CAP_COLORS)
        )
        + "<br><b>Roads – Highway Type</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:14px;height:6px;'
            f'margin-right:5px;margin-bottom:4px;"></span> {lbl}<br>'
            for c, lbl in [
                (ROAD_COLORS[1], "Motorway / Trunk"),
                (ROAD_COLORS[2], "Primary"),
                (ROAD_COLORS[3], "Secondary / Tertiary"),
                (ROAD_COLORS[4], "Residential / Service"),
                (ROAD_COLORS[5], "Path / Track"),
            ]
        )
        + risk_legend
        + "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl(collapsed=False).add_to(m)

    out = out_dir / "preview.html"
    m.save(str(out))
    logger.info(f"Preview map saved: {out}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Export disaster scenario to GAMA-ready Shapefiles + preview map."
    )
    parser.add_argument(
        "--config", default="configs/demak_flood_2024.yaml",
        help="Path to scenario YAML/JSON config (default: configs/demak_flood_2024.yaml)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    cfg = load_app_config(config_path)

    out_dir       = Path(cfg.output_dir) / "gama_shp"
    cache_dir     = Path(cfg.extraction.osm_cache_dir)
    osm_http_cache = Path("data/raw/osm_cache/http")
    risk_cache    = out_dir / "risk_cache.json"
    bbox          = _region_bbox(cfg)
    center_lat, center_lon = _disaster_center(cfg)
    from src.data.models import DisasterType as _DisasterType
    disaster_type = _DisasterType(cfg.disaster.disaster_type)
    road_types    = cfg.extraction.road_types
    pop_density   = cfg.extraction.village_pop_density
    m2_per_person = cfg.extraction.shelter_m2_per_person
    batch_size    = cfg.extraction.inarisk_batch_size
    rate_limit_s  = cfg.extraction.inarisk_rate_limit_s

    logger.info("=" * 60)
    logger.info(f"RespondOR — GAMA SHP Export: {cfg.disaster.name}")
    logger.info(f"Config    : {config_path}")
    logger.info(f"Output    : {out_dir}")
    logger.info(f"Disaster  : {disaster_type.value}  severity={cfg.disaster.severity}")
    logger.info("=" * 60)

    villages = load_villages(cache_dir, pop_density)
    shelters = load_shelters(cache_dir, m2_per_person)
    roads    = load_roads(bbox, osm_http_cache, road_types, cfg.extraction.network_type)

    logger.info("Enriching roads with InaRISK risk scores …")
    roads    = enrich_roads_with_risk(roads, disaster_type.value, batch_size, rate_limit_s, risk_cache)
    logger.info("Enriching shelters with InaRISK risk scores …")
    shelters = enrich_shelters_with_risk(shelters, disaster_type.value, batch_size, rate_limit_s, risk_cache)

    write_shapefiles(villages, shelters, roads, out_dir)
    write_preview(villages, shelters, roads, out_dir, center_lat, center_lon, cfg.disaster.name)

    logger.info("")
    logger.info("Done.")
    logger.info(f"  Villages : {len(villages):,}  ({villages['population'].sum():,} population)")
    logger.info(f"  Shelters : {len(shelters):,}  ({shelters['capacity'].sum():,} capacity)")
    if "risk" in shelters.columns:
        logger.info(f"             risk {shelters['risk'].min():.2f}–{shelters['risk'].max():.2f}")
    logger.info(f"  Roads    : {len(roads):,} segments ({len(set(roads['highway']))} types)")
    if "risk" in roads.columns:
        logger.info(f"             risk {roads['risk'].min():.3f}–{roads['risk'].max():.3f}")
    logger.info(f"  Output   : {out_dir}/")
    logger.info("")
    logger.info("SHP attributes for GAMA:")
    logger.info("  villages.shp — village_id, name, population, area_m2, admin_lvl, pop_class")
    logger.info("  shelters.shp — shelter_id, name, capacity, area_m2, shlt_type, cap_class, risk, risk_cls")
    logger.info("  roads.shp    — u, v, name, highway, speed_kmh, capacity, length_m, road_class, oneway, risk, risk_cls")


if __name__ == "__main__":
    main()
