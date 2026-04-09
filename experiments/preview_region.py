"""
Scenario preview — generates three HTML maps per scenario config:

  preview_villages.html — L8 kecamatan + L9 kelurahan + building clusters + shelters
  preview_roads.html    — Road network only (color-coded by highway type)
  preview_region.html   — All layers combined

Cluster naming:
  C_[L9_kode]_N   — real building cluster, N-th largest by pop within L9
  S_[L9_kode]_1   — synthetic cluster (no OSM buildings in that L9)

Usage:
  python -m experiments.preview_region --config configs/merapi_eruption_2023.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from pathlib import Path

from src.utils.logging_setup import setup_logging as _setup_logging
_setup_logging("preview_region")
logger = logging.getLogger("preview_region")

LAYER_STYLES = {
    "l8":       {"color": "#756bb1", "weight": 1.5, "fill_opacity": 0.08,
                 "label": "Admin L8 – Kecamatan"},
    "l9":       {"color": "#2171b5", "weight": 1.0, "fill_opacity": 0.22,
                 "label": "Admin L9 – Kelurahan"},
    "clusters": {"color": "#31a354", "weight": 0.5, "fill_opacity": 0.55,
                 "label": "Building Clusters"},
    "synthetic":{"color": "#fd8d3c", "weight": 1.0, "fill_opacity": 0.45,
                 "label": "Synthetic Clusters"},
    "shelters": {"color": "#d62728", "weight": 1.5, "fill_opacity": 0.50,
                 "label": "Shelters"},
    "roads": {
        "motorway":      {"color": "#a50f15", "weight": 4},
        "trunk":         {"color": "#de2d26", "weight": 3},
        "primary":       {"color": "#fb6a4a", "weight": 2.5},
        "secondary":     {"color": "#fc9272", "weight": 2},
        "tertiary":      {"color": "#fcbba1", "weight": 1.5},
        "unclassified":  {"color": "#bdbdbd", "weight": 1},
        "residential":   {"color": "#bdbdbd", "weight": 1},
        "service":       {"color": "#d9d9d9", "weight": 0.8},
        "living_street": {"color": "#d9d9d9", "weight": 0.8},
        "_default":      {"color": "#cccccc", "weight": 0.8},
        "label": "Road Network",
    },
}


def _circle_radius_px(pop: int) -> int:
    return max(4, min(18, int(math.log10(max(pop, 10)) * 5)))


# ------------------------------------------------------------------ #
# Cluster context: spatial join → naming + admin breadcrumb
# ------------------------------------------------------------------ #

def build_cluster_context(clusters, l9_villages, l8_villages) -> dict:
    """
    For each cluster determine its containing L9 kelurahan and parent L8 kecamatan,
    assign a structured display name, and return a lookup dict.

    Real cluster in L9 33.10.03.2001, 2nd largest → display_name = "C_33.10.03.2001_2"
    Synthetic cluster in L9 33.10.03.2001         → display_name = "S_33.10.03.2001_1"

    Returns {village_id: {"display_name", "l9_kode", "l9_name", "l8_name"}}
    """
    from shapely.geometry import Point
    from shapely.wkt import loads as wkt_loads
    from shapely.strtree import STRtree

    # Lookups by kode string (strip "wilayah_" prefix)
    l8_name_map = {v.village_id.replace("wilayah_", ""): v.name for v in l8_villages}
    l9_name_map = {v.village_id.replace("wilayah_", ""): v.name for v in l9_villages}

    def l8_name_for_l9(l9_kode: str) -> str:
        l8_kode = l9_kode[:8] if len(l9_kode) >= 8 else l9_kode
        return l8_name_map.get(l8_kode, l8_kode)

    real_clusters  = [v for v in clusters if not v.village_id.startswith("artificial_l9_")]
    synth_clusters = [v for v in clusters if v.village_id.startswith("artificial_l9_")]

    # Spatial join: real cluster centroid → L9
    cluster_l9_kode: dict = {}
    if l9_villages and real_clusters:
        l9_with_geom = [
            (v, wkt_loads(v.geometry_wkt))
            for v in l9_villages if v.geometry_wkt
        ]
        if l9_with_geom:
            tree = STRtree([g for _, g in l9_with_geom])
            for v in real_clusters:
                pt = Point(v.centroid_lon, v.centroid_lat)
                for idx in tree.query(pt):
                    lv, g = l9_with_geom[idx]
                    if g.contains(pt):
                        cluster_l9_kode[v.village_id] = lv.village_id.replace("wilayah_", "")
                        break

    # Synthetic clusters already encode their L9 kode in village_id
    for v in synth_clusters:
        # "artificial_l9_wilayah_33.10.03.2001" → "33.10.03.2001"
        l9_kode = v.village_id.replace("artificial_l9_wilayah_", "")
        cluster_l9_kode[v.village_id] = l9_kode

    # Number real clusters within each L9 by population descending
    l9_buckets: dict = defaultdict(list)
    for v in real_clusters:
        l9_kode = cluster_l9_kode.get(v.village_id, "unknown")
        l9_buckets[l9_kode].append(v)

    cluster_number: dict = {}
    for l9_kode, bucket in l9_buckets.items():
        for n, v in enumerate(sorted(bucket, key=lambda x: -x.population), start=1):
            cluster_number[v.village_id] = n

    # Build final context
    ctx = {}
    for v in real_clusters:
        l9_kode = cluster_l9_kode.get(v.village_id, "unknown")
        n = cluster_number.get(v.village_id, 0)
        ctx[v.village_id] = {
            "display_name": f"C_{l9_kode}_{n}",
            "l9_kode": l9_kode,
            "l9_name": l9_name_map.get(l9_kode, l9_kode),
            "l8_name": l8_name_for_l9(l9_kode),
        }
    for v in synth_clusters:
        l9_kode = cluster_l9_kode.get(v.village_id, "unknown")
        ctx[v.village_id] = {
            "display_name": f"S_{l9_kode}_1",
            "l9_kode": l9_kode,
            "l9_name": l9_name_map.get(l9_kode, l9_kode),
            "l8_name": l8_name_for_l9(l9_kode),
        }
    return ctx


# ------------------------------------------------------------------ #
# Map helpers
# ------------------------------------------------------------------ #

def _add_geom_to_group(geometry_wkt: str, color: str, weight: float,
                       fill_opacity: float, tooltip_html: str, group) -> bool:
    import folium
    from shapely.wkt import loads as wkt_loads
    try:
        geom = wkt_loads(geometry_wkt)
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for part in parts:
            coords = [[pt[1], pt[0]] for pt in part.exterior.coords]
            folium.Polygon(
                locations=coords,
                color=color, weight=weight,
                fill=True, fill_color=color, fill_opacity=fill_opacity,
                tooltip=folium.Tooltip(tooltip_html),
            ).add_to(group)
        return True
    except Exception:
        return False


def _add_circle_to_group(lat, lon, pop, color, tooltip_html, group):
    import folium
    folium.CircleMarker(
        location=[lat, lon], radius=_circle_radius_px(pop),
        color=color, fill=True, fill_color=color, fill_opacity=0.7,
        tooltip=folium.Tooltip(tooltip_html),
    ).add_to(group)


# ------------------------------------------------------------------ #
# Map builder
# ------------------------------------------------------------------ #

def _shelter_admin_context(shelters, l9_villages, l8_villages) -> dict:
    """
    Spatial join: shelter centroid → L9 kode → L9 name + L8 name.
    Returns {shelter_id: {"l9_name": ..., "l8_name": ...}}
    """
    from shapely.geometry import Point
    from shapely.wkt import loads as wkt_loads

    l8_name_map = {v.village_id.replace("wilayah_", ""): v.name for v in l8_villages}
    l9_with_geom = [
        (v, wkt_loads(v.geometry_wkt))
        for v in l9_villages if v.geometry_wkt
    ]
    if not l9_with_geom:
        return {}

    from shapely.strtree import STRtree
    tree = STRtree([g for _, g in l9_with_geom])

    ctx = {}
    for s in shelters:
        pt = Point(s.centroid_lon, s.centroid_lat)
        l9_name, l8_name = "", ""
        for idx in tree.query(pt):
            lv, g = l9_with_geom[idx]
            if g.contains(pt):
                l9_kode = lv.village_id.replace("wilayah_", "")
                l9_name = lv.name
                l8_name = l8_name_map.get(l9_kode[:8] if len(l9_kode) >= 8 else l9_kode, "")
                break
        ctx[s.shelter_id] = {"l9_name": l9_name, "l8_name": l8_name}
    return ctx


def build_map(
    l8_villages,
    l9_villages,
    clusters,
    cluster_context: dict,
    shelters,
    nodes,
    edges,
    disaster_lat: float,
    disaster_lon: float,
    disaster_name: str,
    radius_km: float,
    include_villages: bool = True,
    include_shelters: bool = True,
    include_roads: bool = True,
):
    import folium

    # L8 name lookup for L9 tooltip breadcrumb
    l8_name_map = {v.village_id.replace("wilayah_", ""): v.name for v in l8_villages}

    def l8_name_for_l9_kode(kode: str) -> str:
        return l8_name_map.get(kode[:8] if len(kode) >= 8 else kode, "")

    center = [disaster_lat, disaster_lon]
    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")

    folium.Marker(
        location=center,
        popup=f"<b>{disaster_name}</b>",
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
    ).add_to(m)
    folium.Circle(
        location=center, radius=radius_km * 1000,
        color="#cc0000", weight=1.5, fill=False,
        dash_array="6 4", tooltip=f"{radius_km} km radius",
    ).add_to(m)

    real_clusters  = [v for v in clusters if not v.village_id.startswith("artificial_l9_")]
    synth_clusters = [v for v in clusters if v.village_id.startswith("artificial_l9_")]

    # Shelter admin context (spatial join → L9/L8 names) — only if needed
    shelter_ctx = (
        _shelter_admin_context(shelters, l9_villages, l8_villages)
        if include_shelters else {}
    )

    # Node lookup for edge rendering
    node_map = {n.node_id: n for n in nodes} if include_roads else {}

    # ---- Road network ----
    if include_roads:
        grp_roads = folium.FeatureGroup(name=LAYER_STYLES["roads"]["label"], show=True)
        road_styles = LAYER_STYLES["roads"]
        for e in edges:
            src = node_map.get(e.source_id)
            tgt = node_map.get(e.target_id)
            if src is None or tgt is None:
                continue
            rs = road_styles.get(e.highway_type, road_styles["_default"])
            folium.PolyLine(
                locations=[[src.lat, src.lon], [tgt.lat, tgt.lon]],
                color=rs["color"], weight=rs["weight"], opacity=0.7,
                tooltip=f"{e.highway_type} · {e.length_m:.0f} m · {e.max_speed_kmh:.0f} km/h",
            ).add_to(grp_roads)
        grp_roads.add_to(m)

    # ---- Village layers ----
    if include_villages:
        grp_l8  = folium.FeatureGroup(name=LAYER_STYLES["l8"]["label"],        show=True)
        grp_l9  = folium.FeatureGroup(name=LAYER_STYLES["l9"]["label"],        show=True)
        grp_cls = folium.FeatureGroup(name=LAYER_STYLES["clusters"]["label"],  show=True)
        grp_syn = folium.FeatureGroup(name=LAYER_STYLES["synthetic"]["label"], show=bool(synth_clusters))

        s8 = LAYER_STYLES["l8"]
        for v in l8_villages:
            kode = v.village_id.replace("wilayah_", "")
            tip = (
                f"<b>{v.name}</b><br>"
                f"Kode: {kode}<br>"
                f"Area: {v.area_m2/1e6:.3f} km²"
            )
            if v.geometry_wkt:
                _add_geom_to_group(v.geometry_wkt, s8["color"], s8["weight"],
                                   s8["fill_opacity"], tip, grp_l8)
            else:
                _add_circle_to_group(v.centroid_lat, v.centroid_lon, 10,
                                     s8["color"], tip, grp_l8)

        s9 = LAYER_STYLES["l9"]
        for v in l9_villages:
            kode = v.village_id.replace("wilayah_", "")
            l8_name = l8_name_for_l9_kode(kode)
            breadcrumb = f"{v.name}, {l8_name}" if l8_name else v.name
            tip = (
                f"Kode: {kode}<br>"
                f"<b>{breadcrumb}</b><br>"
                f"Area: {v.area_m2/1e6:.3f} km²"
            )
            if v.geometry_wkt:
                _add_geom_to_group(v.geometry_wkt, s9["color"], s9["weight"],
                                   s9["fill_opacity"], tip, grp_l9)
            else:
                _add_circle_to_group(v.centroid_lat, v.centroid_lon, 10,
                                     s9["color"], tip, grp_l9)

        sc = LAYER_STYLES["clusters"]
        for v in real_clusters:
            ctx = cluster_context.get(v.village_id, {})
            dname   = ctx.get("display_name", v.name)
            l9_name = ctx.get("l9_name", "")
            l8_name = ctx.get("l8_name", "")
            admin   = f"{l9_name}, {l8_name}" if l9_name and l8_name else (l9_name or l8_name)
            tip = (
                f"<b>{dname}</b><br>"
                f"Area: {int(v.area_m2):,} m²<br>"
                f"Population: <b>{v.population:,}</b><br>"
                f"<span style='color:#555'>{admin}</span>"
            )
            if v.geometry_wkt:
                _add_geom_to_group(v.geometry_wkt, sc["color"], sc["weight"],
                                   sc["fill_opacity"], tip, grp_cls)
            else:
                _add_circle_to_group(v.centroid_lat, v.centroid_lon,
                                     v.population, sc["color"], tip, grp_cls)

        ss = LAYER_STYLES["synthetic"]
        for v in synth_clusters:
            ctx = cluster_context.get(v.village_id, {})
            dname   = ctx.get("display_name", v.name)
            l9_name = ctx.get("l9_name", "")
            l8_name = ctx.get("l8_name", "")
            admin   = f"{l9_name}, {l8_name}" if l9_name and l8_name else (l9_name or l8_name)
            tip = (
                f"<b>{dname}</b><br>"
                f"<i>Synthetic — no OSM buildings</i><br>"
                f"Area: {int(v.area_m2):,} m²<br>"
                f"Population: <b>{v.population:,}</b> (1 household)<br>"
                f"<span style='color:#555'>{admin}</span>"
            )
            if v.geometry_wkt:
                _add_geom_to_group(v.geometry_wkt, ss["color"], ss["weight"],
                                   ss["fill_opacity"], tip, grp_syn)
            else:
                _add_circle_to_group(v.centroid_lat, v.centroid_lon,
                                     v.population, ss["color"], tip, grp_syn)

        grp_l8.add_to(m)
        grp_l9.add_to(m)
        grp_cls.add_to(m)
        grp_syn.add_to(m)

    # ---- Shelters ----
    if include_shelters:
        grp_shel = folium.FeatureGroup(name=LAYER_STYLES["shelters"]["label"], show=True)
        ss2 = LAYER_STYLES["shelters"]
        for s in shelters:
            sctx = shelter_ctx.get(s.shelter_id, {})
            l9_name = sctx.get("l9_name", "")
            l8_name = sctx.get("l8_name", "")
            admin = f"{l9_name}, {l8_name}" if l9_name and l8_name else (l9_name or l8_name)
            tip = (
                f"<b>{s.name or s.shelter_id}</b><br>"
                f"Type: {s.shelter_type}<br>"
                f"<span style='color:#555'>{admin}</span><br>"
                f"Area: {int(s.area_m2):,} m²<br>"
                f"Capacity: <b>{s.capacity:,}</b> persons"
            )
            if s.geometry_wkt:
                _add_geom_to_group(s.geometry_wkt, ss2["color"], ss2["weight"],
                                   ss2["fill_opacity"], tip, grp_shel)
            else:
                folium.Marker(
                    location=[s.centroid_lat, s.centroid_lon],
                    icon=folium.Icon(color="red", icon="home", prefix="fa"),
                    tooltip=folium.Tooltip(tip),
                ).add_to(grp_shel)
        grp_shel.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ---- Legend ----
    rows = []
    if include_villages:
        total_real_pop = sum(v.population for v in real_clusters)
        total_syn_pop  = sum(v.population for v in synth_clusters)
        rows += [
            f'<tr><td style="padding:3px 6px">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{LAYER_STYLES["l8"]["color"]};margin-right:4px"></span>'
            f'L8 Kecamatan</td>'
            f'<td style="padding:3px 6px;text-align:right">{len(l8_villages):,}</td>'
            f'<td style="padding:3px 6px;text-align:right;color:#aaa">ref</td></tr>',

            f'<tr><td style="padding:3px 6px">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{LAYER_STYLES["l9"]["color"]};margin-right:4px"></span>'
            f'L9 Kelurahan</td>'
            f'<td style="padding:3px 6px;text-align:right">{len(l9_villages):,}</td>'
            f'<td style="padding:3px 6px;text-align:right;color:#aaa">ref</td></tr>',

            f'<tr><td style="padding:3px 6px">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{LAYER_STYLES["clusters"]["color"]};margin-right:4px"></span>'
            f'Building Clusters</td>'
            f'<td style="padding:3px 6px;text-align:right">{len(real_clusters):,}</td>'
            f'<td style="padding:3px 6px;text-align:right">{total_real_pop:,}</td></tr>',
        ]
        if synth_clusters:
            rows.append(
                f'<tr><td style="padding:3px 6px">'
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'background:{LAYER_STYLES["synthetic"]["color"]};margin-right:4px"></span>'
                f'Synthetic Clusters</td>'
                f'<td style="padding:3px 6px;text-align:right">{len(synth_clusters):,}</td>'
                f'<td style="padding:3px 6px;text-align:right">{total_syn_pop:,}</td></tr>'
            )
    if include_shelters:
        rows.append(
            f'<tr><td style="padding:3px 6px">'
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{LAYER_STYLES["shelters"]["color"]};margin-right:4px"></span>'
            f'Shelters</td>'
            f'<td style="padding:3px 6px;text-align:right">{len(shelters):,}</td>'
            f'<td style="padding:3px 6px;text-align:right">{sum(s.capacity for s in shelters):,}</td></tr>'
        )
    # Road type color key — built separately so it can sit below the main table
    road_color_html = ""
    if include_roads:
        road_styles = LAYER_STYLES["roads"]
        _road_type_order = [
            ("motorway",      "Motorway"),
            ("trunk",         "Trunk"),
            ("primary",       "Primary"),
            ("secondary",     "Secondary"),
            ("tertiary",      "Tertiary"),
            ("unclassified",  "Unclassified"),
            ("residential",   "Residential"),
            ("service",       "Service"),
            ("living_street", "Living Street"),
        ]
        # Count edges per type for the present data
        from collections import Counter
        type_counts = Counter(e.highway_type for e in edges)
        type_rows = []
        for rtype, rlabel in _road_type_order:
            if rtype not in type_counts:
                continue
            rs = road_styles.get(rtype, road_styles["_default"])
            w = rs["weight"]
            type_rows.append(
                f'<tr>'
                f'<td style="padding:2px 6px">'
                f'<span style="display:inline-block;width:22px;height:{max(2,w)}px;'
                f'background:{rs["color"]};vertical-align:middle;margin-right:4px;'
                f'border-radius:1px"></span>{rlabel}</td>'
                f'<td style="padding:2px 6px;text-align:right;color:#666">'
                f'{type_counts[rtype]:,}</td>'
                f'</tr>'
            )
        road_color_html = f"""
      <div style="margin-top:8px;padding-top:6px;border-top:1px solid #ddd">
        <b style="font-size:11px">Road Network ({len(edges):,} edges)</b>
        <table style="margin-top:4px;width:100%;border-collapse:collapse;font-size:11px">
          {"".join(type_rows)}
        </table>
      </div>"""

    footnote = "L8/L9 = spatial reference only" + (" · Pop col = capacity for shelters" if include_shelters else "")
    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:12px;
                font-family:sans-serif;min-width:280px">
      <b style="font-size:13px">{disaster_name}</b>
      <table style="margin-top:8px;width:100%;border-collapse:collapse">
        <tr style="background:#f5f5f5">
          <th style="padding:3px 6px;text-align:left">Layer</th>
          <th style="padding:3px 6px;text-align:right">Count</th>
          <th style="padding:3px 6px;text-align:right">Pop/Cap</th>
        </tr>
        {"".join(rows)}
      </table>
      {road_color_html}
      <div style="margin-top:6px;padding-top:6px;border-top:1px solid #ddd;font-size:11px">
        <span style="color:#888">{footnote}</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Generate scenario preview maps (villages, roads, combined)")
    parser.add_argument("--config",   required=True, help="Path to scenario YAML config")
    parser.add_argument("--output",   default=None,  help="Output HTML path")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore OSM cache for cluster extraction")
    args = parser.parse_args()

    from src.config.config_loader import load_config
    cfg = load_config(Path(args.config))

    from src.data.models import RegionOfInterest, RegionType
    from src.data.osm_extractor import OSMExtractor
    from src.data.wilayah_loader import WilayahLoader

    region = RegionOfInterest(
        region_type=RegionType(cfg.region.region_type),
        center=tuple(cfg.region.center) if cfg.region.center else None,
        bbox=tuple(cfg.region.bbox) if cfg.region.bbox else None,
        radius_km=cfg.region.radius_km,
    )
    bbox = region.to_bbox()

    # ---- 1. Wilayah DB reference layers ----
    logger.info("Loading wilayah DB reference layers …")
    try:
        with WilayahLoader() as loader:
            l8_villages = loader.load_villages(bbox=bbox, admin_levels=[8])
            l9_villages = loader.load_villages(bbox=bbox, admin_levels=[9])
        logger.info(f"  L8 kecamatan : {len(l8_villages)}")
        logger.info(f"  L9 kelurahan : {len(l9_villages)}")
    except Exception as exc:
        logger.warning(f"wilayah DB unavailable: {exc} — reference layers will be empty")
        l8_villages, l9_villages = [], []

    # ---- 2. Building clusters ----
    logger.info("Extracting building clusters …")
    extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)
    use_cache = cfg.extraction.use_cached_osm and not args.no_cache
    clusters = extractor.extract_villages(
        region,
        admin_levels=cfg.extraction.village_admin_levels,
        population_density_per_km2=cfg.extraction.village_pop_density,
        max_population_per_village=cfg.extraction.village_max_pop,
        use_cache=use_cache,
        sources=["building_clusters"],
        cluster_eps_m=cfg.extraction.village_cluster_eps_m,
        cluster_min_buildings=cfg.extraction.village_cluster_min_buildings,
        cluster_max_area_km2=cfg.extraction.village_cluster_max_area_km2,
        persons_per_dwelling=cfg.extraction.village_persons_per_dwelling,
        building_persons=cfg.extraction.village_building_persons,
        fill_uncovered_l9=cfg.extraction.village_fill_uncovered_l9,
    )
    real_cls  = [v for v in clusters if not v.village_id.startswith("artificial_l9_")]
    synth_cls = [v for v in clusters if v.village_id.startswith("artificial_l9_")]
    logger.info(f"  Building clusters  : {len(real_cls):,}  pop={sum(v.population for v in real_cls):,}")
    if synth_cls:
        logger.info(f"  Synthetic clusters : {len(synth_cls):,}  pop={sum(v.population for v in synth_cls):,}")

    oversized = [v for v in clusters if v.area_m2 > 50e6]
    if oversized:
        logger.warning(f"  {len(oversized)} clusters > 50 km²:")
        for v in sorted(oversized, key=lambda x: -x.area_m2)[:5]:
            logger.warning(f"    {v.name!r}  {v.area_m2/1e6:.1f} km²")

    # ---- 3. Shelters ----
    logger.info("Extracting shelters …")
    shelters = extractor.extract_shelters(
        region,
        shelter_tags=cfg.extraction.shelter_tags,
        min_area_m2=cfg.extraction.shelter_min_area_m2,
        m2_per_person=cfg.extraction.shelter_m2_per_person,
        use_cache=use_cache,
        cluster_eps_m=cfg.extraction.shelter_cluster_eps_m,
        cluster_min_shelters=cfg.extraction.shelter_cluster_min_shelters,
    )
    from src.data.population_loader import ShelterCapacityLoader
    ShelterCapacityLoader().apply_capacity(
        shelters,
        capacity_csv=cfg.extraction.shelter_capacity_csv,
        m2_per_person=cfg.extraction.m2_per_person,
    )
    logger.info(f"  Shelters: {len(shelters):,}  total capacity={sum(s.capacity for s in shelters):,}")

    # ---- 4. Road network ----
    logger.info("Extracting road network …")
    nodes, edges = extractor.extract_road_network(
        region,
        network_type=cfg.extraction.network_type,
        road_types=cfg.extraction.road_types,
        use_cache=use_cache,
    )
    logger.info(f"  Nodes: {len(nodes):,}  Edges: {len(edges):,}")

    # ---- 5. Build cluster context (names + admin breadcrumb) ----
    logger.info("Building cluster context (spatial join → L9/L8 names) …")
    cluster_context = build_cluster_context(clusters, l9_villages, l8_villages)

    # ---- 6. Build and save maps ----
    base_dir = Path(args.output).parent if args.output else Path(cfg.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        l8_villages=l8_villages,
        l9_villages=l9_villages,
        clusters=clusters,
        cluster_context=cluster_context,
        shelters=shelters,
        nodes=nodes,
        edges=edges,
        disaster_lat=cfg.disaster.lat,
        disaster_lon=cfg.disaster.lon,
        disaster_name=cfg.disaster.name,
        radius_km=cfg.region.radius_km,
    )

    outputs = [
        ("preview_villages.html", dict(include_villages=True,  include_shelters=True,  include_roads=False)),
        ("preview_roads.html",    dict(include_villages=False, include_shelters=False, include_roads=True)),
        ("preview_region.html",   dict(include_villages=True,  include_shelters=True,  include_roads=True)),
    ]

    for fname, flags in outputs:
        out_path = base_dir / fname
        logger.info(f"Building {fname} …")
        m = build_map(**common, **flags)
        m.save(str(out_path))
        logger.info(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
