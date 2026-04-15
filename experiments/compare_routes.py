"""
Run the evacuation pipeline in both OSM and Legacy input modes,
visualize the top evacuation routes for each, and produce a comparison.

OSM mode   — extracts road network + villages + shelters from OSM cache
Legacy mode — loads preloaded network.json + poi.csv (point-only village/shelter data)

Outputs (in output/<scenario_id>/route_comparison/):
  routes_osm.html        — interactive map: top routes, OSM input
  routes_legacy.html     — interactive map: top routes, legacy input
  routes_comparison.html — both modes on one map, toggle by layer
  comparison_summary.json — numeric metrics side-by-side

Usage:
  python -m experiments.compare_routes
  python -m experiments.compare_routes --osm-config   configs/demak_flood_2024.yaml
                                        --legacy-config output/<scenario_id>/legacy_input/scenario_preloaded.yaml
  python -m experiments.compare_routes --top-n 15
"""

import argparse
import json
import logging
import time
from pathlib import Path

from src.utils.logging_setup import setup_logging as _setup_logging
from src.visualization.visualizer import _quintile_class_list
_setup_logging("compare_routes")
logger = logging.getLogger("compare_routes")

# ── Colour palettes (matching export_shp preview design) ─────────────────────
RANK_COLORS   = ["#1a9641", "#fdae61", "#d7191c"]   # route rank 1/2/3
OSM_COLOR     = "#1a4e8a"   # blue border  – OSM villages
LEGACY_COLOR  = "#5e3d8a"   # purple border – legacy villages
POP_COLORS    = ["#c6dbef", "#6baed6", "#2171b5", "#084594", "#08306b"]  # blue quintiles
CAP_COLORS    = ["#c7e9c0", "#74c476", "#238b45", "#005a32", "#00441b"]  # green quintiles


def _load_polygon_geoms(path: "Path", id_field: str) -> dict:
    """Load {id: shapely_geom} from a GeoJSON file."""
    import json as _json
    from shapely.geometry import shape as _shape
    geoms = {}
    try:
        with open(path) as f:
            for feat in _json.load(f).get("features", []):
                fid = str(feat.get("properties", {}).get(id_field, ""))
                g = feat.get("geometry")
                if fid and g:
                    try:
                        geoms[fid] = _shape(g)
                    except Exception:
                        pass
    except Exception:
        pass
    return geoms


def _latest_cache(cache_dir: "Path", pattern: str):
    """Return most recently modified file matching pattern, or None."""
    candidates = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def _route_coords(route, node_coords: dict) -> list:
    return [node_coords[nid] for nid in route.node_path if nid in node_coords]


def _select_top_routes(routes_by_village, villages, top_n: int):
    v_map = {v.village_id: v for v in villages}
    candidates = [(v_map[vid], routes)
                  for vid, routes in routes_by_village.items()
                  if routes and vid in v_map]
    candidates.sort(key=lambda x: x[0].population, reverse=True)
    return candidates[:top_n]


# ── Pipeline runner ───────────────────────────────────────────────────────────

def run_pipeline(config_path: Path, label: str):
    """
    Run the full evacuation pipeline (naive mode) and return all artefacts.
    Returns dict with keys: villages, shelters, routes_by_village, result,
                             node_coords, timings, cfg
    """
    from src.config.config_loader import load_config
    from src.data.models import DisasterInput, RegionOfInterest, RegionType, DisasterType, ExecutionMode
    from src.data.osm_extractor import OSMExtractor
    from src.data.inarisk_client import InaRISKClient
    from src.data.population_loader import PopulationLoader, ShelterCapacityLoader
    from src.graph.graph_builder import EvacuationGraphBuilder
    from src.routing.heuristic_optimizer import HeuristicOptimizer
    from src.routing.assignment import PopulationAssigner

    logger.info(f"{'='*60}")
    logger.info(f"[{label}] Loading config: {config_path}")
    cfg = load_config(config_path)
    timings = {}

    # ── Build disaster / region objects ──────────────────────────────────────
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

    # ── Stage 1: Extract ──────────────────────────────────────────────────────
    logger.info(f"[{label}] Stage 1: Data extraction …")
    extractor = OSMExtractor(cache_dir=cfg.extraction.osm_cache_dir)
    t0 = time.perf_counter()

    if cfg.preloaded_network_pycgr:
        nodes, edges = extractor.load_network_from_pycgr(cfg.preloaded_network_pycgr)
        logger.info(f"[{label}]   Network: PYCGR ({len(nodes):,} nodes, {len(edges):,} edges)")
    elif cfg.preloaded_network_json:
        nodes, edges = extractor.load_network_from_json(cfg.preloaded_network_json)
        logger.info(f"[{label}]   Network: JSON ({len(nodes):,} nodes, {len(edges):,} edges)")
    else:
        nodes, edges = extractor.extract_road_network(
            region,
            network_type=cfg.extraction.network_type,
            road_types=cfg.extraction.road_types,
            use_cache=cfg.extraction.use_cached_osm,
        )
        logger.info(f"[{label}]   Network: OSM ({len(nodes):,} nodes, {len(edges):,} edges)")

    if cfg.preloaded_poi_csv:
        villages, shelters = extractor.load_pois_from_csv(cfg.preloaded_poi_csv)
        logger.info(f"[{label}]   POIs: CSV ({len(villages)} villages, {len(shelters)} shelters)")
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
        logger.info(f"[{label}]   POIs: GeoJSON/OSM ({len(villages)} villages, {len(shelters)} shelters)")

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
    timings["extraction"] = time.perf_counter() - t0
    logger.info(f"[{label}]   Extraction: {timings['extraction']:.1f}s")

    # ── Stage 2: Risk scoring ─────────────────────────────────────────────────
    logger.info(f"[{label}] Stage 2: Risk scoring …")
    t0 = time.perf_counter()
    if cfg.skip_inarisk:
        logger.warning(f"[{label}] skip_inarisk=true — all risk scores set to 0.0")
        for v in villages:
            v.risk_scores["composite"] = 0.0
        for s in shelters:
            s.risk_scores["composite"] = 0.0
    else:
        inarisk = InaRISKClient(
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
        )
        _hazard_layers_raw = cfg.routing.hazard_layers  # {str: float}
        _hazard_layers = {}
        if _hazard_layers_raw:
            for _name, _weight in _hazard_layers_raw.items():
                try:
                    _hazard_layers[DisasterType(_name)] = float(_weight)
                except ValueError:
                    logger.warning(f"[{label}] Unknown hazard layer '{_name}' — skipping")
        if len(_hazard_layers) > 1:
            logger.info(f"[{label}]   Compound hazard: "
                        f"{', '.join(f'{dt.value}×{w}' for dt, w in _hazard_layers.items())}")
            inarisk.enrich_villages_compound(villages, _hazard_layers, cfg.routing.hazard_aggregation)
            inarisk.enrich_shelters_compound(shelters, _hazard_layers, cfg.routing.hazard_aggregation)
        else:
            inarisk.enrich_villages_with_risk(villages, disaster.disaster_type)
            inarisk.enrich_shelters_with_risk(shelters, disaster.disaster_type)
    timings["risk_scoring"] = time.perf_counter() - t0
    logger.info(f"[{label}]   Risk scoring: {timings['risk_scoring']:.1f}s")

    # ── Stage 3: Graph ────────────────────────────────────────────────────────
    logger.info(f"[{label}] Stage 3: Graph construction …")
    t0 = time.perf_counter()
    builder = EvacuationGraphBuilder()
    G = builder.build(nodes, edges, disaster_type=disaster.disaster_type)
    builder.attach_pois_to_graph(villages, shelters)
    timings["graph_build"] = time.perf_counter() - t0
    logger.info(f"[{label}]   Graph: {G.number_of_nodes():,}N {G.number_of_edges():,}E  "
                f"({timings['graph_build']:.1f}s)")

    # ── Stage 4: Routing ──────────────────────────────────────────────────────
    logger.info(f"[{label}] Stage 4: Route computation …")
    t0 = time.perf_counter()
    optimizer = HeuristicOptimizer(
        weight_distance=cfg.routing.weight_distance,
        weight_risk=cfg.routing.weight_risk,
        weight_road_quality=cfg.routing.weight_road_quality,
        weight_time=cfg.routing.weight_time,
        max_routes_per_village=cfg.routing.max_routes_per_village,
    )
    routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
    routes_by_village = optimizer.rank_routes(routes)
    timings["routing"] = time.perf_counter() - t0
    logger.info(f"[{label}]   Routes: {len(routes)}  ({timings['routing']:.1f}s)")

    # ── Stage 5: Assignment ───────────────────────────────────────────────────
    logger.info(f"[{label}] Stage 5: Population assignment …")
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
    result.runtime_s = sum(timings.values())

    logger.info(
        f"[{label}] Done: {result.total_evacuated:,}/{result.total_population:,} evacuated "
        f"({100*result.evacuation_ratio:.1f}%) in {result.runtime_s:.1f}s"
    )

    # ── Load polygon geometries for visualization ─────────────────────────────
    cache_dir = Path(cfg.extraction.osm_cache_dir)
    village_geoms, shelter_geoms = {}, {}

    try:
        vpath = (Path(cfg.preloaded_villages_geojson)
                 if cfg.preloaded_villages_geojson
                 else _latest_cache(cache_dir, "villages_*.geojson"))
        if vpath and vpath.exists():
            village_geoms = _load_polygon_geoms(vpath, "village_id")
            logger.info(f"[{label}]   Village polygons: {len(village_geoms)} loaded")
    except Exception as e:
        logger.warning(f"[{label}] Village polygon load failed: {e}")

    try:
        spath = (Path(cfg.preloaded_shelters_geojson)
                 if cfg.preloaded_shelters_geojson
                 else _latest_cache(cache_dir, "shelters_*.geojson"))
        if spath and spath.exists():
            shelter_geoms = _load_polygon_geoms(spath, "shelter_id")
            logger.info(f"[{label}]   Shelter polygons: {len(shelter_geoms)} loaded")
    except Exception as e:
        logger.warning(f"[{label}] Shelter polygon load failed: {e}")

    return {
        "cfg":               cfg,
        "disaster":          disaster,
        "villages":          villages,
        "shelters":          shelters,
        "routes_by_village": routes_by_village,
        "result":            result,
        "node_coords":       builder._node_coords,   # {node_id: (lat, lon)}
        "village_geoms":     village_geoms,           # {village_id: shapely_geom}
        "shelter_geoms":     shelter_geoms,           # {shelter_id: shapely_geom}
        "timings":           timings,
        "G":                 G,
    }


# ── Per-mode map ──────────────────────────────────────────────────────────────

def _add_village_layer(m, top_routes, village_geoms, border_color, label_prefix, show=True):
    """Village boundary polygons (or circle fallback), colored by population quintile."""
    import folium
    fg = folium.FeatureGroup(name=f"{label_prefix} – Villages", show=show)
    villages = [v for v, _ in top_routes]
    pop_classes = _quintile_class_list([v.population for v in villages])
    for (village, routes), pc in zip(top_routes, pop_classes):
        risk  = village.risk_scores.get("composite",
                next(iter(village.risk_scores.values()), 0.0))
        color = POP_COLORS[pc - 1]
        tip   = folium.Tooltip(
            f"<b>{village.name}</b><br>"
            f"Population: {village.population:,}<br>"
            f"Risk: {risk:.2f}<br>"
            f"Routes: {len(routes)}"
        )
        geom = village_geoms.get(village.village_id)
        if geom and geom.geom_type in ("Polygon", "MultiPolygon"):
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda _, c=color, bc=border_color: {
                    "fillColor": c, "color": bc, "weight": 2.0, "fillOpacity": 0.6,
                },
                tooltip=tip,
            ).add_to(fg)
        else:
            folium.CircleMarker(
                location=[village.centroid_lat, village.centroid_lon],
                radius=8, color=border_color, fill=True,
                fill_color=color, fill_opacity=0.7, tooltip=tip,
            ).add_to(fg)
    fg.add_to(m)


def _add_shelter_layer(m, shelters, shelter_geoms, label_prefix, disaster_type="volcano", show=True):
    """Shelter boundary polygons (or circle fallback), colored by capacity quintile."""
    import folium
    fg = folium.FeatureGroup(name=f"{label_prefix} – Shelters", show=show)
    cap_classes = _quintile_class_list([s.capacity for s in shelters])
    for s, cc in zip(shelters, cap_classes):
        util  = s.current_occupancy / s.capacity if s.capacity > 0 else 0.0
        risk  = s.risk_scores.get(disaster_type, 0.0)
        color = CAP_COLORS[cc - 1]
        tip   = folium.Tooltip(
            f"<b>{s.name}</b><br>"
            f"Type: {s.shelter_type}<br>"
            f"Capacity: {s.capacity:,}<br>"
            f"Assigned: {s.current_occupancy:,} ({100*util:.0f}%)<br>"
            f"Risk: {risk:.2f}"
        )
        geom = shelter_geoms.get(s.shelter_id)
        if geom and geom.geom_type in ("Polygon", "MultiPolygon"):
            folium.GeoJson(
                geom.__geo_interface__,
                style_function=lambda _, c=color: {
                    "fillColor": c, "color": "#1a5e2a", "weight": 2.0, "fillOpacity": 0.7,
                },
                tooltip=tip,
            ).add_to(fg)
        else:
            folium.CircleMarker(
                location=[s.centroid_lat, s.centroid_lon],
                radius=10, color="#1a5e2a", fill=True,
                fill_color=color, fill_opacity=0.8, tooltip=tip,
            ).add_to(fg)
    fg.add_to(m)


def _add_route_layers(m, top_routes, node_coords, label_prefix, show=True):
    """Route polylines: rank 1 shown, rank 2-3 hidden."""
    import folium
    fg_routes = folium.FeatureGroup(name=f"{label_prefix} – Routes (rank 1)", show=show)
    fg_alt    = folium.FeatureGroup(name=f"{label_prefix} – Alt routes (rank 2-3)", show=False)

    for village, routes in top_routes:
        for route in routes:
            coords = _route_coords(route, node_coords)
            if len(coords) < 2:
                if route.node_path and route.node_path[-1] in node_coords:
                    coords = [(village.centroid_lat, village.centroid_lon),
                              node_coords[route.node_path[-1]]]
            if len(coords) < 2:
                continue

            color   = RANK_COLORS[min(route.rank - 1, len(RANK_COLORS) - 1)]
            weight  = max(2, 5 - route.rank)
            opacity = 0.9 if route.rank == 1 else 0.55
            tip     = folium.Tooltip(
                f"<b>Rank {route.rank}</b>  {village.name} → {route.shelter_id}<br>"
                f"Distance: {route.total_distance_km:.1f} km<br>"
                f"Travel time: {route.total_time_min:.0f} min<br>"
                f"Avg risk: {route.avg_risk_score:.2f}<br>"
                f"Score: {route.composite_score:.3f}"
            )
            folium.PolyLine(
                locations=coords, color=color, weight=weight,
                opacity=opacity, tooltip=tip,
            ).add_to(fg_routes if route.rank == 1 else fg_alt)

    fg_routes.add_to(m)
    fg_alt.add_to(m)


def _disaster_marker(m, cfg):
    import folium
    folium.Marker(
        location=[cfg.disaster.lat, cfg.disaster.lon],
        icon=folium.Icon(color="red", icon="fire", prefix="fa"),
        tooltip=folium.Tooltip(f"<b>{cfg.disaster.name}</b>"),
    ).add_to(m)
    folium.Circle(
        location=[cfg.disaster.lat, cfg.disaster.lon],
        radius=cfg.region.radius_km * 1000,
        color="red", fill=False, opacity=0.25, dash_array="6",
    ).add_to(m)


def _legend(m, label_prefix, border_color):
    import folium
    html = (
        f'<div style="position:fixed;bottom:40px;left:40px;z-index:9999;background:white;'
        f'padding:10px 14px;border-radius:8px;border:1px solid #aaa;font-size:12px;">'
        f'<b>{label_prefix}</b><br><br>'
        "<b>Villages (population)</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:13px;height:13px;'
            f'border:2px solid {border_color};margin-right:5px;"></span>Class {i+1}<br>'
            for i, c in enumerate(POP_COLORS)
        )
        + "<br><b>Shelters (capacity)</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:13px;height:13px;'
            f'border:2px solid #1a5e2a;margin-right:5px;"></span>Class {i+1}<br>'
            for i, c in enumerate(CAP_COLORS)
        )
        + "<br><b>Routes</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:20px;height:3px;'
            f'margin-right:5px;margin-bottom:4px;vertical-align:middle;"></span>Rank {i+1}<br>'
            for i, c in enumerate(RANK_COLORS)
        )
        + "</div>"
    )
    m.get_root().html.add_child(folium.Element(html))


def write_single_map(data: dict, label: str, border_color: str,
                     top_routes, out_path: Path):
    """Write a standalone per-mode route map."""
    import folium
    cfg = data["cfg"]
    disaster_type = cfg.disaster.disaster_type
    m = folium.Map(
        location=[cfg.disaster.lat, cfg.disaster.lon],
        zoom_start=11,
        tiles="CartoDB positron",
    )
    _add_village_layer(m, top_routes, data["village_geoms"], border_color, label, show=True)
    _add_shelter_layer(m, data["shelters"], data["shelter_geoms"], label,
                       disaster_type=disaster_type, show=True)
    _add_route_layers(m, top_routes, data["node_coords"], label, show=True)
    _disaster_marker(m, cfg)
    _legend(m, label, border_color)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_path))
    logger.info(f"  Map saved: {out_path}")


def write_comparison_map(osm: dict, leg: dict,
                          top_osm, top_leg,
                          out_path: Path):
    """Write a single map with both modes as toggleable layer groups."""
    import folium
    cfg = osm["cfg"]
    disaster_type = cfg.disaster.disaster_type
    m = folium.Map(
        location=[cfg.disaster.lat, cfg.disaster.lon],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # OSM layers (shown by default)
    _add_village_layer(m, top_osm, osm["village_geoms"], OSM_COLOR,    "OSM",    show=True)
    _add_route_layers(m, top_osm, osm["node_coords"],                  "OSM",    show=True)

    # Legacy layers (hidden by default)
    _add_village_layer(m, top_leg, leg["village_geoms"], LEGACY_COLOR, "Legacy", show=False)
    _add_route_layers(m, top_leg, leg["node_coords"],                  "Legacy", show=False)

    # Shared shelter layer (same geometry for both modes)
    _add_shelter_layer(m, osm["shelters"], osm["shelter_geoms"], "Shared",
                       disaster_type=disaster_type, show=True)

    _disaster_marker(m, cfg)

    html = (
        '<div style="position:fixed;bottom:40px;left:40px;z-index:9999;background:white;'
        'padding:10px 14px;border-radius:8px;border:1px solid #aaa;font-size:12px;">'
        '<b>Mode comparison</b><br><br>'
        "<b>Villages (population)</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:13px;height:13px;'
            f'border:2px solid {OSM_COLOR};margin-right:5px;"></span>Class {i+1}<br>'
            for i, c in enumerate(POP_COLORS)
        )
        + f'<small style="color:#666">OSM border: ■&nbsp; Legacy border: '
          f'<span style="color:{LEGACY_COLOR}">■</span></small><br><br>'
        + "<b>Shelters (capacity)</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:13px;height:13px;'
            f'border:2px solid #1a5e2a;margin-right:5px;"></span>Class {i+1}<br>'
            for i, c in enumerate(CAP_COLORS)
        )
        + "<br><b>Routes</b><br>"
        + "".join(
            f'<span style="background:{c};display:inline-block;width:20px;height:3px;'
            f'margin-right:5px;margin-bottom:4px;vertical-align:middle;"></span>Rank {i+1}<br>'
            for i, c in enumerate(RANK_COLORS)
        )
        + "<br><i>Toggle layers via control →</i></div>"
    )
    m.get_root().html.add_child(folium.Element(html))
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(str(out_path))
    logger.info(f"  Comparison map saved: {out_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(osm: dict, leg: dict, top_osm, top_leg, out_path: Path):
    """Write JSON comparison of key metrics between both modes."""
    def _route_stats(top_routes):
        r1 = [routes[0] for _, routes in top_routes if routes]
        if not r1:
            return {}
        return {
            "count": len(r1),
            "avg_distance_km": round(sum(r.total_distance_km for r in r1) / len(r1), 2),
            "avg_time_min":    round(sum(r.total_time_min for r in r1) / len(r1), 1),
            "avg_risk":        round(sum(r.avg_risk_score for r in r1) / len(r1), 3),
            "avg_score":       round(sum(r.composite_score for r in r1) / len(r1), 4),
        }

    def _result_summary(data):
        r = data["result"]
        return {
            "total_population":  r.total_population,
            "total_evacuated":   r.total_evacuated,
            "evacuation_ratio":  round(r.evacuation_ratio, 4),
            "avg_route_risk":    round(r.avg_route_risk, 3),
            "avg_distance_km":   round(r.avg_route_distance_km, 2),
            "avg_time_min":      round(r.avg_route_time_min, 1),
            "n_villages":        len(data["villages"]),
            "n_shelters":        len(data["shelters"]),
            "timings_s":         {k: round(v, 2) for k, v in data["timings"].items()},
            "total_runtime_s":   round(r.runtime_s, 2),
            "top_route_stats":   _route_stats(top_osm if data is osm else top_leg),
        }

    summary = {
        "scenario": osm["cfg"].scenario_id,
        "disaster": osm["cfg"].disaster.name,
        "osm":    _result_summary(osm),
        "legacy": _result_summary(leg),
        "delta": {
            "evacuation_ratio":  round(
                leg["result"].evacuation_ratio - osm["result"].evacuation_ratio, 4),
            "avg_distance_km":   round(
                leg["result"].avg_route_distance_km - osm["result"].avg_route_distance_km, 2),
            "avg_risk":          round(
                leg["result"].avg_route_risk - osm["result"].avg_route_risk, 3),
            "runtime_speedup":   round(
                osm["result"].runtime_s / max(leg["result"].runtime_s, 0.001), 2),
        },
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Summary saved: {out_path}")
    return summary


def _print_comparison_table(summary: dict):
    osm = summary["osm"]
    leg = summary["legacy"]
    d   = summary["delta"]

    def fmt_delta(v, unit="", higher_better=True):
        sign = "+" if v > 0 else ""
        arrow = ("▲" if higher_better else "▼") if v > 0 else ("▼" if higher_better else "▲")
        if v == 0:
            return "  ±0"
        return f"  {sign}{v:.4g}{unit} {arrow}"

    rows = [
        ("Villages",         osm["n_villages"],                    leg["n_villages"],                    ""),
        ("Shelters",         osm["n_shelters"],                    leg["n_shelters"],                    ""),
        ("Population",       f"{osm['total_population']:,}",       f"{leg['total_population']:,}",       ""),
        ("Evacuated",        f"{osm['total_evacuated']:,}",        f"{leg['total_evacuated']:,}",        fmt_delta(d["evacuation_ratio"]*100, "%")),
        ("Evacuation ratio", f"{100*osm['evacuation_ratio']:.1f}%", f"{100*leg['evacuation_ratio']:.1f}%", ""),
        ("Avg distance",     f"{osm['avg_distance_km']:.1f} km",  f"{leg['avg_distance_km']:.1f} km",  fmt_delta(-d["avg_distance_km"], "km", False)),
        ("Avg travel time",  f"{osm['avg_time_min']:.0f} min",    f"{leg['avg_time_min']:.0f} min",     ""),
        ("Avg route risk",   f"{osm['avg_route_risk']:.3f}",      f"{leg['avg_route_risk']:.3f}",       fmt_delta(-d["avg_risk"], "", False)),
        ("Runtime",          f"{osm['total_runtime_s']:.1f}s",    f"{leg['total_runtime_s']:.1f}s",     f"  {d['runtime_speedup']:.1f}× faster (legacy)"),
    ]

    w = 24
    logger.info("")
    logger.info("=" * 72)
    logger.info(f"{'METRIC':<{w}}  {'OSM':>16}  {'LEGACY':>16}  DELTA")
    logger.info("-" * 72)
    for label, oval, lval, delta in rows:
        logger.info(f"{label:<{w}}  {str(oval):>16}  {str(lval):>16}  {delta}")
    logger.info("=" * 72)
    logger.info("")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare OSM vs legacy input: run both pipelines, visualize top routes."
    )
    parser.add_argument("--osm-config",
                        default="configs/demak_flood_2024.yaml",
                        help="Config for OSM mode")
    parser.add_argument("--legacy-config",
                        default="output/<scenario_id>/legacy_input/scenario_preloaded.yaml",
                        help="Config for legacy mode (preloaded files)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top villages to show routes for (default: 10)")
    parser.add_argument("--output-dir",
                        help="Override output directory")
    args = parser.parse_args()

    osm_path    = Path(args.osm_config)
    legacy_path = Path(args.legacy_config)

    for p in [osm_path, legacy_path]:
        if not p.exists():
            logger.error(f"Config not found: {p}")
            raise SystemExit(1)

    # ── Run both pipelines ────────────────────────────────────────────────────
    logger.info("Running OSM pipeline …")
    osm = run_pipeline(osm_path, "OSM")

    logger.info("Running Legacy pipeline …")
    leg = run_pipeline(legacy_path, "Legacy")

    # ── Select top routes ─────────────────────────────────────────────────────
    top_n    = args.top_n
    top_osm  = _select_top_routes(osm["routes_by_village"], osm["villages"], top_n)
    top_leg  = _select_top_routes(leg["routes_by_village"], leg["villages"], top_n)
    logger.info(f"Selected top {len(top_osm)} OSM villages, {len(top_leg)} legacy villages")

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(osm["cfg"].output_dir) / "route_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing outputs to {out_dir} …")

    # ── Maps ──────────────────────────────────────────────────────────────────
    write_single_map(osm, "OSM",    OSM_COLOR,    top_osm, out_dir / "routes_osm.html")
    write_single_map(leg, "Legacy", LEGACY_COLOR, top_leg, out_dir / "routes_legacy.html")
    write_comparison_map(osm, leg, top_osm, top_leg,        out_dir / "routes_comparison.html")

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = write_summary(osm, leg, top_osm, top_leg, out_dir / "comparison_summary.json")
    _print_comparison_table(summary)

    logger.info("Done.")
    logger.info(f"  {out_dir}/routes_osm.html        — OSM mode top routes")
    logger.info(f"  {out_dir}/routes_legacy.html     — Legacy mode top routes")
    logger.info(f"  {out_dir}/routes_comparison.html — Both modes, toggle layers")
    logger.info(f"  {out_dir}/comparison_summary.json — Numeric comparison")


if __name__ == "__main__":
    main()
