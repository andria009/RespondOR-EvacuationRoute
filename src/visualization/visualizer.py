"""
Visualization module.
Generates:
  - Interactive Folium maps (HTML)
  - Static matplotlib maps (PNG)
  - Route overlays, hazard heatmaps, shelter utilization charts
  - Benchmark comparison charts
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Risk color gradient: green (0) → yellow → red (1)
def _risk_color(risk: float) -> str:
    r = int(min(255, 2 * risk * 255))
    g = int(min(255, 2 * (1 - risk) * 255))
    return f"#{r:02x}{g:02x}00"

# Route color by rank (rank 1=green, 2=orange, 3=red)
ROUTE_COLORS = ["#1a9641", "#fdae61", "#d7191c", "#2c7bb6", "#756bb1"]

# Village population: blue quintiles (light → dark)
POP_COLORS = ["#c6dbef", "#6baed6", "#2171b5", "#084594", "#08306b"]
# Shelter capacity: green quintiles (light → dark)
CAP_COLORS = ["#c7e9c0", "#74c476", "#238b45", "#005a32", "#00441b"]

# Hazard color ramps (same as preview_hazard.py — 5-step low→high risk)
HAZARD_COLORS = {
    "volcano":      ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
    "landslide":    ["#ffffcc", "#c7e9b4", "#7fcdbb", "#2c7fb8", "#253494"],
    "flood":        ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"],
    "tsunami":      ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "liquefaction": ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "flash_flood":  ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
}
HAZARD_LABELS = {
    "volcano": "Volcano Hazard", "landslide": "Landslide Hazard",
    "flood": "Flood Hazard", "tsunami": "Tsunami Hazard",
    "liquefaction": "Liquefaction Hazard", "flash_flood": "Flash Flood Hazard",
}

def _hazard_score_color(score: float, colors: list) -> str:
    idx = min(int(score * len(colors)), len(colors) - 1)
    return colors[idx]


def _quintile_class_list(values: list) -> list:
    """Assign 1–5 quintile class to each value in the list."""
    if not values:
        return []
    nonzero = sorted(v for v in values if v > 0)
    if not nonzero:
        return [1] * len(values)
    m = len(nonzero)
    q = [nonzero[max(0, int(p / 100 * m) - 1)] for p in [20, 40, 60, 80]]
    result = []
    for v in values:
        if v <= 0:       result.append(1)
        elif v <= q[0]:  result.append(1)
        elif v <= q[1]:  result.append(2)
        elif v <= q[2]:  result.append(3)
        elif v <= q[3]:  result.append(4)
        else:            result.append(5)
    return result


class EvacuationVisualizer:
    """
    Generates interactive and static visualizations for the evacuation system.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Interactive map (Folium)
    # ------------------------------------------------------------------ #

    def create_interactive_map(
        self,
        villages,
        shelters,
        routes_by_village,
        disaster_location: Optional[Tuple[float, float]] = None,
        map_filename: str = "evacuation_map.html",
        disaster_type: str = "earthquake",
        village_geoms: Optional[Dict] = None,
        shelter_geoms: Optional[Dict] = None,
        node_coords: Optional[Dict] = None,
        disaster_name: Optional[str] = None,
        region_radius_km: Optional[float] = None,
        village_admin_ctx: Optional[Dict] = None,
        shelter_admin_ctx: Optional[Dict] = None,
        hazard_scores: Optional[Dict] = None,
        hazard_step_km: float = 1.0,
    ):
        """
        Create interactive Folium map with:
        - Village polygons (blue quintile; C_/S_ names, L9/L8 admin breadcrumb)
        - Shelter polygons (green quintile; L9/L8, sqm, capacity, risk)
        - Hazard grid rectangles (same style as preview_hazard.html)
        - Evacuation routes per shelter (road-following polylines, colored by rank)
        - Shelter filter panel (bottom-right) — per-shelter route toggles
        - Risk heatmap layer (toggleable, off by default)
        - Disaster location marker + region circle

        village_admin_ctx: {village_id: {"display_name", "l9_name", "l8_name"}}
        shelter_admin_ctx: {shelter_id: {"l9_name", "l8_name"}}
        hazard_scores:     {hazard_type: {"lat,lon": score}} from InaRISK grid cache
        """
        try:
            import folium
            from folium.plugins import HeatMap
        except ImportError:
            logger.warning("folium not installed; skipping interactive map")
            return

        center = (list(disaster_location) if disaster_location
                  else ([sum(v.centroid_lat for v in villages) / len(villages),
                         sum(v.centroid_lon for v in villages) / len(villages)]
                        if villages else [-7.0, 110.0]))

        m = folium.Map(location=center, zoom_start=11, tiles="CartoDB positron")
        map_var = m.get_name()   # JS variable name, e.g. "map_abc123"

        vg  = village_geoms or {}
        sg  = shelter_geoms or {}
        nc  = node_coords or {}
        vac = village_admin_ctx or {}   # {village_id: {"display_name","l9_name","l8_name"}}
        sac = shelter_admin_ctx or {}   # {shelter_id: {"l9_name","l8_name"}}
        v_map = {v.village_id: v for v in villages}
        s_map = {s.shelter_id: s for s in shelters}

        # ── Villages – boundary polygons, blue quintile by population ─────────
        real_cls  = [v for v in villages if not v.village_id.startswith("artificial_l9_")]
        synth_cls = [v for v in villages if v.village_id.startswith("artificial_l9_")]

        fg_villages = folium.FeatureGroup(name="Building Clusters", show=True)
        fg_synthetic = folium.FeatureGroup(name="Synthetic Clusters", show=bool(synth_cls))
        pop_classes = _quintile_class_list([v.population for v in villages])
        pop_class_map = {v.village_id: pc for v, pc in zip(villages, pop_classes)}

        for v in villages:
            is_synth = v.village_id.startswith("artificial_l9_")
            pc    = pop_class_map[v.village_id]
            color = POP_COLORS[pc - 1]
            risk  = v.risk_scores.get(disaster_type, v.risk_scores.get("composite", 0.0))
            vctx  = vac.get(v.village_id, {})
            dname = vctx.get("display_name") or v.name
            l9nm  = vctx.get("l9_name", "")
            l8nm  = vctx.get("l8_name", "")
            kind  = "Synthetic (no OSM buildings)" if is_synth else "Building Cluster"
            area_sqm = int(v.area_m2) if hasattr(v, "area_m2") and v.area_m2 else 0
            admin_line = ""
            if l9nm or l8nm:
                admin_line = f"<span style='color:#555;font-size:10px'>{l9nm}{' · ' + l8nm if l8nm else ''}</span><br>"
            tip   = folium.Tooltip(
                f"<b>{dname}</b><br>"
                f"<i style='color:#666'>{kind}</i><br>"
                f"{admin_line}"
                f"Population: <b>{v.population:,}</b><br>"
                f"Area: {area_sqm:,} m²<br>"
                f"Risk ({disaster_type}): {risk:.3f}"
            )
            geom  = vg.get(v.village_id)
            target_fg = fg_synthetic if is_synth else fg_villages
            if geom and geom.geom_type in ("Polygon", "MultiPolygon"):
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda _, c=color: {
                        "fillColor": c, "color": "#1a4e8a",
                        "weight": 1.0, "fillOpacity": 0.55,
                    },
                    tooltip=tip,
                ).add_to(target_fg)
            else:
                max_pop = max((v2.population for v2 in villages), default=1)
                radius = max(4, min(14, 4 + 10 * (v.population / max_pop) ** 0.5))
                folium.CircleMarker(
                    location=[v.centroid_lat, v.centroid_lon],
                    radius=radius, color="#1a4e8a", fill=True,
                    fill_color=color, fill_opacity=0.6, tooltip=tip,
                ).add_to(target_fg)
        fg_villages.add_to(m)
        fg_synthetic.add_to(m)

        # ── Shelters – boundary polygons, green quintile by capacity ──────────
        # Also track JS var per shelter for filter panel show/hide via setStyle.
        fg_shelters = folium.FeatureGroup(name="Shelters", show=True)
        cap_classes = _quintile_class_list([s.capacity for s in shelters])
        # shelter_geom_info[sid] = {"var": js_var, "fill_opacity": float}
        shelter_geom_info: Dict[str, dict] = {}

        for s, cc in zip(shelters, cap_classes):
            risk  = s.risk_scores.get(disaster_type, s.risk_scores.get("composite", 0.0))
            color = CAP_COLORS[cc - 1]
            sctx  = sac.get(s.shelter_id, {})
            l9nm  = sctx.get("l9_name", "")
            l8nm  = sctx.get("l8_name", "")
            s_area_sqm = int(s.area_m2) if hasattr(s, "area_m2") and s.area_m2 else 0
            admin_line = ""
            if l9nm or l8nm:
                admin_line = (f"<span style='color:#555;font-size:10px'>"
                              f"{l9nm}{' · ' + l8nm if l8nm else ''}</span><br>")
            tip = folium.Tooltip(
                f"<b>{s.name or s.shelter_id}</b><br>"
                f"Type: <i>{getattr(s, 'shelter_type', '')}</i><br>"
                f"{admin_line}"
                f"Area: {s_area_sqm:,} m²<br>"
                f"Capacity: <b>{s.capacity:,}</b> persons<br>"
                f"Risk: {risk:.3f}"
            )
            geom = sg.get(s.shelter_id)
            if geom and geom.geom_type in ("Polygon", "MultiPolygon"):
                lyr = folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda _, c=color: {
                        "fillColor": c, "color": "#1a5e2a",
                        "weight": 2.0, "fillOpacity": 0.75,
                    },
                    tooltip=tip,
                )
                shelter_geom_info[s.shelter_id] = {"var": lyr.get_name(), "fill_opacity": 0.75}
                lyr.add_to(fg_shelters)
            else:
                lyr = folium.CircleMarker(
                    location=[s.centroid_lat, s.centroid_lon],
                    radius=10, color="#1a5e2a", fill=True,
                    fill_color=color, fill_opacity=0.8, tooltip=tip,
                )
                shelter_geom_info[s.shelter_id] = {"var": lyr.get_name(), "fill_opacity": 0.8}
                lyr.add_to(fg_shelters)
        fg_shelters.add_to(m)

        # ── Routes — two FeatureGroups (Primary + Alternative) in LayerControl ─
        # Shelter filter panel hides/shows individual polylines via setStyle({opacity:0}).
        # Layers stay in their FeatureGroup so LayerControl toggling still works.
        fg_primary = folium.FeatureGroup(name="Primary Routes", show=True)
        fg_alt     = folium.FeatureGroup(name="Alternative Routes", show=True)

        # shelter_poly_data[sid] = [{"var", "opacity", "weight"}, ...]
        shelter_poly_data: Dict[str, list]  = {s.shelter_id: [] for s in shelters}
        shelter_route_counts: Dict[str, int] = {s.shelter_id: 0 for s in shelters}

        for vid, routes in routes_by_village.items():
            if not routes:
                continue
            v_obj = v_map.get(vid)
            vname = (vac.get(vid, {}).get("display_name") or
                     (v_obj.name if v_obj else vid))
            for rank, route in enumerate(routes[:3]):
                sid   = route.shelter_id
                s_obj = s_map.get(sid)
                if s_obj is None:
                    continue

                color   = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
                weight  = max(1, 4 - rank)
                opacity = round(0.80 - rank * 0.18, 2)

                coords = []
                if nc and route.node_path:
                    coords = [nc[n] for n in route.node_path if n in nc]
                if len(coords) < 2:
                    if v_obj and s_obj:
                        coords = [(v_obj.centroid_lat, v_obj.centroid_lon),
                                  (s_obj.centroid_lat, s_obj.centroid_lon)]
                if len(coords) < 2:
                    continue

                sname      = s_obj.name if s_obj else sid
                rank_label = ["Primary", "Alternative 1", "Alternative 2"][min(rank, 2)]
                tip = folium.Tooltip(
                    f"<b>{rank_label}</b><br>"
                    f"<b>{vname}</b> → <b>{sname}</b><br>"
                    f"Distance: {route.total_distance_km:.1f} km · "
                    f"Time: {route.total_time_min:.0f} min<br>"
                    f"Avg risk: {route.avg_risk_score:.3f} · "
                    f"Score: {route.composite_score:.4f}"
                )
                poly = folium.PolyLine(
                    locations=coords, color=color, weight=weight,
                    opacity=opacity, tooltip=tip,
                )
                poly_var = poly.get_name()
                if rank == 0:
                    poly.add_to(fg_primary)
                else:
                    poly.add_to(fg_alt)

                shelter_poly_data[sid].append(
                    {"var": poly_var, "opacity": opacity, "weight": weight}
                )
                shelter_route_counts[sid] += 1

        fg_primary.add_to(m)
        fg_alt.add_to(m)

        # ── Risk heatmap (off by default) ─────────────────────────────────────
        heat_data = [
            [v.centroid_lat, v.centroid_lon,
             v.risk_scores.get(disaster_type, v.risk_scores.get("composite", 0.0))]
            for v in villages
            if v.risk_scores.get(disaster_type, v.risk_scores.get("composite", 0.0)) > 0
        ]
        if heat_data:
            fg_heat = folium.FeatureGroup(name="Risk Heatmap", show=False)
            HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(fg_heat)
            fg_heat.add_to(m)

        # ── Hazard grid (InaRISK rectangles, same style as preview_hazard) ────
        if hazard_scores:
            for htype, scores in hazard_scores.items():
                if not scores:
                    continue
                colors = HAZARD_COLORS.get(htype, HAZARD_COLORS["landslide"])
                label  = HAZARD_LABELS.get(htype, htype.replace("_", " ").title())
                nonzero = [v for v in scores.values() if v > 0]
                max_score = max(nonzero) if nonzero else 1.0
                fg_haz = folium.FeatureGroup(name=f"Hazard: {label}", show=False)
                half = hazard_step_km / 111.0 / 2
                for key, score in scores.items():
                    if score <= 0.0:
                        continue
                    lat_s, lon_s = key.split(",")
                    lat, lon = float(lat_s), float(lon_s)
                    color = _hazard_score_color(score / max_score, colors)
                    folium.Rectangle(
                        bounds=[[lat - half, lon - half], [lat + half, lon + half]],
                        color=None, weight=0,
                        fill=True, fill_color=color, fill_opacity=0.65,
                        tooltip=f"{label}: {score:.2f}",
                    ).add_to(fg_haz)
                fg_haz.add_to(m)

        # ── Disaster location ─────────────────────────────────────────────────
        if disaster_location:
            folium.Marker(
                location=list(disaster_location),
                icon=folium.Icon(color="red", icon="fire", prefix="fa"),
                tooltip=folium.Tooltip(f"<b>{disaster_name or 'Disaster'}</b>"),
            ).add_to(m)
            if region_radius_km:
                folium.Circle(
                    location=list(disaster_location),
                    radius=region_radius_km * 1000,
                    color="red", fill=False, opacity=0.25, dash_array="6",
                ).add_to(m)

        # ── Shelter filter panel (bottom-right) ───────────────────────────────
        # Uses setStyle({opacity:0}) — layers remain in their FeatureGroups so
        # the LayerControl Primary/Alternative toggles still work independently.
        import json as _json

        shelter_poly_js = _json.dumps(shelter_poly_data)
        shelter_geom_js = _json.dumps(shelter_geom_info)

        total_routes = sum(shelter_route_counts.values())
        shelter_rows_html = ""
        for s in shelters:
            sid    = s.shelter_id
            sname  = s.name or sid
            count  = shelter_route_counts.get(sid, 0)
            util   = s.current_occupancy / s.capacity if s.capacity > 0 else 0.0
            color  = CAP_COLORS[min(int(util * 5), 4)]
            shelter_rows_html += (
                f'<div class="shel-row">'
                f'<label style="display:flex;align-items:center;gap:5px;cursor:pointer;">'
                f'<input type="checkbox" class="shel-cb" data-sid="{sid}" checked '
                f'{"disabled" if count == 0 else ""}>'
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'background:{color};border:1px solid #555;flex-shrink:0"></span>'
                f'<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"'
                f' title="{sname}">{sname}</span>'
                f'<span style="color:#888;font-size:10px;white-space:nowrap">'
                f'{count}R · {s.capacity:,}</span>'
                f'</label></div>\n'
            )

        filter_panel_html = f"""
<div id="sfp" style="position:fixed;bottom:30px;right:10px;z-index:9999;
     background:white;padding:8px 12px;border-radius:8px;
     box-shadow:0 2px 10px rgba(0,0,0,0.25);font-size:11px;
     font-family:sans-serif;width:255px;max-height:55vh;display:flex;flex-direction:column;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;">
    <b style="font-size:12px">Shelters &amp; Routes ({total_routes})</b>
    <div style="display:flex;gap:3px;align-items:center;">
      <button onclick="sfpSetAll(true)"
              style="font-size:10px;padding:1px 5px;cursor:pointer">All</button>
      <button onclick="sfpSetAll(false)"
              style="font-size:10px;padding:1px 5px;cursor:pointer">None</button>
      <button id="sfp-min-btn" onclick="sfpToggleMinimize()"
              style="font-size:13px;font-weight:bold;padding:0 5px;cursor:pointer;
                     border:1px solid #ccc;border-radius:3px;line-height:1.3">−</button>
    </div>
  </div>
  <div id="sfp-body" style="overflow-y:auto;flex:1;">{shelter_rows_html}</div>
  <div id="sfp-foot" style="margin-top:4px;padding-top:4px;border-top:1px solid #ddd;
       font-size:10px;color:#888;">R = routes · cap = capacity</div>
</div>

<script>
(function() {{
  // Data injected from Python
  var polyData = {shelter_poly_js};   // sid → [{{var, opacity, weight}}, ...]
  var geomData = {shelter_geom_js};   // sid → {{var, fill_opacity}}

  function getLayer(varName) {{
    return window[varName];
  }}

  function setShelterVisible(sid, show) {{
    // Shelter geometry
    var gi = geomData[sid];
    if (gi) {{
      var gl = getLayer(gi["var"]);
      if (gl && gl.setStyle) {{
        gl.setStyle(show
          ? {{opacity: 1, fillOpacity: gi["fill_opacity"]}}
          : {{opacity: 0, fillOpacity: 0}});
      }}
    }}
    // Route polylines
    var routes = polyData[sid] || [];
    routes.forEach(function(p) {{
      var pl = getLayer(p["var"]);
      if (pl && pl.setStyle) {{
        pl.setStyle(show
          ? {{opacity: p["opacity"], weight: p["weight"]}}
          : {{opacity: 0, weight: 0}});
      }}
    }});
  }}

  // Wire checkboxes
  document.querySelectorAll(".shel-cb").forEach(function(cb) {{
    cb.addEventListener("change", function() {{
      setShelterVisible(this.getAttribute("data-sid"), this.checked);
    }});
  }});

  // All / None buttons
  window.sfpSetAll = function(state) {{
    document.querySelectorAll(".shel-cb").forEach(function(cb) {{
      if (cb.disabled) return;
      cb.checked = state;
      setShelterVisible(cb.getAttribute("data-sid"), state);
    }});
  }};

  // Minimize
  var minimized = false;
  window.sfpToggleMinimize = function() {{
    minimized = !minimized;
    document.getElementById("sfp-body").style.display = minimized ? "none" : "";
    document.getElementById("sfp-foot").style.display = minimized ? "none" : "";
    document.getElementById("sfp-min-btn").textContent = minimized ? "+" : "−";
  }};
}})();
</script>
"""
        m.get_root().html.add_child(folium.Element(filter_panel_html))

        # ── Legend ────────────────────────────────────────────────────────────
        legend_html = (
            '<div style="position:fixed;bottom:30px;left:30px;z-index:9999;background:white;'
            'padding:12px 16px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.2);'
            'font-size:11px;font-family:sans-serif;">'
            "<b>Villages (population)</b><br>"
            + "".join(
                f'<span style="background:{c};display:inline-block;width:12px;height:12px;'
                f'margin-right:5px;border:1px solid #555;"></span>Class {i+1}<br>'
                for i, c in enumerate(POP_COLORS)
            )
            + "<br><b>Shelters (capacity)</b><br>"
            + "".join(
                f'<span style="background:{c};display:inline-block;width:12px;height:12px;'
                f'margin-right:5px;border:1px solid #555;"></span>Class {i+1}<br>'
                for i, c in enumerate(CAP_COLORS)
            )
            + "<br><b>Routes by rank</b><br>"
            + "".join(
                f'<span style="background:{c};display:inline-block;width:20px;height:3px;'
                f'margin-right:5px;margin-bottom:4px;vertical-align:middle;"></span>'
                f'{"Primary" if i==0 else f"Alternative {i}"}<br>'
                for i, c in enumerate(ROUTE_COLORS[:3])
            )
            + "</div>"
        )
        m.get_root().html.add_child(folium.Element(legend_html))

        folium.LayerControl(collapsed=False).add_to(m)

        out_path = self.output_dir / map_filename
        m.save(str(out_path))
        logger.info(f"Interactive map saved to {out_path}")
        return str(out_path)

    # ------------------------------------------------------------------ #
    # Static matplotlib charts
    # ------------------------------------------------------------------ #

    def create_evacuation_summary_chart(
        self,
        result,
        filename: str = "evacuation_summary.png",
    ):
        """Summary chart: evacuated vs unmet, shelter utilization."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping summary chart")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Pie: evacuated vs unmet
        evacuated = result.total_evacuated
        unmet = result.total_unmet
        if evacuated + unmet > 0:
            axes[0].pie(
                [evacuated, unmet],
                labels=[f"Evacuated\n{evacuated:,}", f"Unmet\n{unmet:,}"],
                colors=["#2ecc71", "#e74c3c"],
                autopct="%1.1f%%",
                startangle=90,
            )
        axes[0].set_title("Evacuation Coverage")

        # Bar: shelter utilization
        shelter_ids = list(result.shelter_utilization.keys())[:10]
        utils = [result.shelter_utilization[sid] for sid in shelter_ids]
        colors = [_risk_color(u) for u in utils]
        axes[1].barh(range(len(shelter_ids)), utils, color=colors)
        axes[1].set_yticks(range(len(shelter_ids)))
        axes[1].set_yticklabels([sid[:15] for sid in shelter_ids])
        axes[1].set_xlabel("Utilization Rate")
        axes[1].set_title("Shelter Utilization (top 10)")
        axes[1].axvline(1.0, color="red", linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Summary chart saved to {out_path}")

    # ------------------------------------------------------------------ #
    # CSV/JSON summary export
    # ------------------------------------------------------------------ #

    def export_result_csv(self, result, villages, shelters, filename: str = "evacuation_results.csv"):
        """Export assignment results as CSV."""
        import csv
        village_map = {v.village_id: v for v in villages}
        shelter_map = {s.shelter_id: s for s in shelters}

        out_path = self.output_dir / filename
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "village_id", "village_name", "population",
                "shelter_id", "shelter_name", "shelter_capacity",
                "assigned_population", "fraction",
                "distance_km", "travel_time_min",
                "avg_risk", "composite_score"
            ])
            writer.writeheader()
            for a in result.assignments:
                v = village_map.get(a.village_id)
                s = shelter_map.get(a.shelter_id)
                r = a.route
                writer.writerow({
                    "village_id": a.village_id,
                    "village_name": v.name if v else "",
                    "population": v.population if v else 0,
                    "shelter_id": a.shelter_id,
                    "shelter_name": s.name if s else "",
                    "shelter_capacity": s.capacity if s else 0,
                    "assigned_population": a.assigned_population,
                    "fraction": round(a.fraction, 3),
                    "distance_km": round(r.total_distance_km, 2) if r else 0,
                    "travel_time_min": round(r.total_time_min, 1) if r else 0,
                    "avg_risk": round(r.avg_risk_score, 3) if r else 0,
                    "composite_score": round(r.composite_score, 4) if r else 0,
                })
        logger.info(f"Results CSV saved to {out_path}")
        return str(out_path)
