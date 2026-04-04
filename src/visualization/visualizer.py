"""
Visualization module.
Generates:
  - Interactive Folium maps (HTML)
  - Static matplotlib maps (PNG)
  - Route overlays, hazard heatmaps, shelter utilization charts
  - Benchmark comparison charts
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
    ):
        """
        Create interactive Folium map with:
        - Village boundary polygons (blue quintile by population)
        - Shelter boundary polygons (green quintile by capacity, utilization in tooltip)
        - Evacuation route polylines (colored by rank, actual road paths when node_coords provided)
        - Risk heatmap layer (toggleable, off by default)
        - Disaster location marker + region circle
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

        vg = village_geoms or {}
        sg = shelter_geoms or {}
        nc = node_coords or {}
        v_map = {v.village_id: v for v in villages}
        s_map = {s.shelter_id: s for s in shelters}

        # ── Villages – boundary polygons, blue quintile by population ─────────
        fg_villages = folium.FeatureGroup(name="Villages – Population", show=True)
        pop_classes = _quintile_class_list([v.population for v in villages])
        for v, pc in zip(villages, pop_classes):
            risk  = v.risk_scores.get(disaster_type, 0.0)
            color = POP_COLORS[pc - 1]
            tip   = folium.Tooltip(
                f"<b>{v.name}</b><br>"
                f"Population: {v.population:,}<br>"
                f"Risk ({disaster_type}): {risk:.2f}"
            )
            geom = vg.get(v.village_id)
            if geom and geom.geom_type in ("Polygon", "MultiPolygon"):
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda _, c=color: {
                        "fillColor": c, "color": "#1a4e8a",
                        "weight": 1.5, "fillOpacity": 0.6,
                    },
                    tooltip=tip,
                ).add_to(fg_villages)
            else:
                max_pop = max((v2.population for v2 in villages), default=1)
                radius = 5 + 12 * (v.population / max_pop)
                folium.CircleMarker(
                    location=[v.centroid_lat, v.centroid_lon],
                    radius=radius, color="#1a4e8a", fill=True,
                    fill_color=color, fill_opacity=0.6, tooltip=tip,
                ).add_to(fg_villages)
        fg_villages.add_to(m)

        # ── Shelters – boundary polygons, green quintile by capacity ──────────
        fg_shelters = folium.FeatureGroup(name="Shelters – Capacity", show=True)
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
            geom = sg.get(s.shelter_id)
            if geom and geom.geom_type in ("Polygon", "MultiPolygon"):
                folium.GeoJson(
                    geom.__geo_interface__,
                    style_function=lambda _, c=color: {
                        "fillColor": c, "color": "#1a5e2a",
                        "weight": 2.0, "fillOpacity": 0.7,
                    },
                    tooltip=tip,
                ).add_to(fg_shelters)
            else:
                folium.CircleMarker(
                    location=[s.centroid_lat, s.centroid_lon],
                    radius=10, color="#1a5e2a", fill=True,
                    fill_color=color, fill_opacity=0.8, tooltip=tip,
                ).add_to(fg_shelters)
        fg_shelters.add_to(m)

        # ── Routes – rank 1 (shown) and rank 2-3 (hidden) ────────────────────
        fg_routes = folium.FeatureGroup(name="Routes – Rank 1 (primary)", show=True)
        fg_alt    = folium.FeatureGroup(name="Routes – Rank 2-3 (alt)", show=False)
        for vid, routes in routes_by_village.items():
            if not routes:
                continue
            for rank, route in enumerate(routes[:3]):
                color   = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
                weight  = max(2, 5 - rank)
                opacity = 0.85 - rank * 0.2

                coords = []
                if nc and route.node_path:
                    coords = [nc[n] for n in route.node_path if n in nc]
                if len(coords) < 2:
                    v_obj = v_map.get(vid)
                    s_obj = s_map.get(route.shelter_id)
                    if v_obj and s_obj:
                        coords = [(v_obj.centroid_lat, v_obj.centroid_lon),
                                  (s_obj.centroid_lat, s_obj.centroid_lon)]
                if len(coords) < 2:
                    continue

                tip = (f"<b>Rank {rank+1}</b>  {v_map.get(vid, type('', (), {'name': vid})()).name}"
                       f" → {route.shelter_id}<br>"
                       f"Distance: {route.total_distance_km:.1f} km<br>"
                       f"Travel time: {route.total_time_min:.0f} min<br>"
                       f"Avg risk: {route.avg_risk_score:.2f}")
                line = folium.PolyLine(
                    locations=coords, color=color, weight=weight,
                    opacity=opacity, tooltip=folium.Tooltip(tip),
                )
                (fg_routes if rank == 0 else fg_alt).add_to(m) if False else None
                line.add_to(fg_routes if rank == 0 else fg_alt)
        fg_routes.add_to(m)
        fg_alt.add_to(m)

        # ── Risk heatmap (off by default) ─────────────────────────────────────
        heat_data = [
            [v.centroid_lat, v.centroid_lon, v.risk_scores.get(disaster_type, 0.0)]
            for v in villages if v.risk_scores.get(disaster_type, 0.0) > 0
        ]
        if heat_data:
            fg_heat = folium.FeatureGroup(name="Risk heatmap", show=False)
            HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(fg_heat)
            fg_heat.add_to(m)

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

        # ── Legend ────────────────────────────────────────────────────────────
        legend_html = (
            '<div style="position:fixed;bottom:40px;left:40px;z-index:9999;background:white;'
            'padding:12px 16px;border-radius:8px;border:1px solid #aaa;font-size:12px;">'
            "<b>Villages (population)</b><br>"
            + "".join(
                f'<span style="background:{c};display:inline-block;width:13px;height:13px;'
                f'margin-right:5px;border:1px solid #555;"></span>Class {i+1}<br>'
                for i, c in enumerate(POP_COLORS)
            )
            + "<br><b>Shelters (capacity)</b><br>"
            + "".join(
                f'<span style="background:{c};display:inline-block;width:13px;height:13px;'
                f'margin-right:5px;border:1px solid #555;"></span>Class {i+1}<br>'
                for i, c in enumerate(CAP_COLORS)
            )
            + "<br><b>Routes</b><br>"
            + "".join(
                f'<span style="background:{c};display:inline-block;width:20px;height:3px;'
                f'margin-right:5px;margin-bottom:4px;vertical-align:middle;"></span>Rank {i+1}<br>'
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

    def create_benchmark_chart(
        self,
        benchmark_results,
        filename: str = "benchmark_chart.png",
    ):
        """Bar chart comparing wall time, speedup, efficiency across modes."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib not installed; skipping benchmark chart")
            return

        modes = [r.mode.value for r in benchmark_results]
        times = [r.wall_time_s for r in benchmark_results]
        speedups = [r.speedup for r in benchmark_results]
        efficiencies = [r.efficiency for r in benchmark_results]

        x = np.arange(len(modes))
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(x, times, color=["#3498db", "#2ecc71", "#e74c3c"][:len(modes)])
        axes[0].set_title("Wall Time (s)")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(modes)
        axes[0].set_ylabel("Seconds")
        for i, t in enumerate(times):
            axes[0].text(i, t * 1.01, f"{t:.1f}s", ha="center", fontsize=9)

        axes[1].bar(x, speedups, color=["#3498db", "#2ecc71", "#e74c3c"][:len(modes)])
        axes[1].set_title("Speedup vs Naive")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(modes)
        axes[1].set_ylabel("Speedup")
        axes[1].axhline(1.0, color="gray", linestyle="--")

        axes[2].bar(x, efficiencies, color=["#3498db", "#2ecc71", "#e74c3c"][:len(modes)])
        axes[2].set_title("Parallel Efficiency")
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(modes)
        axes[2].set_ylabel("Efficiency")
        axes[2].axhline(1.0, color="gray", linestyle="--")

        plt.tight_layout()
        out_path = self.output_dir / filename
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Benchmark chart saved to {out_path}")

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
