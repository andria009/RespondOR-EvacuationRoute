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

# Route color by rank
ROUTE_COLORS = ["#1a9641", "#fdae61", "#d7191c", "#2c7bb6", "#756bb1"]


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
    ):
        """
        Create interactive Folium map with:
        - Villages (blue circles, sized by population)
        - Shelters (green markers, colored by utilization)
        - Route lines (colored by rank)
        - Disaster location marker
        - Risk heatmap layer
        """
        try:
            import folium
            from folium.plugins import HeatMap, MarkerCluster
        except ImportError:
            logger.warning("folium not installed; skipping interactive map")
            return

        # Map center
        if disaster_location:
            center = list(disaster_location)
        elif villages:
            center = [
                sum(v.centroid_lat for v in villages) / len(villages),
                sum(v.centroid_lon for v in villages) / len(villages),
            ]
        else:
            center = [-7.0, 110.0]  # Default: Java, Indonesia

        m = folium.Map(location=center, zoom_start=12, tiles="CartoDB positron")

        # Feature groups for layer control
        fg_villages = folium.FeatureGroup(name="Villages (population areas)", show=True)
        fg_shelters = folium.FeatureGroup(name="Evacuation shelters", show=True)
        fg_routes = folium.FeatureGroup(name="Evacuation routes", show=True)
        fg_disaster = folium.FeatureGroup(name="Disaster location", show=True)

        # --- Villages ---
        max_pop = max((v.population for v in villages), default=1)
        for v in villages:
            radius = 5 + 15 * (v.population / max_pop) if max_pop > 0 else 8
            risk = v.risk_scores.get(disaster_type, 0.0)
            color = _risk_color(risk)
            folium.CircleMarker(
                location=[v.centroid_lat, v.centroid_lon],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.6,
                tooltip=folium.Tooltip(
                    f"<b>{v.name}</b><br>"
                    f"Population: {v.population:,}<br>"
                    f"Risk ({disaster_type}): {risk:.2f}"
                ),
            ).add_to(fg_villages)

        # --- Shelters ---
        total_cap = sum(s.capacity for s in shelters)
        for s in shelters:
            assigned = s.current_occupancy
            util = assigned / s.capacity if s.capacity > 0 else 0.0
            color = _risk_color(util)
            risk = s.risk_scores.get(disaster_type, 0.0)
            folium.CircleMarker(
                location=[s.centroid_lat, s.centroid_lon],
                radius=10,
                color="#2ca25f",
                fill_color=color,
                fill=True,
                fill_opacity=0.8,
                tooltip=folium.Tooltip(
                    f"<b>{s.name}</b><br>"
                    f"Type: {s.shelter_type}<br>"
                    f"Capacity: {s.capacity:,}<br>"
                    f"Assigned: {assigned:,} ({100*util:.0f}%)<br>"
                    f"Risk: {risk:.2f}"
                ),
            ).add_to(fg_shelters)

        # --- Routes ---
        # Build node coord lookup from routes
        for vid, routes in routes_by_village.items():
            if not routes:
                continue
            for rank, route in enumerate(routes[:3]):
                color = ROUTE_COLORS[rank % len(ROUTE_COLORS)]
                opacity = 0.8 - rank * 0.2

                # Draw route as polyline if we have node coordinates
                if hasattr(route, "_node_coords") and route._node_coords:
                    coords = [(route._node_coords[n][0], route._node_coords[n][1])
                              for n in route.node_path if n in route._node_coords]
                    if len(coords) >= 2:
                        folium.PolyLine(
                            locations=coords,
                            color=color,
                            weight=3 - rank,
                            opacity=opacity,
                            tooltip=f"Route rank {rank+1}: {route.total_distance_km:.1f}km, "
                                    f"risk={route.avg_risk_score:.2f}",
                        ).add_to(fg_routes)
                else:
                    # Fallback: draw line from village to shelter centroid
                    v_obj = next((v for v in villages if v.village_id == vid), None)
                    s_obj = next((s for s in shelters if s.shelter_id == route.shelter_id), None)
                    if v_obj and s_obj:
                        folium.PolyLine(
                            locations=[
                                [v_obj.centroid_lat, v_obj.centroid_lon],
                                [s_obj.centroid_lat, s_obj.centroid_lon],
                            ],
                            color=color,
                            weight=2 + (1 if route.assigned_population > 0 else 0),
                            opacity=opacity,
                            tooltip=f"{'★ ' if route.assigned_population > 0 else ''}"
                                    f"{v_obj.name} → {s_obj.name} | "
                                    f"{route.total_distance_km:.1f}km | "
                                    f"risk={route.avg_risk_score:.2f}",
                        ).add_to(fg_routes)

        # --- Disaster location ---
        if disaster_location:
            folium.Marker(
                location=list(disaster_location),
                icon=folium.Icon(color="red", icon="warning-sign", prefix="glyphicon"),
                tooltip=folium.Tooltip("<b>⚠ Disaster Location</b>"),
            ).add_to(fg_disaster)
            folium.Circle(
                location=list(disaster_location),
                radius=5000,
                color="red",
                fill=False,
                opacity=0.4,
            ).add_to(fg_disaster)

        # --- Risk heatmap ---
        heat_data = []
        for v in villages:
            risk = v.risk_scores.get(disaster_type, 0.0)
            if risk > 0:
                heat_data.append([v.centroid_lat, v.centroid_lon, risk])

        if heat_data:
            fg_heat = folium.FeatureGroup(name="Risk heatmap", show=False)
            HeatMap(heat_data, radius=25, blur=15, max_zoom=13).add_to(fg_heat)
            fg_heat.add_to(m)

        # Add layers to map
        for fg in [fg_villages, fg_shelters, fg_routes, fg_disaster]:
            fg.add_to(m)

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
