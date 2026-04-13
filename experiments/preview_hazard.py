"""
Hazard risk index preview — generates preview_hazard.html per scenario.

Queries InaRISK BNPB for available hazard layers over a grid of points
covering the scenario bbox, then renders a colour-coded choropleth map.

Earthquake layer is currently unavailable (INDEKS_BAHAYA_GEMPABUMI down);
scenarios whose only hazard is earthquake are skipped with a note.

Results are cached to data/raw/inarisk_cache/hazard_grid_cache.json (shared
across all scenarios) so overlapping regions reuse cached grid points.

Usage:
  python -m experiments.preview_hazard --config configs/merapi_eruption_2023.yaml
  python -m experiments.preview_hazard --all
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.utils.logging_setup import setup_logging as _setup_logging
_setup_logging("preview_hazard")
logger = logging.getLogger("preview_hazard")

# Hazard layers available from InaRISK (earthquake currently down)
AVAILABLE_LAYERS = {"volcano", "landslide", "flood", "tsunami", "liquefaction", "flash_flood"}
UNAVAILABLE_LAYERS = {"earthquake"}

# Colour ramps per hazard type (low → high risk)
HAZARD_COLORS = {
    "volcano":      ["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
    "landslide":    ["#ffffcc", "#c7e9b4", "#7fcdbb", "#2c7fb8", "#253494"],
    "flood":        ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#084594"],
    "tsunami":      ["#fff5eb", "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704"],
    "liquefaction": ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"],
    "flash_flood":  ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"],
}

HAZARD_LABELS = {
    "volcano":      "Volcano Hazard Index",
    "landslide":    "Landslide Hazard Index",
    "flood":        "Flood Hazard Index",
    "tsunami":      "Tsunami Hazard Index",
    "liquefaction": "Liquefaction Hazard Index",
    "flash_flood":  "Flash Flood Hazard Index",
}


def _score_to_color(score: float, colors: List[str]) -> str:
    """Map score [0,1] to a color from a 5-step ramp."""
    idx = min(int(score * len(colors)), len(colors) - 1)
    return colors[idx]


def _score_label(score: float) -> str:
    if score <= 0.0:
        return "No data"
    if score < 0.2:
        return f"Very Low ({score:.2f})"
    if score < 0.4:
        return f"Low ({score:.2f})"
    if score < 0.6:
        return f"Medium ({score:.2f})"
    if score < 0.8:
        return f"High ({score:.2f})"
    return f"Very High ({score:.2f})"


def _grid_points(
    bbox: Tuple[float, float, float, float],
    step_km: float = 1.0,
) -> List[Tuple[float, float]]:
    """Generate a lat/lon grid over the bbox with ~step_km spacing."""
    south, west, north, east = bbox
    dlat = step_km / 111.0
    mid_lat = (south + north) / 2
    dlon = step_km / (111.0 * math.cos(math.radians(mid_lat)))

    pts = []
    lat = south
    while lat <= north:
        lon = west
        while lon <= east:
            pts.append((round(lat, 6), round(lon, 6)))
            lon += dlon
        lat += dlat
    return pts


def _save_hazard_cache(cache_path: Path, hazard_type: str, cache: Dict[str, float]) -> None:
    """Merge and save hazard cache for one layer to disk."""
    full_cache: Dict = {}
    if cache_path.exists():
        with open(cache_path) as f:
            full_cache = json.load(f)
    full_cache[hazard_type] = cache
    with open(cache_path, "w") as f:
        json.dump(full_cache, f)


def query_hazard_grid(
    bbox: Tuple[float, float, float, float],
    hazard_type: str,
    cache_path: Path,
    step_km: float = 1.0,
    batch_size: int = 20,
    rate_limit_s: float = 1.0,
    save_every_n_batches: int = 50,
    use_cache: bool = True,
) -> Dict[str, float]:
    """
    Query InaRISK for hazard scores on a grid. Results keyed by "lat,lon".
    Loads existing cache and only queries missing points.
    Saves cache incrementally every save_every_n_batches batches to avoid
    losing progress on large grids (e.g. Palu 50km radius = 10k+ points).
    """
    from src.data.models import DisasterType
    from src.data.inarisk_client import InaRISKClient

    cache: Dict[str, float] = {}
    if use_cache and cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f).get(hazard_type, {})

    grid = _grid_points(bbox, step_km)
    missing = [(lat, lon) for lat, lon in grid
               if f"{lat:.6f},{lon:.6f}" not in cache]

    if not missing:
        logger.info(f"  [{hazard_type}] All {len(grid)} points cached")
        return cache

    n_cached = len(grid) - len(missing)
    logger.info(f"  [{hazard_type}] Querying {len(missing)} points "
                f"({n_cached} cached) …")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dt = DisasterType(hazard_type)
    client = InaRISKClient(batch_size=batch_size, rate_limit_s=rate_limit_s)

    batches_done = 0
    for batch_pts, batch_scores in client.iter_risk_scores_batched(missing, dt):
        for (lat, lon), score in zip(batch_pts, batch_scores):
            cache[f"{lat:.6f},{lon:.6f}"] = score
        batches_done += 1
        if batches_done % save_every_n_batches == 0:
            pts_done = min(batches_done * batch_size, len(missing))
            logger.info(f"  [{hazard_type}] {pts_done}/{len(missing)} points done — saving cache …")
            _save_hazard_cache(cache_path, hazard_type, cache)

    _save_hazard_cache(cache_path, hazard_type, cache)
    return cache


def build_hazard_map(
    bbox: Tuple[float, float, float, float],
    hazard_scores: Dict[str, Dict[str, float]],
    disaster_lat: float,
    disaster_lon: float,
    disaster_name: str,
    radius_km: float,
    step_km: float,
) -> object:
    """Build a Folium map with hazard grid layers."""
    import folium

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

    legend_rows = []

    for hazard_type, scores in hazard_scores.items():
        colors = HAZARD_COLORS.get(hazard_type, HAZARD_COLORS["landslide"])
        label = HAZARD_LABELS.get(hazard_type, hazard_type.replace("_", " ").title())

        grp = folium.FeatureGroup(name=label, show=True)

        nonzero = [v for v in scores.values() if v > 0]
        max_score = max(nonzero) if nonzero else 1.0
        coverage = len(nonzero)
        total = len(scores)

        for key, score in scores.items():
            if score <= 0.0:
                continue
            lat_s, lon_s = key.split(",")
            lat, lon = float(lat_s), float(lon_s)
            color = _score_to_color(score / max_score if max_score > 0 else 0, colors)
            # Each grid point rendered as a small rectangle
            half = step_km / 111.0 / 2
            folium.Rectangle(
                bounds=[[lat - half, lon - half], [lat + half, lon + half]],
                color=None, weight=0,
                fill=True, fill_color=color, fill_opacity=0.65,
                tooltip=f"{label}: {_score_label(score)}",
            ).add_to(grp)

        grp.add_to(m)

        # Legend row: color swatch + stats
        mean_score = sum(nonzero) / len(nonzero) if nonzero else 0.0
        swatches = "".join(
            f'<span style="display:inline-block;width:14px;height:10px;'
            f'background:{c};margin-right:1px"></span>'
            for c in colors
        )
        legend_rows.append(
            f'<tr>'
            f'<td style="padding:3px 6px">{label}<br>'
            f'<span style="font-size:10px">{swatches} 0 → 1</span></td>'
            f'<td style="padding:3px 6px;text-align:right">{coverage:,}/{total:,}</td>'
            f'<td style="padding:3px 6px;text-align:right">{mean_score:.2f}</td>'
            f'</tr>'
        )

    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = f"""
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:12px 16px;border-radius:8px;
                box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:12px;
                font-family:sans-serif;min-width:320px">
      <b style="font-size:13px">Hazard Risk Index — {disaster_name}</b>
      <table style="margin-top:8px;width:100%;border-collapse:collapse">
        <tr style="background:#f5f5f5">
          <th style="padding:3px 6px;text-align:left">Layer</th>
          <th style="padding:3px 6px;text-align:right">Points</th>
          <th style="padding:3px 6px;text-align:right">Mean</th>
        </tr>
        {"".join(legend_rows)}
      </table>
      <div style="margin-top:6px;padding-top:6px;border-top:1px solid #ddd;font-size:11px">
        <span style="color:#888">Grid: ~{step_km} km · Source: InaRISK BNPB</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m


def run_scenario(cfg_path: Path, step_km: float = 1.0) -> Optional[Path]:
    """
    Run hazard preview for a single scenario config.
    Returns output path, or None if skipped.
    """
    from src.config.config_loader import load_config
    from src.data.models import RegionOfInterest, RegionType

    cfg = load_config(cfg_path)

    # Determine which hazard layers to query
    if cfg.routing.hazard_layers:
        requested = set(cfg.routing.hazard_layers.keys())
    else:
        requested = {cfg.disaster.disaster_type}

    available = requested & AVAILABLE_LAYERS
    skipped = requested & UNAVAILABLE_LAYERS

    if skipped:
        logger.warning(f"  Skipping unavailable layers: {skipped}")
    if not available:
        logger.warning(
            f"  [{cfg.scenario_id}] No available hazard layers — skipping preview"
        )
        return None

    region = RegionOfInterest(
        region_type=RegionType(cfg.region.region_type),
        center=tuple(cfg.region.center) if cfg.region.center else None,
        bbox=tuple(cfg.region.bbox) if cfg.region.bbox else None,
        radius_km=cfg.region.radius_km,
    )
    bbox = region.to_bbox()

    cache_path = Path(cfg.extraction.inarisk_cache_dir) / "hazard_grid_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    south, west, north, east = bbox

    # Query available layers — cache is shared across scenarios (overlapping regions
    # reuse cached points). After querying, filter to this scenario's bbox so that
    # hazard data from neighbouring scenarios does not bleed into this map.
    hazard_scores: Dict[str, Dict[str, float]] = {}
    for hazard_type in sorted(available):
        logger.info(f"  Querying {hazard_type} …")
        all_scores = query_hazard_grid(
            bbox=bbox,
            hazard_type=hazard_type,
            cache_path=cache_path,
            step_km=step_km,
            batch_size=cfg.extraction.inarisk_batch_size,
            rate_limit_s=cfg.extraction.inarisk_rate_limit_s,
            use_cache=cfg.extraction.use_cached_inarisk,
        )
        # Clip to this scenario's bbox — cache may contain points from other scenarios
        scores = {
            k: v for k, v in all_scores.items()
            if south <= float(k.split(",")[0]) <= north
            and west  <= float(k.split(",")[1]) <= east
        }
        hazard_scores[hazard_type] = scores

    # Build and save map
    out_path = Path(cfg.output_dir) / "preview_hazard.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Building map …")
    m = build_hazard_map(
        bbox=bbox,
        hazard_scores=hazard_scores,
        disaster_lat=cfg.disaster.lat,
        disaster_lon=cfg.disaster.lon,
        disaster_name=cfg.disaster.name,
        radius_km=cfg.region.radius_km,
        step_km=step_km,
    )
    m.save(str(out_path))
    logger.info(f"  Saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate hazard risk index preview maps from InaRISK"
    )
    parser.add_argument("--config", default=None, help="Path to a single scenario YAML")
    parser.add_argument("--all",    action="store_true", help="Run all configs/*.yaml")
    parser.add_argument("--step-km", type=float, default=1.0,
                        help="Grid spacing in km (default: 1.0)")
    args = parser.parse_args()

    if args.all:
        configs = sorted(Path("configs").glob("*.yaml"))
    elif args.config:
        configs = [Path(args.config)]
    else:
        parser.error("Provide --config or --all")

    for cfg_path in configs:
        logger.info(f"=== {cfg_path.stem} ===")
        try:
            out = run_scenario(cfg_path, step_km=args.step_km)
            if out:
                logger.info(f"  Done: {out}")
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)


if __name__ == "__main__":
    main()
