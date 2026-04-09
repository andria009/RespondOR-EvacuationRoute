"""
Pre-run validation for RespondOR-EvacuationRoute.

Checks config, estimates data counts, API call budget, and routing time
BEFORE running the full pipeline.  Flags problems so they can be fixed
without wasting hours of compute.

Usage:
    python -m experiments.prerun_validation --config configs/banjarnegara_landslide_2021.yaml
    python -m experiments.prerun_validation --config configs/merapi_eruption_2023.yaml --verbose
"""

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

from src.config.config_loader import load_config
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def _bbox_from_circle(lat, lon, radius_km):
    """Return (south, west, north, east) for a circular region."""
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return lat - deg_lat, lon - deg_lon, lat + deg_lat, lon + deg_lon


def _count_cached_poi_cache(cache_path, key):
    """Return number of cached points for a cache key in poi_risk_cache.json."""
    p = Path(cache_path)
    if not p.exists():
        return 0
    try:
        with open(p) as f:
            cache = json.load(f)
        return len(cache.get(key, {}))
    except Exception:
        return 0


def _count_grid_cache(cache_path, key):
    """Return number of cached grid cells in road_risk_cache.json for a key."""
    p = Path(cache_path)
    if not p.exists():
        return 0
    try:
        with open(p) as f:
            cache = json.load(f)
        return len(cache.get(key, {}))
    except Exception:
        return 0


def _estimate_village_count(cfg):
    """
    Rough estimate of building clusters based on region area and cluster radius.
    This is a heuristic — actual count depends on OSM building density.
    """
    if cfg.region.region_type == "circle":
        area_km2 = math.pi * cfg.region.radius_km ** 2
    elif cfg.region.bbox:
        s, w, n, e = cfg.region.bbox
        area_km2 = _haversine_km(s, w, s, e) * _haversine_km(s, w, n, w)
    else:
        area_km2 = math.pi * (cfg.region.radius_km or 10) ** 2

    # Buildings per km² (Java lowland ~200-500, highland ~50-200)
    bld_per_km2 = cfg.extraction.village_pop_density / 4.0  # rough: pop/household
    eps_m = cfg.extraction.village_cluster_eps_m
    cluster_area_m2 = math.pi * (eps_m / 2) ** 2
    cluster_area_km2 = cluster_area_m2 / 1e6
    buildings_in_region = area_km2 * bld_per_km2
    est_clusters = max(1, int(buildings_in_region * cluster_area_km2))
    # Cap: one cluster per ~0.5 km²
    return min(est_clusters, int(area_km2 / 0.5))


# ------------------------------------------------------------------ #
# Validation checks
# ------------------------------------------------------------------ #

class ValidationResult:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.info = []

    def error(self, msg):
        self.errors.append(f"[ERROR] {msg}")

    def warn(self, msg):
        self.warnings.append(f"[WARN]  {msg}")

    def note(self, msg):
        self.info.append(f"[INFO]  {msg}")

    def print_summary(self):
        for line in self.info:
            print(line)
        for line in self.warnings:
            print(line)
        for line in self.errors:
            print(line)
        print()
        if self.errors:
            print(f"VALIDATION FAILED — {len(self.errors)} error(s), {len(self.warnings)} warning(s)")
        elif self.warnings:
            print(f"VALIDATION PASSED WITH WARNINGS — {len(self.warnings)} warning(s)")
        else:
            print("VALIDATION PASSED")
        return len(self.errors) == 0


def check_config(cfg, vr: ValidationResult):
    """Check configuration values for obvious problems."""
    vr.note(f"Scenario:        {cfg.scenario_id}")
    vr.note(f"Disaster:        {cfg.disaster.disaster_type} @ ({cfg.disaster.lat}, {cfg.disaster.lon})")
    vr.note(f"Region:          {cfg.region.region_type}, radius={cfg.region.radius_km} km")
    vr.note(f"Execution mode:  {cfg.execution.mode}, n_workers={cfg.execution.n_workers}")
    vr.note(f"skip_inarisk:    {cfg.skip_inarisk}")

    # Weights sum check
    w_sum = (cfg.routing.weight_distance + cfg.routing.weight_risk
             + cfg.routing.weight_road_quality + cfg.routing.weight_time
             + cfg.routing.weight_disaster_distance)
    if abs(w_sum - 1.0) > 0.01:
        vr.warn(f"Routing weights sum to {w_sum:.3f} (expected ~1.0) — "
                f"scores will be offset but ranking still valid")

    if cfg.routing.min_routes_per_village > cfg.routing.max_routes_per_village:
        vr.error(f"min_routes_per_village ({cfg.routing.min_routes_per_village}) "
                 f"> max_routes_per_village ({cfg.routing.max_routes_per_village})")

    if cfg.extraction.village_cluster_eps_m > 200:
        vr.warn(f"village_cluster_eps_m={cfg.extraction.village_cluster_eps_m} m is large "
                f"(radius={cfg.extraction.village_cluster_eps_m/2:.0f} m) — may over-merge clusters")

    if cfg.extraction.village_cluster_eps_m < 50:
        vr.warn(f"village_cluster_eps_m={cfg.extraction.village_cluster_eps_m} m is small — "
                f"may create thousands of 1-building clusters, making routing very slow")

    if cfg.region.radius_km and cfg.region.radius_km > 30:
        vr.warn(f"Region radius {cfg.region.radius_km} km is large — "
                f"OSM extraction and InaRISK queries will be slow")


def check_osm_cache(cfg, vr: ValidationResult):
    """Check OSM cache files."""
    cache_dir = Path(cfg.extraction.osm_cache_dir)
    if not cache_dir.exists():
        vr.warn(f"OSM cache dir does not exist: {cache_dir}  (will be created on first run)")
        return

    # Check for network cache
    net_files = list(cache_dir.glob("network_*.json")) + list(cache_dir.glob("network_*.pycgr"))
    if net_files:
        vr.note(f"Network cache:   {len(net_files)} file(s) in {cache_dir}")
    else:
        vr.warn(f"No network cache found in {cache_dir} — road network will be queried from OSM")

    # Check for village/shelter partial caches (indicates prior interrupted run)
    partial_files = list(cache_dir.glob("*.partial.json"))
    if partial_files:
        vr.note(f"Partial cache files found ({len(partial_files)}) — extraction will resume from checkpoint")


def check_inarisk_cache(cfg, vr: ValidationResult, est_villages: int, est_shelters: int):
    """Check InaRISK cache and estimate API call budget."""
    if cfg.skip_inarisk:
        vr.note("InaRISK:         SKIPPED (skip_inarisk=true)")
        return

    cache_dir = Path(cfg.extraction.inarisk_cache_dir)
    poi_cache_path = cache_dir / "poi_risk_cache.json"
    grid_cache_path = cache_dir / "road_risk_cache.json"
    dt = cfg.disaster.disaster_type

    # Estimate expected grid cells for road edges
    if cfg.region.region_type == "circle":
        area_km2 = math.pi * cfg.region.radius_km ** 2
    else:
        area_km2 = 100.0  # rough fallback
    # Grid at precision=2 → ~1.1 km cells
    est_grid_cells = int(area_km2 / 1.21)

    # Check grid cache
    grid_key = f"edges_{dt}"
    n_grid_cached = _count_grid_cache(grid_cache_path, grid_key)
    vr.note(f"Road grid cache: {n_grid_cached}/{est_grid_cells} cells cached "
            f"(key='{grid_key}') in {grid_cache_path.name}")
    if n_grid_cached == 0:
        vr.warn(f"Road grid cache empty — edge InaRISK will query ~{est_grid_cells} grid cells "
                f"(~{est_grid_cells//cfg.extraction.inarisk_batch_size + 1} batches)")
    elif n_grid_cached < est_grid_cells * 0.5:
        vr.warn(f"Road grid cache only {n_grid_cached/max(est_grid_cells,1)*100:.0f}% complete")

    # Check POI cache
    vil_key = f"villages_{dt}"
    shel_key = f"shelters_{dt}"
    n_vil_cached = _count_cached_poi_cache(poi_cache_path, vil_key)
    n_shel_cached = _count_cached_poi_cache(poi_cache_path, shel_key)

    vr.note(f"Village risk cache: {n_vil_cached} cached (est. {est_villages} needed)")
    vr.note(f"Shelter risk cache: {n_shel_cached} cached (est. ~50-200 needed)")

    # Grid-snap coverage: grid cells can serve as POI lookup too
    if n_grid_cached > 0:
        vr.note(f"Grid-snap enabled: POI lookups will first try road_risk_cache.json "
                f"(snapping to {grid_cache_path.name})")

    # Estimate uncached POI API calls
    missing_vil = max(0, est_villages - n_vil_cached)
    missing_shel = max(0, est_shelters - n_shel_cached)
    if n_grid_cached > 0:
        # Most POIs will be covered by grid snap
        missing_vil = max(0, missing_vil - int(missing_vil * 0.85))
        missing_shel = max(0, missing_shel - int(missing_shel * 0.85))

    if missing_vil + missing_shel > 0:
        est_calls = missing_vil + missing_shel
        est_time_s = est_calls * (1.0 / cfg.extraction.inarisk_batch_size
                                  * cfg.extraction.inarisk_rate_limit_s
                                  + 0.5)  # 0.5s per point (sequential in batch)
        vr.note(f"Estimated InaRISK POI calls: {est_calls} "
                f"(~{est_time_s/60:.1f} min at {cfg.extraction.inarisk_rate_limit_s}s rate limit)")
        if est_time_s > 600:
            vr.warn(f"POI InaRISK may take >{est_time_s/60:.0f} min — "
                    f"ensure use_cached_inarisk=true or grid cache is populated")


def check_routing_estimate(cfg, vr: ValidationResult, est_villages: int):
    """Estimate routing time based on village count and execution mode."""
    # Rough: Dijkstra on a 50k-node graph for one village ≈ 0.1s sequential
    t_per_village_s = 0.10
    total_s = est_villages * t_per_village_s

    if cfg.execution.mode == "parallel":
        total_s /= max(1, cfg.execution.n_workers)

    vr.note(f"Routing estimate: ~{est_villages} villages × {t_per_village_s:.2f}s "
            f"({'÷' + str(cfg.execution.n_workers) if cfg.execution.mode == 'parallel' else 'sequential'}) "
            f"≈ {total_s/60:.1f} min")

    if total_s > 600 and cfg.execution.mode == "naive":
        vr.warn(f"Routing in NAIVE mode with ~{est_villages} villages may take "
                f">{total_s/60:.0f} min — consider mode: parallel")
    if est_villages > 5000 and cfg.execution.mode == "parallel" and cfg.execution.n_workers < 4:
        vr.warn(f"Large village count ({est_villages}) with only {cfg.execution.n_workers} workers — "
                f"increase n_workers to speed up routing")

    if cfg.routing.min_routes_per_village > cfg.routing.max_routes_per_village:
        vr.error("min_routes_per_village > max_routes_per_village — no routes will be guaranteed")


def check_output_dir(cfg, vr: ValidationResult):
    """Check if output dir is writable; warn if it already has results."""
    out = Path(cfg.output_dir)
    if out.exists():
        existing = list(out.glob("*.json")) + list(out.glob("*.csv"))
        if existing:
            vr.note(f"Output dir {out} has {len(existing)} existing file(s) — will be overwritten")
    else:
        vr.note(f"Output dir {out} will be created")


def check_preloaded_paths(cfg, vr: ValidationResult):
    """Check all preloaded file paths exist."""
    checks = [
        ("preloaded_network_json", cfg.preloaded_network_json),
        ("preloaded_network_pycgr", cfg.preloaded_network_pycgr),
        ("preloaded_villages_geojson", cfg.preloaded_villages_geojson),
        ("preloaded_shelters_geojson", cfg.preloaded_shelters_geojson),
        ("preloaded_poi_csv", cfg.preloaded_poi_csv),
        ("population_csv", cfg.extraction.population_csv),
        ("shelter_capacity_csv", cfg.extraction.shelter_capacity_csv),
    ]
    for name, path in checks:
        if path and not Path(path).exists():
            vr.error(f"{name} = '{path}' does not exist")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def validate(config_path: str, verbose: bool = False) -> bool:
    cfg = load_config(config_path)
    vr = ValidationResult()

    print(f"\n{'='*60}")
    print(f"  Pre-run Validation: {config_path}")
    print(f"{'='*60}\n")

    check_config(cfg, vr)
    check_preloaded_paths(cfg, vr)
    check_osm_cache(cfg, vr)

    est_villages = _estimate_village_count(cfg)
    est_shelters = 100  # rough: most scenarios have 50–300 shelter polygons
    vr.note(f"Village estimate: ~{est_villages} clusters "
            f"(eps={cfg.extraction.village_cluster_eps_m} m, "
            f"radius={cfg.region.radius_km} km)")

    check_inarisk_cache(cfg, vr, est_villages, est_shelters)
    check_routing_estimate(cfg, vr, est_villages)
    check_output_dir(cfg, vr)

    print()
    passed = vr.print_summary()
    return passed


def main():
    p = argparse.ArgumentParser(description="Pre-run validation for RespondOR")
    p.add_argument("--config", "-c", required=True, help="Path to YAML config file")
    p.add_argument("--verbose", "-v", action="store_true", help="Extra detail")
    args = p.parse_args()

    setup_logging("prerun_validation", level="WARNING")
    ok = validate(args.config, verbose=args.verbose)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
