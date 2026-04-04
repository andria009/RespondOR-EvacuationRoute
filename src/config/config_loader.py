"""
Configuration loader for RespondOR-EvacuationRoute.
Supports YAML/JSON configs with environment variable overrides.
"""

import os
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, field

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

logger = logging.getLogger(__name__)


@dataclass
class DisasterConfig:
    name: str
    lat: float
    lon: float
    disaster_type: str              # earthquake|volcano|flood|landslide
    severity: float = 1.0


@dataclass
class RegionConfig:
    region_type: str                # bbox|circle
    # bbox: [south, west, north, east]
    bbox: Optional[list] = None
    center: Optional[list] = None  # [lat, lon]
    radius_km: Optional[float] = None


@dataclass
class ExtractionConfig:
    osm_cache_dir: str = "data/raw/osm_cache"
    use_cached_osm: bool = True
    network_type: str = "drive"     # drive|walk|all
    include_path_types: list = field(default_factory=lambda: [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "unclassified", "residential", "service", "living_street",
        "pedestrian", "footway", "path", "track"
    ])
    # Shelter polygon extraction.
    # Only Polygon/MultiPolygon features matching these tags AND with area
    # >= shelter_min_area_m2 are kept. Point nodes are discarded.
    # Default: hospitals + official assembly points only.
    # Add more types (e.g. place_of_worship, school, university) as needed.
    shelter_tags: dict = field(default_factory=lambda: {
        "amenity":   ["hospital"],
        "emergency": ["assembly_point", "shelter"],
    })
    # Minimum polygon area to qualify as a shelter (m²)
    shelter_min_area_m2: float = 500.0
    # Floor area per occupant for capacity estimation (m²/person)
    shelter_m2_per_person: float = 2.0
    # Road network extraction.
    # Keys are OSM highway types to include; set to null/empty to exclude a type.
    # Each entry: {speed_kmh: <default speed>, capacity_veh_h: <vehicles/hour/lane>}
    # Omit a highway type to exclude it from extraction entirely.
    road_types: dict = field(default_factory=lambda: {
        "motorway":      {"speed_kmh": 100, "capacity_veh_h": 2200},
        "trunk":         {"speed_kmh":  80, "capacity_veh_h": 2000},
        "primary":       {"speed_kmh":  60, "capacity_veh_h": 1800},
        "secondary":     {"speed_kmh":  50, "capacity_veh_h": 1500},
        "tertiary":      {"speed_kmh":  40, "capacity_veh_h": 1200},
        "unclassified":  {"speed_kmh":  30, "capacity_veh_h":  800},
        "residential":   {"speed_kmh":  30, "capacity_veh_h":  600},
        "service":       {"speed_kmh":  20, "capacity_veh_h":  400},
        "living_street": {"speed_kmh":  15, "capacity_veh_h":  200},
        "track":         {"speed_kmh":  15, "capacity_veh_h":  100},
        "path":          {"speed_kmh":   8, "capacity_veh_h":   50},
        "footway":       {"speed_kmh":   5, "capacity_veh_h":   50},
    })

    # Village extraction sources — ordered list; each source adds settlements not
    # already covered by previous sources (coverage = centroid inside existing polygon).
    # Options: "admin_boundary" | "place_nodes" | "building_clusters"
    #   admin_boundary   — OSM boundary=administrative polygons (most accurate)
    #   place_nodes      — OSM place=village|hamlet|... point nodes → synthetic circles
    #   building_clusters — DBSCAN-grouped building footprints → synthetic polygons
    # Default covers admin boundaries first, place nodes fill remaining gaps.
    village_sources: list = field(default_factory=lambda: ["admin_boundary", "place_nodes"])

    # admin_boundary source: ordered list of OSM admin levels to try
    #   9 = desa/kelurahan, 8 = kecamatan, 7 = kabupaten
    village_admin_levels: list = field(default_factory=lambda: [9, 8, 7])

    # place_nodes source: which OSM place values to query
    village_place_tags: list = field(default_factory=lambda: [
        "village", "hamlet", "town", "suburb", "quarter"
    ])
    # Per-place-tag radius (metres) and population density (persons/km²).
    # Tags absent from this dict fall back to village_place_radius_m (or auto)
    # and village_pop_density.
    village_place_settings: dict = field(default_factory=lambda: {
        "town":    {"radius_m": 2000, "pop_density": 3000},
        "suburb":  {"radius_m": 1200, "pop_density": 4000},
        "quarter": {"radius_m":  600, "pop_density": 3500},
        "village": {"radius_m":  800, "pop_density": 1200},
        "hamlet":  {"radius_m":  300, "pop_density":  600},
    })
    # Global fallback radius when a place tag is not in village_place_settings;
    # null = auto-compute from village_pop_density (target ~2000 persons/circle)
    village_place_radius_m: Optional[float] = None

    # building_clusters source: DBSCAN clustering parameters
    village_cluster_eps_m: float = 300.0       # cluster radius (metres)
    village_cluster_min_buildings: int = 10    # min buildings to form a cluster
    # Default occupancy when building type is not in village_building_persons
    village_persons_per_dwelling: float = 4.0
    # Per-OSM-building-type occupancy (persons). Types absent from this dict use
    # village_persons_per_dwelling as fallback. Set a type to 0 to exclude it
    # from population counting (e.g. commercial, religious buildings).
    village_building_persons: dict = field(default_factory=lambda: {
        # residential
        "house":              4.5,
        "detached":           4.5,
        "semidetached_house": 4.0,
        "terrace":            3.5,
        "bungalow":           4.0,
        "cabin":              2.0,
        "residential":        6.0,   # small multi-unit
        "apartments":        20.0,
        "dormitory":         30.0,
        "farm":               5.0,   # extended family
        "yes":                4.0,   # untagged — assume residential
        # non-residential → 0 (excluded from population count)
        "commercial":         0.0,
        "retail":             0.0,
        "industrial":         0.0,
        "warehouse":          0.0,
        "office":             0.0,
        "school":             0.0,
        "hospital":           0.0,
        "clinic":             0.0,
        "church":             0.0,
        "mosque":             0.0,
        "temple":             0.0,
        "cathedral":          0.0,
        "civic":              0.0,
        "government":         0.0,
        "public":             0.0,
        "service":            0.0,
        "garage":             0.0,
        "garages":            0.0,
        "shed":               0.0,
        "roof":               0.0,
        "greenhouse":         0.0,
    })

    # Population density (persons/km²) for area-based estimation
    village_pop_density: float = 800.0
    # Upper cap on estimated population per village (prevents inflated estimates
    # on large polygons or when a higher admin level is used as fallback)
    village_max_pop: int = 50000
    inarisk_batch_size: int = 20
    inarisk_rate_limit_s: float = 1.0
    population_csv: Optional[str] = None       # path to pop CSV
    shelter_capacity_csv: Optional[str] = None # path to capacity CSV
    # Area per person estimate for capacity estimation from polygon area
    m2_per_person: float = 2.0


@dataclass
class RoutingConfig:
    max_routes_per_village: int = 3
    max_route_risk_threshold: float = 0.8
    # Composite score weights
    weight_distance: float = 0.3
    weight_risk: float = 0.4
    weight_road_quality: float = 0.2
    weight_time: float = 0.1
    # Congestion
    bpr_alpha: float = 0.15
    bpr_beta: float = 4.0
    # Population estimation area fallback
    area_m2_per_person: float = 100.0


@dataclass
class ExecutionConfig:
    mode: str = "naive"             # naive|parallel|hpc
    n_workers: int = 4
    # HPC/MPI settings
    mpi_hosts_file: Optional[str] = None   # optional hostfile for mpirun outside SLURM


@dataclass
class AppConfig:
    scenario_id: str
    output_dir: str
    disaster: DisasterConfig
    region: RegionConfig
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    log_level: str = "INFO"

    # Paths to pre-extracted data (skip extraction if provided)
    preloaded_network_json: Optional[str] = None
    preloaded_network_pycgr: Optional[str] = None
    preloaded_villages_geojson: Optional[str] = None
    preloaded_shelters_geojson: Optional[str] = None
    # Legacy POI CSV (RespondOR v1): name,type,lat,lon — loads both villages+shelters
    preloaded_poi_csv: Optional[str] = None
    # Limit villages processed (for benchmarking; 0 = no limit)
    benchmark_village_limit: int = 0


def load_config(config_path: Union[str, Path]) -> AppConfig:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        if config_path.suffix in (".yaml", ".yml"):
            if not HAS_YAML:
                raise ImportError("PyYAML is required for YAML configs: pip install pyyaml")
            raw = yaml.safe_load(f)
        else:
            raw = json.load(f)

    return _parse_config(raw)


def _parse_config(raw: Dict[str, Any]) -> AppConfig:
    """Parse raw dict into AppConfig dataclass."""
    disaster_raw = raw.get("disaster", {})
    disaster = DisasterConfig(
        name=disaster_raw.get("name", "event"),
        lat=float(disaster_raw["lat"]),
        lon=float(disaster_raw["lon"]),
        disaster_type=disaster_raw.get("type", "earthquake"),
        severity=float(disaster_raw.get("severity", 1.0)),
    )

    region_raw = raw.get("region", {})
    region = RegionConfig(
        region_type=region_raw.get("type", "circle"),
        bbox=region_raw.get("bbox"),
        center=region_raw.get("center"),
        radius_km=region_raw.get("radius_km", 20.0),
    )
    if region.region_type == "circle" and region.center is None:
        region.center = [disaster.lat, disaster.lon]

    extraction_raw = raw.get("extraction", {})
    # village_admin_levels: accept list or single int (backward compat)
    _al = extraction_raw.get("village_admin_levels",
                             extraction_raw.get("village_admin_level", [9, 8, 7]))
    village_admin_levels = [int(_al)] if isinstance(_al, (int, float)) else [int(x) for x in _al]
    # village_sources: list of extraction strategies (strings)
    _vs_raw = extraction_raw.get("village_sources", ["admin_boundary", "place_nodes"])
    village_sources = [str(s) for s in _vs_raw] if isinstance(_vs_raw, list) else [str(_vs_raw)]
    _place_tags_raw = extraction_raw.get("village_place_tags",
                                         ["village", "hamlet", "town", "suburb", "quarter"])
    village_place_tags = list(_place_tags_raw)
    _shelter_tags = extraction_raw.get("shelter_tags", None)
    _road_types_raw = extraction_raw.get("road_types", None)
    _default_road_types = {
        "motorway":      {"speed_kmh": 100, "capacity_veh_h": 2200},
        "trunk":         {"speed_kmh":  80, "capacity_veh_h": 2000},
        "primary":       {"speed_kmh":  60, "capacity_veh_h": 1800},
        "secondary":     {"speed_kmh":  50, "capacity_veh_h": 1500},
        "tertiary":      {"speed_kmh":  40, "capacity_veh_h": 1200},
        "unclassified":  {"speed_kmh":  30, "capacity_veh_h":  800},
        "residential":   {"speed_kmh":  30, "capacity_veh_h":  600},
        "service":       {"speed_kmh":  20, "capacity_veh_h":  400},
        "living_street": {"speed_kmh":  15, "capacity_veh_h":  200},
        "track":         {"speed_kmh":  15, "capacity_veh_h":  100},
        "path":          {"speed_kmh":   8, "capacity_veh_h":   50},
        "footway":       {"speed_kmh":   5, "capacity_veh_h":   50},
    }
    _road_types = _road_types_raw if _road_types_raw is not None else _default_road_types
    extraction = ExtractionConfig(
        osm_cache_dir=extraction_raw.get("osm_cache_dir", "data/raw/osm_cache"),
        use_cached_osm=extraction_raw.get("use_cached_osm", True),
        network_type=extraction_raw.get("network_type", "all"),
        road_types=_road_types,
        village_sources=village_sources,
        village_admin_levels=village_admin_levels,
        village_place_tags=village_place_tags,
        village_place_settings={
            **ExtractionConfig.__dataclass_fields__["village_place_settings"].default_factory(),
            **(extraction_raw.get("village_place_settings") or {}),
        },
        village_place_radius_m=extraction_raw.get("village_place_radius_m"),
        village_cluster_eps_m=float(extraction_raw.get("village_cluster_eps_m", 300.0)),
        village_cluster_min_buildings=int(extraction_raw.get("village_cluster_min_buildings", 10)),
        village_persons_per_dwelling=float(extraction_raw.get("village_persons_per_dwelling", 4.0)),
        village_building_persons={
            **ExtractionConfig.__dataclass_fields__["village_building_persons"].default_factory(),
            **(extraction_raw.get("village_building_persons") or {}),
        },
        village_pop_density=float(extraction_raw.get("village_pop_density",
                                  extraction_raw.get("default_pop_density", 800.0))),
        village_max_pop=int(extraction_raw.get("village_max_pop", 50000)),
        shelter_tags=_shelter_tags if _shelter_tags is not None else {
            "amenity":   ["hospital"],
            "emergency": ["assembly_point", "shelter"],
        },
        shelter_min_area_m2=float(extraction_raw.get("shelter_min_area_m2", 500.0)),
        shelter_m2_per_person=float(extraction_raw.get("shelter_m2_per_person",
                                    extraction_raw.get("m2_per_person", 2.0))),
        inarisk_batch_size=int(extraction_raw.get("inarisk_batch_size", 20)),
        inarisk_rate_limit_s=float(extraction_raw.get("inarisk_rate_limit_s", 1.0)),
        population_csv=extraction_raw.get("population_csv"),
        shelter_capacity_csv=extraction_raw.get("shelter_capacity_csv"),
        m2_per_person=float(extraction_raw.get("m2_per_person", 2.0)),
    )

    routing_raw = raw.get("routing", {})
    routing = RoutingConfig(
        max_routes_per_village=int(routing_raw.get("max_routes_per_village", 3)),
        max_route_risk_threshold=float(routing_raw.get("max_route_risk_threshold", 0.8)),
        weight_distance=float(routing_raw.get("weight_distance", 0.3)),
        weight_risk=float(routing_raw.get("weight_risk", 0.4)),
        weight_road_quality=float(routing_raw.get("weight_road_quality", 0.2)),
        weight_time=float(routing_raw.get("weight_time", 0.1)),
    )

    exec_raw = raw.get("execution", {})
    execution = ExecutionConfig(
        mode=exec_raw.get("mode", "naive"),
        n_workers=int(exec_raw.get("n_workers", 4)),
        mpi_hosts_file=exec_raw.get("mpi_hosts_file"),
    )

    return AppConfig(
        scenario_id=raw.get("scenario_id", "scenario_001"),
        output_dir=raw.get("output_dir", "output"),
        disaster=disaster,
        region=region,
        extraction=extraction,
        routing=routing,
        execution=execution,
        log_level=raw.get("log_level", "INFO"),
        preloaded_network_json=raw.get("preloaded_network_json"),
        preloaded_network_pycgr=raw.get("preloaded_network_pycgr"),
        preloaded_villages_geojson=raw.get("preloaded_villages_geojson"),
        preloaded_shelters_geojson=raw.get("preloaded_shelters_geojson"),
        preloaded_poi_csv=raw.get("preloaded_poi_csv"),
        benchmark_village_limit=int(raw.get("benchmark_village_limit", 0)),
    )
