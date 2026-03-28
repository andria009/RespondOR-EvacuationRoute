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
    shelter_tags: dict = field(default_factory=lambda: {
        "emergency": ["assembly_point", "shelter"],
        "amenity": ["community_centre", "hospital", "clinic",
                    "place_of_worship", "school", "university"],
        "building": ["public", "civic", "government", "mosque",
                     "church", "cathedral", "temple"]
    })
    village_admin_level: int = 9
    inarisk_batch_size: int = 20
    inarisk_rate_limit_s: float = 1.0
    population_csv: Optional[str] = None       # path to pop CSV
    shelter_capacity_csv: Optional[str] = None # path to capacity CSV
    # Area per person estimate for capacity estimation from polygon area
    m2_per_person: float = 2.0
    # Population density default (persons/km²) if no data
    default_pop_density: float = 500.0


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
class SimulationConfig:
    gama_executable: str = "gama-headless"
    gaml_model_path: str = "simulation/models/EvacuationModel.gaml"
    gama_output_dir: str = "output/simulation"
    n_runs: int = 5
    max_simulation_steps: int = 500
    time_step_minutes: float = 1.0
    parallel_runs: int = 1
    random_seed_base: int = 42


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
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
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
    extraction = ExtractionConfig(
        osm_cache_dir=extraction_raw.get("osm_cache_dir", "data/raw/osm_cache"),
        use_cached_osm=extraction_raw.get("use_cached_osm", True),
        network_type=extraction_raw.get("network_type", "all"),
        inarisk_batch_size=int(extraction_raw.get("inarisk_batch_size", 20)),
        inarisk_rate_limit_s=float(extraction_raw.get("inarisk_rate_limit_s", 1.0)),
        population_csv=extraction_raw.get("population_csv"),
        shelter_capacity_csv=extraction_raw.get("shelter_capacity_csv"),
        m2_per_person=float(extraction_raw.get("m2_per_person", 2.0)),
        default_pop_density=float(extraction_raw.get("default_pop_density", 500.0)),
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

    sim_raw = raw.get("simulation", {})
    simulation = SimulationConfig(
        gama_executable=sim_raw.get("gama_executable", "gama-headless"),
        gaml_model_path=sim_raw.get("gaml_model_path", "simulation/models/EvacuationModel.gaml"),
        gama_output_dir=sim_raw.get("gama_output_dir", "output/simulation"),
        n_runs=int(sim_raw.get("n_runs", 5)),
        max_simulation_steps=int(sim_raw.get("max_simulation_steps", 500)),
        parallel_runs=int(sim_raw.get("parallel_runs", 1)),
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
        simulation=simulation,
        execution=execution,
        log_level=raw.get("log_level", "INFO"),
        preloaded_network_json=raw.get("preloaded_network_json"),
        preloaded_network_pycgr=raw.get("preloaded_network_pycgr"),
        preloaded_villages_geojson=raw.get("preloaded_villages_geojson"),
        preloaded_shelters_geojson=raw.get("preloaded_shelters_geojson"),
        preloaded_poi_csv=raw.get("preloaded_poi_csv"),
        benchmark_village_limit=int(raw.get("benchmark_village_limit", 0)),
    )
