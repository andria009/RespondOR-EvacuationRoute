"""
Data models for the RespondOR-EvacuationRoute system.
Uses dataclasses for lightweight structured data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum


class DisasterType(str, Enum):
    EARTHQUAKE = "earthquake"
    VOLCANO = "volcano"
    FLOOD = "flood"
    LANDSLIDE = "landslide"


class ExecutionMode(str, Enum):
    NAIVE = "naive"
    PARALLEL = "parallel"
    HPC = "hpc"


class RegionType(str, Enum):
    BBOX = "bbox"
    CIRCLE = "circle"


@dataclass
class DisasterInput:
    """Disaster event parameters."""
    location: Tuple[float, float]       # (lat, lon)
    disaster_type: DisasterType
    name: str = "disaster_event"
    severity: float = 1.0               # 0.0-1.0 scaling factor


@dataclass
class RegionOfInterest:
    """Geographic region for data extraction."""
    region_type: RegionType
    # For BBOX: (south, west, north, east)
    bbox: Optional[Tuple[float, float, float, float]] = None
    # For CIRCLE: center + radius_km
    center: Optional[Tuple[float, float]] = None
    radius_km: Optional[float] = None

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """Return bounding box (south, west, north, east)."""
        if self.region_type == RegionType.BBOX:
            return self.bbox
        import math
        lat, lon = self.center
        r = self.radius_km
        # Approximate degree offsets
        dlat = r / 111.0
        dlon = r / (111.0 * math.cos(math.radians(lat)))
        return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


@dataclass
class NetworkNode:
    """Road network node."""
    node_id: int
    lat: float
    lon: float
    risk_scores: Dict[str, float] = field(default_factory=dict)  # hazard -> score [0-1]

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.lat, self.lon)


@dataclass
class NetworkEdge:
    """Road network edge."""
    source_id: int
    target_id: int
    length_m: float
    highway_type: str = "residential"
    max_speed_kmh: float = 30.0
    bidirectional: bool = True
    lanes: int = 1
    risk_score: float = 0.0             # aggregated hazard risk [0-1]
    passable: bool = True

    @property
    def capacity_veh_per_hour(self) -> float:
        """Approximate hourly lane capacity by road type."""
        caps = {
            "motorway": 2200, "trunk": 2000, "primary": 1800,
            "secondary": 1500, "tertiary": 1200, "residential": 600,
            "service": 400, "living_street": 200,
            "footway": 0, "path": 0, "steps": 0
        }
        base = caps.get(self.highway_type, 600)
        return base * self.lanes

    @property
    def travel_time_s(self) -> float:
        """Free-flow travel time in seconds."""
        spd = self.max_speed_kmh
        # Guard against NaN or zero speed (pandas uses NaN for missing maxspeed tags)
        if spd != spd or spd <= 0:
            spd = 5.0
        speed_ms = max(spd, 5.0) / 3.6
        return self.length_m / speed_ms

    @property
    def quality_weight(self) -> float:
        """Road quality factor (lower = better road)."""
        weights = {
            "motorway": 0.5, "trunk": 0.6, "primary": 0.7,
            "secondary": 0.8, "tertiary": 0.9, "residential": 1.0,
            "service": 1.2, "living_street": 1.3, "unclassified": 1.1,
            "footway": 2.0, "path": 2.5, "steps": 5.0, "track": 1.8
        }
        return weights.get(self.highway_type, 1.0)


@dataclass
class Village:
    """Population area / village."""
    village_id: str
    name: str
    centroid_lat: float
    centroid_lon: float
    population: int
    area_m2: float = 0.0
    admin_level: int = 9
    geometry_wkt: Optional[str] = None
    risk_scores: Dict[str, float] = field(default_factory=dict)
    nearest_node_id: Optional[int] = None

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.centroid_lat, self.centroid_lon)


@dataclass
class Shelter:
    """Evacuation shelter."""
    shelter_id: str
    name: str
    centroid_lat: float
    centroid_lon: float
    capacity: int
    shelter_type: str = "assembly_point"
    area_m2: float = 0.0
    geometry_wkt: Optional[str] = None
    risk_scores: Dict[str, float] = field(default_factory=dict)
    nearest_node_id: Optional[int] = None
    current_occupancy: int = 0

    @property
    def coordinates(self) -> Tuple[float, float]:
        return (self.centroid_lat, self.centroid_lon)

    @property
    def remaining_capacity(self) -> int:
        return max(0, self.capacity - self.current_occupancy)


@dataclass
class RiskLayer:
    """InaRISK hazard risk layer for a point."""
    lat: float
    lon: float
    disaster_type: DisasterType
    risk_score: float       # 0.0 (low) to 1.0 (high)
    raw_value: Optional[float] = None
    source: str = "inarisk"


@dataclass
class EvacuationRoute:
    """A candidate evacuation route from a village to a shelter."""
    route_id: str
    village_id: str
    shelter_id: str
    node_path: List[int]
    total_distance_m: float
    total_time_s: float
    avg_risk_score: float
    max_risk_score: float
    worst_road_quality: float
    composite_score: float              # lower = better
    rank: int = 1
    assigned_population: int = 0

    @property
    def total_distance_km(self) -> float:
        return self.total_distance_m / 1000.0

    @property
    def total_time_min(self) -> float:
        return self.total_time_s / 60.0


@dataclass
class Assignment:
    """Population-to-shelter assignment result."""
    village_id: str
    shelter_id: str
    route_id: str
    assigned_population: int
    fraction: float
    route: Optional[EvacuationRoute] = None


@dataclass
class OptimizationResult:
    """Full optimization result for a scenario."""
    scenario_id: str
    disaster: DisasterInput
    assignments: List[Assignment]
    routes: List[EvacuationRoute]
    total_population: int
    total_evacuated: int
    total_unmet: int
    avg_route_risk: float
    avg_route_distance_km: float
    avg_route_time_min: float
    shelter_utilization: Dict[str, float]   # shelter_id -> fill ratio
    bottleneck_edges: List[int]             # edge ids with high flow
    runtime_s: float = 0.0
    mode: ExecutionMode = ExecutionMode.NAIVE

    @property
    def evacuation_ratio(self) -> float:
        if self.total_population == 0:
            return 0.0
        return self.total_evacuated / self.total_population


@dataclass
class SimulationOutput:
    """Output from GAMA-platform simulation."""
    scenario_id: str
    run_id: int
    total_saved: int
    total_delayed: int
    total_failed: int
    evacuation_completion_ratio: float
    avg_evacuation_time_min: float
    worst_evacuation_time_min: float
    bottleneck_road_ids: List[str]
    overloaded_shelter_ids: List[str]
    congestion_timeline: List[Dict[str, Any]]   # time -> edge_id -> flow
    raw_output_path: Optional[str] = None


@dataclass
class BenchmarkResult:
    """Computational performance result."""
    scenario_id: str
    mode: ExecutionMode
    phase: str                          # extraction, graph_build, routing, optimization, simulation
    n_workers: int
    n_nodes: int
    n_edges: int
    n_villages: int
    n_shelters: int
    wall_time_s: float
    cpu_time_s: float
    peak_memory_mb: float
    speedup: float = 1.0               # vs naive
    efficiency: float = 1.0            # speedup / n_workers
