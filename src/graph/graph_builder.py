"""
Evacuation graph builder.
Constructs a weighted NetworkX graph from road network nodes/edges
with hazard risk scores, road quality, and capacity annotations.

Edge weight formula (composite):
  w(e) = distance(e) × quality_weight(e) × (1 + risk_score(e))

BPR congestion:
  t_congested = t_free × (1 + α × (flow/capacity)^β)
"""

import math
import logging
from typing import List, Dict, Optional, Tuple, Set

import networkx as nx

from src.data.models import NetworkNode, NetworkEdge, Village, Shelter, DisasterType

logger = logging.getLogger(__name__)

class EvacuationGraphBuilder:
    """
    Builds and manages the weighted evacuation graph.
    """

    def __init__(self):
        self.G: Optional[nx.Graph] = None
        self._node_coords: Dict[int, Tuple[float, float]] = {}  # node_id -> (lat, lon)
        self._risk_weight: float = 0.4  # stored during build() for reuse in edge enrichment

    def build(
        self,
        nodes: List[NetworkNode],
        edges: List[NetworkEdge],
        disaster_type: Optional[DisasterType] = None,
        risk_weight: float = 0.4,
        prune_impassable: bool = False,
        impassable_risk_threshold: float = 0.9,
    ) -> nx.Graph:
        """
        Build weighted evacuation graph.

        Args:
            nodes: Road network nodes with optional risk scores.
            edges: Road network edges.
            disaster_type: Primary disaster type (used to select risk score field).
            risk_weight: How much risk contributes to composite edge weight.
            prune_impassable: Remove edges with risk > threshold.  Default False —
                              pruning disconnects the graph and leaves villages unreachable.
                              Risk is penalised through composite weight instead.
            impassable_risk_threshold: Risk value above which edges are removed (only
                                       used when prune_impassable=True).
        Returns:
            Weighted NetworkX Graph.
        """
        self._risk_weight = risk_weight
        G = nx.Graph()

        # Add nodes
        for n in nodes:
            G.add_node(n.node_id, lat=n.lat, lon=n.lon, risk_scores=n.risk_scores)
            self._node_coords[n.node_id] = (n.lat, n.lon)

        logger.info(f"Graph: added {G.number_of_nodes()} nodes")

        # Determine risk field
        risk_field = disaster_type.value if disaster_type else None

        # Add edges
        skipped = 0
        for e in edges:
            if e.source_id not in self._node_coords or e.target_id not in self._node_coords:
                skipped += 1
                continue
            if not e.passable:
                skipped += 1
                continue

            risk = e.risk_score
            if risk_field and risk_field in (getattr(e, "risk_scores", {}) or {}):
                risk = e.risk_scores[risk_field]

            # Remove impassable edges
            if prune_impassable and risk > impassable_risk_threshold:
                skipped += 1
                continue

            # Composite weight: lower = better route
            composite_weight = (
                e.length_m
                * e.quality_weight
                * (1.0 + risk_weight * risk)
            )

            edge_attrs = {
                "length_m": e.length_m,
                "highway_type": e.highway_type,
                "max_speed_kmh": e.max_speed_kmh,
                "lanes": e.lanes,
                "travel_time_s": e.travel_time_s,
                "capacity_veh_h": e.capacity_veh_per_hour,
                "risk_score": risk,
                "quality_weight": e.quality_weight,
                "weight": composite_weight,      # used by nx.shortest_path
                "flow": 0.0,                     # current flow (updated during assignment)
            }

            G.add_edge(e.source_id, e.target_id, **edge_attrs)
            if e.bidirectional:
                G.add_edge(e.target_id, e.source_id, **edge_attrs)

        self.G = G
        logger.info(
            f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
            f"(skipped {skipped})"
        )
        return G

    def find_nearest_node(self, lat: float, lon: float) -> Optional[int]:
        """Find nearest graph node to a coordinate using haversine distance."""
        if not self._node_coords:
            return None
        min_dist = float("inf")
        nearest = None
        for node_id, (nlat, nlon) in self._node_coords.items():
            d = _haversine_m(lat, lon, nlat, nlon)
            if d < min_dist:
                min_dist = d
                nearest = node_id
        return nearest

    def attach_pois_to_graph(
        self,
        villages: List[Village],
        shelters: List[Shelter],
    ) -> None:
        """
        Find nearest graph node for each village and shelter.
        Modifies objects in-place.
        """
        for v in villages:
            v.nearest_node_id = self.find_nearest_node(v.centroid_lat, v.centroid_lon)

        for s in shelters:
            s.nearest_node_id = self.find_nearest_node(s.centroid_lat, s.centroid_lon)

        no_node_v = sum(1 for v in villages if v.nearest_node_id is None)
        no_node_s = sum(1 for s in shelters if s.nearest_node_id is None)
        if no_node_v:
            logger.warning(f"{no_node_v} villages have no nearest node")
        if no_node_s:
            logger.warning(f"{no_node_s} shelters have no nearest node")

    def apply_inarisk_to_edges(
        self,
        inarisk,                        # InaRISKClient — avoid circular import
        hazard_layers: dict,            # {DisasterType: weight}
        aggregation: str = "weighted_sum",
        cache_path=None,
        use_cache: bool = True,
    ) -> int:
        """
        Query InaRISK at road segment midpoints and update edge risk_score + weight.

        Delegates to InaRISKClient.enrich_graph_edges(), passing this builder's
        graph and node coordinate map.  Must be called after build().

        Args:
            inarisk:       Configured InaRISKClient instance.
            hazard_layers: {DisasterType: weight} from _resolve_hazard_layers().
            aggregation:   "weighted_sum" or "max".
            cache_path:    Path to JSON cache file (shared with export_shp if desired).
        Returns:
            Number of edges updated (risk raised above 0.0).
        """
        if self.G is None:
            return 0
        return inarisk.enrich_graph_edges(
            G=self.G,
            node_coords=self._node_coords,
            hazard_layers=hazard_layers,
            aggregation=aggregation,
            risk_weight=self._risk_weight,
            cache_path=cache_path,
            use_cache=use_cache,
        )

    def propagate_poi_risk_to_graph(
        self,
        villages: List[Village],
        shelters: List[Shelter],
        risk_weight: float = 0.4,
    ) -> None:
        """
        Propagate InaRISK composite risk scores from villages and shelters
        onto their adjacent graph edges, then recompute composite edge weights.

        Must be called AFTER attach_pois_to_graph() and InaRISK enrichment.
        Without this step, all edge risk_scores remain 0.0 because OSM road
        data carries no hazard information — routing would be risk-blind.

        Args:
            villages: Village objects with risk_scores["composite"] populated.
            shelters: Shelter objects with risk_scores["composite"] populated.
            risk_weight: Same risk_weight used in build() — scales risk in
                         composite_weight = length × quality × (1 + rw × risk).
        """
        if self.G is None:
            return

        updated = 0
        for poi in (*villages, *shelters):
            node = getattr(poi, "nearest_node_id", None)
            if node is None or node not in self.G:
                continue
            poi_risk = poi.risk_scores.get("composite", 0.0)
            if poi_risk <= 0.0:
                continue
            for nbr in self.G.neighbors(node):
                data = self.G[node][nbr]
                old = data.get("risk_score", 0.0)
                new_risk = max(old, poi_risk)
                if new_risk > old:
                    data["risk_score"] = new_risk
                    data["weight"] = (
                        data["length_m"]
                        * data["quality_weight"]
                        * (1.0 + risk_weight * new_risk)
                    )
                    updated += 1

        logger.info(
            f"propagate_poi_risk_to_graph: updated risk on {updated} edges "
            f"from {len(villages)} villages + {len(shelters)} shelters"
        )


# ------------------------------------------------------------------ #
# Utility functions
# ------------------------------------------------------------------ #

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in meters between two lat/lon points."""
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))
