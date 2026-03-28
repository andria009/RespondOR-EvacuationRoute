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

# BPR parameters
BPR_ALPHA = 0.15
BPR_BETA = 4.0


class EvacuationGraphBuilder:
    """
    Builds and manages the weighted evacuation graph.
    """

    def __init__(self):
        self.G: Optional[nx.Graph] = None
        self._node_coords: Dict[int, Tuple[float, float]] = {}  # node_id -> (lat, lon)

    def build(
        self,
        nodes: List[NetworkNode],
        edges: List[NetworkEdge],
        disaster_type: Optional[DisasterType] = None,
        risk_weight: float = 0.4,
        prune_impassable: bool = True,
        impassable_risk_threshold: float = 0.9,
    ) -> nx.Graph:
        """
        Build weighted evacuation graph.

        Args:
            nodes: Road network nodes with optional risk scores.
            edges: Road network edges.
            disaster_type: Primary disaster type (used to select risk score field).
            risk_weight: How much risk contributes to composite edge weight.
            prune_impassable: Remove edges with risk > threshold.
            impassable_risk_threshold: Risk value above which edges are removed.
        Returns:
            Weighted NetworkX Graph.
        """
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

    def score_edges_with_risk(
        self,
        edges: List[NetworkEdge],
        risk_scores: Dict[int, Dict[str, float]],  # edge_index -> {disaster_type: score}
        disaster_type: Optional[DisasterType] = None,
    ) -> None:
        """
        Assign risk scores to edges.
        risk_scores: maps edge index to dict of {disaster_type: score}.
        Updates NetworkEdge.risk_score in place.
        """
        for i, edge in enumerate(edges):
            if i in risk_scores:
                scores_dict = risk_scores[i]
                if disaster_type and disaster_type.value in scores_dict:
                    edge.risk_score = scores_dict[disaster_type.value]
                elif scores_dict:
                    edge.risk_score = float(sum(scores_dict.values()) / len(scores_dict))

    def apply_flow_congestion(self, G: nx.Graph) -> None:
        """
        Update edge travel times using BPR congestion function.
        Applied after flow assignment.
        """
        for u, v, data in G.edges(data=True):
            flow = data.get("flow", 0.0)
            cap = data.get("capacity_veh_h", 1.0)
            if cap <= 0:
                cap = 1.0
            t0 = data.get("travel_time_s", 1.0)
            ratio = flow / cap
            t_congested = t0 * (1.0 + BPR_ALPHA * (ratio ** BPR_BETA))
            G[u][v]["travel_time_congested_s"] = t_congested

    def get_subgraph_for_pois(
        self,
        villages: List[Village],
        shelters: List[Shelter],
        max_hops: int = 20,
    ) -> nx.Graph:
        """
        Extract subgraph connecting village and shelter nodes.
        Uses ego_graph expansion to include connecting nodes.
        """
        if self.G is None:
            raise ValueError("Graph not built yet. Call build() first.")

        poi_nodes: Set[int] = set()
        for v in villages:
            if v.nearest_node_id is not None:
                poi_nodes.add(v.nearest_node_id)
        for s in shelters:
            if s.nearest_node_id is not None:
                poi_nodes.add(s.nearest_node_id)

        # Include all nodes on shortest paths between POIs
        relevant = set(poi_nodes)
        for n in poi_nodes:
            if n in self.G:
                ego = nx.ego_graph(self.G, n, radius=max_hops, distance="weight")
                # Only add nodes also reachable to another POI
                relevant.update(ego.nodes)

        sub = self.G.subgraph(relevant).copy()
        logger.info(f"Subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")
        return sub

    def export_to_geojson(self, output_path: str) -> None:
        """Export graph edges as GeoJSON LineString features."""
        import json
        if self.G is None:
            raise ValueError("Graph not built")

        features = []
        for u, v, data in self.G.edges(data=True):
            if u not in self._node_coords or v not in self._node_coords:
                continue
            u_lat, u_lon = self._node_coords[u]
            v_lat, v_lon = self._node_coords[v]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[u_lon, u_lat], [v_lon, v_lat]]
                },
                "properties": {
                    "highway": data.get("highway_type", ""),
                    "length_m": data.get("length_m", 0),
                    "risk_score": data.get("risk_score", 0),
                    "weight": data.get("weight", 0),
                }
            })

        with open(output_path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)
        logger.info(f"Exported graph to {output_path}")


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
