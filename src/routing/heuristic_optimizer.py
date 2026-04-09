"""
Heuristic evacuation route optimizer.

Algorithm:
1. For each village, compute K shortest paths to each shelter using
   Dijkstra on the weighted graph (weight = distance × quality × (1 + risk)).
2. Score each route using composite metric.
3. Rank routes per village.

Supports:
- Naive (sequential) execution
- Parallel (multiprocessing) execution
"""

import math
import logging
import uuid
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import networkx as nx

from src.data.models import (
    Village, Shelter, EvacuationRoute, DisasterType,
    NetworkEdge, ExecutionMode
)

logger = logging.getLogger(__name__)


class HeuristicOptimizer:
    """
    Computes candidate evacuation routes from villages to shelters.

    Composite score (lower = better):
        score = w_dist     × norm_dist
              + w_risk     × avg_risk
              + w_quality  × worst_quality
              + w_time     × norm_time
              + w_disaster × norm_disaster_proximity  (higher = shelter closer to disaster = worse)

    Risk is NOT used as a hard filter — all road-reachable shelters are scored and ranked.
    This ensures villages are never left without routes due to a risk threshold.
    """

    def __init__(
        self,
        weight_distance: float = 0.3,
        weight_risk: float = 0.4,
        weight_road_quality: float = 0.2,
        weight_time: float = 0.1,
        weight_disaster_distance: float = 0.05,
        max_routes_per_village: int = 5,
        disaster_location: Optional[Tuple[float, float]] = None,
    ):
        self.w_dist = weight_distance
        self.w_risk = weight_risk
        self.w_quality = weight_road_quality
        self.w_time = weight_time
        self.w_disaster = weight_disaster_distance
        self.max_routes = max_routes_per_village
        self.disaster_location = disaster_location  # (lat, lon) of disaster center

    def compute_routes(
        self,
        G: nx.Graph,
        villages: List[Village],
        shelters: List[Shelter],
        mode: ExecutionMode = ExecutionMode.NAIVE,
        n_workers: int = 4,
    ) -> List[EvacuationRoute]:
        """
        Compute evacuation routes for all villages.
        Returns flat list of EvacuationRoute objects.
        """
        if mode == ExecutionMode.NAIVE:
            return self._compute_sequential(G, villages, shelters)
        elif mode == ExecutionMode.PARALLEL:
            return self._compute_parallel(G, villages, shelters, n_workers)
        else:
            # HPC mode: parallel is handled externally by Ray/Dask
            return self._compute_parallel(G, villages, shelters, n_workers)

    def _compute_sequential(
        self,
        G: nx.Graph,
        villages: List[Village],
        shelters: List[Shelter],
    ) -> List[EvacuationRoute]:
        """Sequential route computation (naive mode)."""
        all_routes = []
        shelter_nodes = {s.nearest_node_id: s for s in shelters
                         if s.nearest_node_id is not None}

        n = len(villages)
        log_every = max(1, n // 10)
        for i, v in enumerate(villages):
            if v.nearest_node_id is None or v.nearest_node_id not in G:
                logger.debug(f"Village {v.name} has no graph node, skipping")
                continue
            routes = self._routes_for_village(G, v, shelters, shelter_nodes)
            all_routes.extend(routes)
            if (i + 1) % log_every == 0 or (i + 1) == n:
                logger.info(f"  Routing: {i+1}/{n} villages done ({len(all_routes)} routes so far)")

        logger.info(f"Computed {len(all_routes)} candidate routes for {len(villages)} villages")
        return all_routes

    def _compute_parallel(
        self,
        G: nx.Graph,
        villages: List[Village],
        shelters: List[Shelter],
        n_workers: int,
    ) -> List[EvacuationRoute]:
        """
        Parallel route computation using multiprocessing.
        Note: G must be picklable (NetworkX graphs are).
        """
        shelter_nodes = {s.nearest_node_id: s for s in shelters
                         if s.nearest_node_id is not None}
        valid_villages = [v for v in villages
                          if v.nearest_node_id is not None and v.nearest_node_id in G]

        compute_fn = partial(
            _routes_for_village_standalone,
            G=G,
            shelters=shelters,
            shelter_nodes=shelter_nodes,
            w_dist=self.w_dist,
            w_risk=self.w_risk,
            w_quality=self.w_quality,
            w_time=self.w_time,
            w_disaster=self.w_disaster,
            max_routes=self.max_routes,
            disaster_location=self.disaster_location,
        )

        all_routes = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(compute_fn, v): v for v in valid_villages}
            for future in as_completed(futures):
                try:
                    routes = future.result()
                    all_routes.extend(routes)
                except Exception as e:
                    v = futures[future]
                    logger.warning(f"Route computation failed for {v.name}: {e}")

        logger.info(f"Parallel: computed {len(all_routes)} routes for {len(villages)} villages")
        return all_routes

    def _routes_for_village(
        self,
        G: nx.Graph,
        village: Village,
        shelters: List[Shelter],
        shelter_nodes: Dict[int, Shelter],
    ) -> List[EvacuationRoute]:
        """Compute top-K routes from one village to all shelters."""
        return _routes_for_village_standalone(
            village=village,
            G=G,
            shelters=shelters,
            shelter_nodes=shelter_nodes,
            w_dist=self.w_dist,
            w_risk=self.w_risk,
            w_quality=self.w_quality,
            w_time=self.w_time,
            w_disaster=self.w_disaster,
            max_routes=self.max_routes,
            disaster_location=self.disaster_location,
        )

    def rank_routes(
        self,
        routes: List[EvacuationRoute],
    ) -> Dict[str, List[EvacuationRoute]]:
        """
        Group routes by village_id and rank by composite_score.
        Returns dict: village_id -> sorted list of routes.
        """
        by_village: Dict[str, List[EvacuationRoute]] = {}
        for r in routes:
            by_village.setdefault(r.village_id, []).append(r)

        for vid, vr in by_village.items():
            vr.sort(key=lambda r: r.composite_score)
            for i, r in enumerate(vr):
                r.rank = i + 1

        return by_village


# ------------------------------------------------------------------ #
# Module-level (picklable) function for multiprocessing
# ------------------------------------------------------------------ #

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _score_route(
    total_dist: float,
    total_time: float,
    avg_risk: float,
    worst_quality: float,
    shelter_dist_from_disaster_km: float,
    max_disaster_dist_km: float,
    w_dist: float,
    w_risk: float,
    w_quality: float,
    w_time: float,
    w_disaster: float,
) -> float:
    """Compute composite score (lower = better evacuation route)."""
    norm_dist = total_dist / 50_000.0      # normalise by 50 km
    norm_time = total_time / 7200.0        # normalise by 2 hours
    # Disaster proximity: shelters closer to disaster get a HIGHER penalty.
    # Normalised so max_disaster_dist → 0 penalty; 0 km → 1.0 penalty.
    if max_disaster_dist_km > 0 and w_disaster > 0:
        norm_disaster_prox = max(0.0, 1.0 - shelter_dist_from_disaster_km / max_disaster_dist_km)
    else:
        norm_disaster_prox = 0.0
    return (
        w_dist    * norm_dist
        + w_risk  * avg_risk
        + w_quality * (worst_quality / 3.0)   # max quality_weight ~3
        + w_time  * norm_time
        + w_disaster * norm_disaster_prox
    )


def _routes_for_village_standalone(
    village: Village,
    G: nx.Graph,
    shelters: List[Shelter],
    shelter_nodes: Dict[int, Shelter],
    w_dist: float,
    w_risk: float,
    w_quality: float,
    w_time: float,
    w_disaster: float,
    max_routes: int,
    disaster_location: Optional[Tuple[float, float]] = None,
) -> List[EvacuationRoute]:
    """
    Standalone function (picklable for ProcessPoolExecutor).
    Compute up to max_routes from one village to all reachable shelters, ranked by
    composite score.  Risk is penalised through composite scoring — not filtered out.
    Villages on graph components disconnected from shelters return an empty list.
    """
    src_node = village.nearest_node_id
    if src_node not in G:
        return []

    # Pre-compute shortest paths from source to all reachable nodes
    try:
        lengths, paths = nx.single_source_dijkstra(
            G, src_node, weight="weight", cutoff=None
        )
    except nx.NetworkXError:
        return []

    # Pre-compute shelter distances from disaster center for penalty scoring
    if disaster_location is not None and w_disaster > 0:
        d_lat, d_lon = disaster_location
        shelter_disaster_dists = {
            s.shelter_id: _haversine_km(d_lat, d_lon, s.centroid_lat, s.centroid_lon)
            for s in shelters
        }
        max_disaster_dist_km = max(shelter_disaster_dists.values()) if shelter_disaster_dists else 1.0
    else:
        shelter_disaster_dists = {}
        max_disaster_dist_km = 1.0

    candidate_routes = []
    for shelter in shelters:
        tgt_node = shelter.nearest_node_id
        if tgt_node is None or tgt_node not in G:
            continue
        if tgt_node not in paths:
            continue

        path = paths[tgt_node]
        if len(path) < 2:
            continue

        total_dist = 0.0
        total_time = 0.0
        risk_scores = []
        quality_scores = []

        for u, v_node in zip(path[:-1], path[1:]):
            edge_data = G[u][v_node] if G.has_edge(u, v_node) else {}
            total_dist += float(edge_data.get("length_m", 0.0))
            total_time += float(edge_data.get("travel_time_s", 0.0))
            risk_scores.append(float(edge_data.get("risk_score", 0.0)))
            quality_scores.append(float(edge_data.get("quality_weight", 1.0)))

        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        max_risk_on_path = max(risk_scores) if risk_scores else 0.0
        worst_quality = max(quality_scores) if quality_scores else 1.0

        shelter_dist_km = shelter_disaster_dists.get(shelter.shelter_id, 0.0)
        composite = _score_route(
            total_dist, total_time, avg_risk, worst_quality,
            shelter_dist_km, max_disaster_dist_km,
            w_dist, w_risk, w_quality, w_time, w_disaster,
        )

        candidate_routes.append(EvacuationRoute(
            route_id=f"{village.village_id}_to_{shelter.shelter_id}",
            village_id=village.village_id,
            shelter_id=shelter.shelter_id,
            node_path=path,
            total_distance_m=total_dist,
            total_time_s=total_time,
            avg_risk_score=avg_risk,
            max_risk_score=max_risk_on_path,
            worst_road_quality=worst_quality,
            composite_score=composite,
        ))

    # Sort by composite score (lower = better); return up to max_routes
    # min_routes is a soft guarantee — if fewer shelters are reachable via road network,
    # we return all we found (can't create routes to unreachable components).
    candidate_routes.sort(key=lambda r: r.composite_score)
    return candidate_routes[:max_routes]
