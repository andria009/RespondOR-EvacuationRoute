"""
Population-to-shelter assignment optimizer.
Solves a multi-commodity flow assignment problem to maximize evacuated population.

Problem:
  maximize  sum_ij xij * pop_i * feasibility_ij
  subject to:
    sum_j xij <= 1               (each village fully assigned at most once)
    sum_i xij * pop_i <= cap_j  (shelter capacity)
    xij >= 0

Approach:
  - Greedy heuristic: sort village-shelter pairs by composite_score asc,
    assign greedily while respecting capacity constraints.
  - Optional LP relaxation via scipy.optimize.linprog for optimal solution.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.data.models import (
    Village, Shelter, EvacuationRoute, Assignment, OptimizationResult,
    DisasterInput, ExecutionMode
)

logger = logging.getLogger(__name__)


class PopulationAssigner:
    """
    Assigns village populations to shelters via evacuation routes.
    Supports greedy and LP-based assignment.
    """

    def __init__(self, method: str = "greedy"):
        """
        Args:
            method: "greedy" (fast) or "lp" (optimal, requires scipy).
        """
        self.method = method

    def assign(
        self,
        villages: List[Village],
        shelters: List[Shelter],
        routes_by_village: Dict[str, List[EvacuationRoute]],
        scenario_id: str = "scenario",
        disaster: Optional[DisasterInput] = None,
        mode: ExecutionMode = ExecutionMode.NAIVE,
        runtime_s: float = 0.0,
    ) -> OptimizationResult:
        """
        Perform population-to-shelter assignment.
        Returns full OptimizationResult.
        """
        if self.method == "lp":
            try:
                return self._assign_lp(
                    villages, shelters, routes_by_village,
                    scenario_id, disaster, mode, runtime_s
                )
            except Exception as e:
                logger.warning(f"LP assignment failed ({e}), falling back to greedy")

        return self._assign_greedy(
            villages, shelters, routes_by_village,
            scenario_id, disaster, mode, runtime_s
        )

    # ------------------------------------------------------------------ #
    # Greedy assignment
    # ------------------------------------------------------------------ #

    def _assign_greedy(
        self,
        villages: List[Village],
        shelters: List[Shelter],
        routes_by_village: Dict[str, List[EvacuationRoute]],
        scenario_id: str,
        disaster: Optional[DisasterInput],
        mode: ExecutionMode,
        runtime_s: float,
    ) -> OptimizationResult:
        """
        Greedy: sort all (village, shelter, route) triplets by composite_score asc.
        Assign greedily while respecting capacity.
        """
        # Build flat list of (score, village, shelter, route)
        candidates = []
        village_map = {v.village_id: v for v in villages}
        shelter_map = {s.shelter_id: s for s in shelters}

        for vid, routes in routes_by_village.items():
            v = village_map.get(vid)
            if v is None:
                continue
            for route in routes:
                s = shelter_map.get(route.shelter_id)
                if s is None:
                    continue
                candidates.append((route.composite_score, v, s, route))

        candidates.sort(key=lambda x: x[0])

        # Capacity tracking
        shelter_remaining: Dict[str, int] = {s.shelter_id: s.capacity for s in shelters}
        village_assigned: Dict[str, int] = {v.village_id: 0 for v in villages}

        assignments: List[Assignment] = []
        assigned_routes: List[EvacuationRoute] = []

        for _, v, s, route in candidates:
            remaining_pop = v.population - village_assigned.get(v.village_id, 0)
            if remaining_pop <= 0:
                continue  # Already fully assigned

            shelter_cap = shelter_remaining.get(s.shelter_id, 0)
            if shelter_cap <= 0:
                continue  # Shelter full

            assign_pop = min(remaining_pop, shelter_cap)
            fraction = assign_pop / v.population if v.population > 0 else 0.0

            village_assigned[v.village_id] = village_assigned.get(v.village_id, 0) + assign_pop
            shelter_remaining[s.shelter_id] -= assign_pop

            route.assigned_population = assign_pop
            assignments.append(Assignment(
                village_id=v.village_id,
                shelter_id=s.shelter_id,
                route_id=route.route_id,
                assigned_population=assign_pop,
                fraction=fraction,
                route=route,
            ))
            assigned_routes.append(route)

        return self._build_result(
            scenario_id, disaster, villages, shelters,
            assignments, assigned_routes, shelter_map,
            mode, runtime_s
        )

    # ------------------------------------------------------------------ #
    # LP assignment
    # ------------------------------------------------------------------ #

    def _assign_lp(
        self,
        villages: List[Village],
        shelters: List[Shelter],
        routes_by_village: Dict[str, List[EvacuationRoute]],
        scenario_id: str,
        disaster: Optional[DisasterInput],
        mode: ExecutionMode,
        runtime_s: float,
    ) -> OptimizationResult:
        """
        LP-based optimal assignment via scipy.optimize.linprog.
        Maximizes total evacuated population (minimize negative).
        """
        from scipy.optimize import linprog
        import numpy as np

        village_map = {v.village_id: v for v in villages}
        shelter_map = {s.shelter_id: s for s in shelters}

        # Build decision variable index: (village_id, shelter_id) -> idx
        var_list = []
        for vid, routes in routes_by_village.items():
            v = village_map.get(vid)
            if v is None:
                continue
            for route in routes:
                s = shelter_map.get(route.shelter_id)
                if s is not None:
                    var_list.append((vid, route.shelter_id, v.population, route))

        n_vars = len(var_list)
        if n_vars == 0:
            return self._assign_greedy(villages, shelters, routes_by_village,
                                        scenario_id, disaster, mode, runtime_s)

        village_ids = [v.village_id for v in villages]
        shelter_ids = [s.shelter_id for s in shelters]

        # Objective: minimize -sum(xij * pop_i)  (maximize evacuated)
        c = np.array([-v_pop for (_, _, v_pop, _) in var_list])

        # Constraint 1: sum_j xij <= 1  (village assignment fraction <= 1)
        A_village = np.zeros((len(village_ids), n_vars))
        for col_idx, (vid, sid, pop, _) in enumerate(var_list):
            row_idx = village_ids.index(vid) if vid in village_ids else -1
            if row_idx >= 0:
                A_village[row_idx, col_idx] = 1.0

        b_village = np.ones(len(village_ids))

        # Constraint 2: sum_i xij * pop_i <= cap_j
        A_shelter = np.zeros((len(shelter_ids), n_vars))
        b_shelter = np.array([shelter_map[sid].capacity for sid in shelter_ids], dtype=float)

        for col_idx, (vid, sid, pop, _) in enumerate(var_list):
            row_idx = shelter_ids.index(sid) if sid in shelter_ids else -1
            if row_idx >= 0:
                A_shelter[row_idx, col_idx] = pop

        A_ub = np.vstack([A_village, A_shelter])
        b_ub = np.hstack([b_village, b_shelter])
        bounds = [(0.0, 1.0)] * n_vars

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        if not result.success:
            logger.warning(f"LP did not converge: {result.message}, falling back to greedy")
            return self._assign_greedy(villages, shelters, routes_by_village,
                                        scenario_id, disaster, mode, runtime_s)

        # Build assignments from LP solution
        assignments = []
        assigned_routes = []
        for col_idx, (vid, sid, pop, route) in enumerate(var_list):
            frac = float(result.x[col_idx])
            if frac < 0.001:
                continue
            assign_pop = int(round(frac * pop))
            if assign_pop <= 0:
                continue
            route.assigned_population = assign_pop
            assignments.append(Assignment(
                village_id=vid,
                shelter_id=sid,
                route_id=route.route_id,
                assigned_population=assign_pop,
                fraction=frac,
                route=route,
            ))
            assigned_routes.append(route)

        return self._build_result(
            scenario_id, disaster, villages, shelters,
            assignments, assigned_routes, shelter_map,
            mode, runtime_s
        )

    # ------------------------------------------------------------------ #
    # Result builder
    # ------------------------------------------------------------------ #

    def _build_result(
        self,
        scenario_id: str,
        disaster: Optional[DisasterInput],
        villages: List[Village],
        shelters: List[Shelter],
        assignments: List[Assignment],
        routes: List[EvacuationRoute],
        shelter_map: Dict[str, Shelter],
        mode: ExecutionMode,
        runtime_s: float,
    ) -> OptimizationResult:
        total_population = sum(v.population for v in villages)
        total_evacuated = sum(a.assigned_population for a in assignments)
        total_unmet = total_population - total_evacuated

        # Shelter utilization
        shelter_assigned: Dict[str, int] = {}
        for a in assignments:
            shelter_assigned[a.shelter_id] = shelter_assigned.get(a.shelter_id, 0) + a.assigned_population

        shelter_util: Dict[str, float] = {}
        for sid, s in shelter_map.items():
            cap = s.capacity if s.capacity > 0 else 1
            shelter_util[sid] = shelter_assigned.get(sid, 0) / cap

        # Route metrics — use only assigned routes (non-zero population)
        used_routes = [r for r in routes if r.assigned_population > 0]
        if used_routes:
            avg_risk = sum(r.avg_risk_score for r in used_routes) / len(used_routes)
            avg_dist = sum(r.total_distance_km for r in used_routes) / len(used_routes)
            valid_times = [r.total_time_min for r in used_routes
                           if r.total_time_min == r.total_time_min]  # filter NaN
            avg_time = sum(valid_times) / len(valid_times) if valid_times else 0.0
        else:
            avg_risk = avg_dist = avg_time = 0.0

        logger.info(
            f"Assignment: evacuated={total_evacuated}/{total_population} "
            f"({100*total_evacuated/max(1,total_population):.1f}%), "
            f"unmet={total_unmet}"
        )

        return OptimizationResult(
            scenario_id=scenario_id,
            disaster=disaster,
            assignments=assignments,
            routes=routes,
            total_population=total_population,
            total_evacuated=total_evacuated,
            total_unmet=total_unmet,
            avg_route_risk=avg_risk,
            avg_route_distance_km=avg_dist,
            avg_route_time_min=avg_time,
            shelter_utilization=shelter_util,
            bottleneck_edges=[],        # filled by congestion analysis
            runtime_s=runtime_s,
            mode=mode,
        )
