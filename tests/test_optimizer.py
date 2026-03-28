"""
Tests for heuristic optimizer and population assigner.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.models import (
    NetworkNode, NetworkEdge, Village, Shelter, DisasterType, ExecutionMode
)
from src.graph.graph_builder import EvacuationGraphBuilder
from src.routing.heuristic_optimizer import HeuristicOptimizer
from src.routing.assignment import PopulationAssigner


# ------------------------------------------------------------------ #
# Fixtures: simple linear network
# v1 --- n0 --- n1 --- n2 --- n3 --- n4 --- s1
# ------------------------------------------------------------------ #

@pytest.fixture
def linear_network():
    nodes = [NetworkNode(i, -7.5 - i * 0.01, 110.4) for i in range(6)]
    edges = [
        NetworkEdge(i, i+1, 1000.0, "primary", 60.0, True, 2)
        for i in range(5)
    ]
    return nodes, edges


@pytest.fixture
def one_village_one_shelter():
    villages = [Village("v1", "TestVillage", -7.505, 110.401, 200)]
    shelters = [Shelter("s1", "TestShelter", -7.545, 110.399, 500)]
    return villages, shelters


@pytest.fixture
def built_graph(linear_network, one_village_one_shelter):
    nodes, edges = linear_network
    villages, shelters = one_village_one_shelter
    builder = EvacuationGraphBuilder()
    G = builder.build(nodes, edges)
    builder.attach_pois_to_graph(villages, shelters)
    return G, villages, shelters


# ------------------------------------------------------------------ #
# Optimizer tests
# ------------------------------------------------------------------ #

def test_optimizer_finds_routes(built_graph):
    G, villages, shelters = built_graph
    optimizer = HeuristicOptimizer()
    routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
    assert len(routes) > 0


def test_optimizer_route_attributes(built_graph):
    G, villages, shelters = built_graph
    optimizer = HeuristicOptimizer()
    routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
    for r in routes:
        assert r.village_id is not None
        assert r.shelter_id is not None
        assert r.total_distance_m > 0
        assert r.total_time_s > 0
        assert 0.0 <= r.avg_risk_score <= 1.0
        assert r.composite_score >= 0


def test_optimizer_rank_routes(built_graph):
    G, villages, shelters = built_graph
    optimizer = HeuristicOptimizer(max_routes_per_village=2)
    routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
    by_village = optimizer.rank_routes(routes)
    for vid, vr in by_village.items():
        assert len(vr) <= 2
        for i in range(len(vr) - 1):
            assert vr[i].composite_score <= vr[i+1].composite_score


def test_optimizer_respects_risk_threshold(linear_network):
    """Routes with max_risk above threshold should be excluded."""
    nodes, edges = linear_network
    # Set all edges to high risk
    for e in edges:
        e.risk_score = 0.95
    villages = [Village("v1", "V1", -7.505, 110.401, 100)]
    shelters = [Shelter("s1", "S1", -7.545, 110.399, 500)]
    builder = EvacuationGraphBuilder()
    G = builder.build(nodes, edges, prune_impassable=True, impassable_risk_threshold=0.9)
    builder.attach_pois_to_graph(villages, shelters)

    optimizer = HeuristicOptimizer(max_risk_threshold=0.8)
    routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
    # All routes should be excluded (high risk)
    assert len(routes) == 0


# ------------------------------------------------------------------ #
# Assignment tests
# ------------------------------------------------------------------ #

def test_greedy_assignment_basic(built_graph):
    G, villages, shelters = built_graph
    optimizer = HeuristicOptimizer()
    routes = optimizer.compute_routes(G, villages, shelters, mode=ExecutionMode.NAIVE)
    by_village = optimizer.rank_routes(routes)

    assigner = PopulationAssigner(method="greedy")
    result = assigner.assign(villages, shelters, by_village, scenario_id="test")

    assert result.total_population == 200
    assert result.total_evacuated <= 200
    assert result.total_evacuated >= 0
    assert result.evacuation_ratio >= 0.0


def test_shelter_capacity_respected(linear_network):
    """Shelter with tiny capacity should not be overfilled."""
    nodes, edges = linear_network
    villages = [
        Village("v1", "V1", -7.505, 110.401, 1000),
        Village("v2", "V2", -7.515, 110.401, 500),
    ]
    shelters = [
        Shelter("s1", "S1", -7.545, 110.399, 300),   # small capacity
    ]
    builder = EvacuationGraphBuilder()
    G = builder.build(nodes, edges)
    builder.attach_pois_to_graph(villages, shelters)

    optimizer = HeuristicOptimizer()
    routes = optimizer.compute_routes(G, villages, shelters)
    by_village = optimizer.rank_routes(routes)

    assigner = PopulationAssigner(method="greedy")
    result = assigner.assign(villages, shelters, by_village, scenario_id="test")

    # Total assigned to s1 must not exceed 300
    total_s1 = sum(
        a.assigned_population for a in result.assignments
        if a.shelter_id == "s1"
    )
    assert total_s1 <= 300


def test_assignment_unmet_demand():
    """If shelter capacity < population, there should be unmet demand."""
    nodes = [NetworkNode(i, -7.5 - i * 0.01, 110.4) for i in range(3)]
    edges = [NetworkEdge(0, 1, 1000.0, "primary", 60.0, True),
             NetworkEdge(1, 2, 1000.0, "primary", 60.0, True)]
    villages = [Village("v1", "V1", -7.505, 110.401, 5000)]  # 5000 people
    shelters = [Shelter("s1", "S1", -7.52, 110.399, 100)]    # only 100 capacity

    builder = EvacuationGraphBuilder()
    G = builder.build(nodes, edges)
    builder.attach_pois_to_graph(villages, shelters)

    optimizer = HeuristicOptimizer()
    routes = optimizer.compute_routes(G, villages, shelters)
    by_village = optimizer.rank_routes(routes)

    assigner = PopulationAssigner(method="greedy")
    result = assigner.assign(villages, shelters, by_village)

    assert result.total_unmet > 0
    assert result.total_evacuated == 100
    assert result.evacuation_ratio < 1.0
