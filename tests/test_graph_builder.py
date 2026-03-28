"""
Tests for graph builder module.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.models import NetworkNode, NetworkEdge, Village, Shelter, DisasterType
from src.graph.graph_builder import EvacuationGraphBuilder, _haversine_m


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def sample_nodes():
    return [
        NetworkNode(0, -7.5, 110.4),
        NetworkNode(1, -7.51, 110.4),
        NetworkNode(2, -7.52, 110.4),
        NetworkNode(3, -7.53, 110.4),
        NetworkNode(4, -7.54, 110.4),
    ]

@pytest.fixture
def sample_edges():
    return [
        NetworkEdge(0, 1, 1200.0, "primary", 60.0, True, 2),
        NetworkEdge(1, 2, 1100.0, "secondary", 50.0, True, 2),
        NetworkEdge(2, 3, 1000.0, "tertiary", 40.0, True, 1),
        NetworkEdge(3, 4, 900.0,  "residential", 30.0, True, 1),
    ]

@pytest.fixture
def sample_villages():
    return [
        Village("v1", "Village A", -7.505, 110.401, 500),
        Village("v2", "Village B", -7.525, 110.399, 300),
    ]

@pytest.fixture
def sample_shelters():
    return [
        Shelter("s1", "School A", -7.54, 110.40, 1000),
        Shelter("s2", "Hall B", -7.50, 110.41, 500),
    ]


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

def test_graph_build_basic(sample_nodes, sample_edges):
    builder = EvacuationGraphBuilder()
    G = builder.build(sample_nodes, sample_edges)
    assert G.number_of_nodes() == 5
    # Bidirectional edges: 4 pairs = 8 directed edges in undirected = 4
    assert G.number_of_edges() >= 4


def test_graph_has_weights(sample_nodes, sample_edges):
    builder = EvacuationGraphBuilder()
    G = builder.build(sample_nodes, sample_edges)
    for u, v, data in G.edges(data=True):
        assert "weight" in data
        assert data["weight"] > 0


def test_graph_edge_attributes(sample_nodes, sample_edges):
    builder = EvacuationGraphBuilder()
    G = builder.build(sample_nodes, sample_edges)
    for u, v, data in G.edges(data=True):
        assert "length_m" in data
        assert "risk_score" in data
        assert "travel_time_s" in data
        assert data["travel_time_s"] > 0


def test_find_nearest_node(sample_nodes, sample_edges):
    builder = EvacuationGraphBuilder()
    builder.build(sample_nodes, sample_edges)
    # Point close to node 0 (-7.5, 110.4)
    nearest = builder.find_nearest_node(-7.502, 110.400)
    assert nearest == 0


def test_attach_pois(sample_nodes, sample_edges, sample_villages, sample_shelters):
    builder = EvacuationGraphBuilder()
    builder.build(sample_nodes, sample_edges)
    builder.attach_pois_to_graph(sample_villages, sample_shelters)
    for v in sample_villages:
        assert v.nearest_node_id is not None
    for s in sample_shelters:
        assert s.nearest_node_id is not None


def test_impassable_edge_pruning(sample_nodes, sample_edges):
    """Edges with risk > threshold should be removed."""
    sample_edges[0].risk_score = 0.95  # above threshold
    builder = EvacuationGraphBuilder()
    G = builder.build(sample_nodes, sample_edges,
                      impassable_risk_threshold=0.9)
    # Edge 0-1 should be removed
    # Other edges remain
    assert not G.has_edge(0, 1)
    assert G.has_edge(1, 2)


def test_haversine_distance():
    # Known distance: Jakarta to Bandung ~120km
    d = _haversine_m(-6.2, 106.8, -6.9, 107.6)
    assert 100_000 < d < 140_000


def test_high_risk_penalty():
    """Higher risk edges should have higher weight."""
    nodes = [NetworkNode(0, 0.0, 0.0), NetworkNode(1, 0.0, 0.001)]
    low_risk = NetworkEdge(0, 1, 100.0, "primary", 60.0, True, 1)
    low_risk.risk_score = 0.0
    high_risk = NetworkEdge(0, 1, 100.0, "primary", 60.0, True, 1)
    high_risk.risk_score = 0.9

    b1 = EvacuationGraphBuilder()
    G1 = b1.build(nodes, [low_risk])
    w1 = G1[0][1]["weight"]

    b2 = EvacuationGraphBuilder()
    G2 = b2.build(nodes, [high_risk])
    w2 = G2[0][1]["weight"]

    assert w2 > w1, "High-risk edge should have higher weight"
