"""
Tests for InaRISK client (unit tests with mocking).
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.models import DisasterType
from src.data.inarisk_client import InaRISKClient, _latlon_to_mercator


def test_latlon_to_mercator_equator():
    x, y = _latlon_to_mercator(0.0, 0.0)
    assert abs(x) < 1e-6
    assert abs(y) < 1e-6


def test_latlon_to_mercator_jakarta():
    # Jakarta: approx (-6.2, 106.8)
    x, y = _latlon_to_mercator(-6.2, 106.8)
    assert x > 0       # East of meridian
    assert y < 0       # South of equator


def test_empty_points():
    client = InaRISKClient()
    result = client.get_risk_scores([], DisasterType.EARTHQUAKE)
    assert result == []


def test_mock_api_call():
    """Mock InaRISK API response."""
    client = InaRISKClient()

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "features": [{"attributes": {"INDEKS_BAHAYA": 2.0}}]
    }
    mock_response.raise_for_status = MagicMock()

    with patch.object(client.session, "get", return_value=mock_response):
        scores = client.get_risk_scores([(-7.5, 110.4)], DisasterType.VOLCANO)

    assert len(scores) == 1
    # InaRISK value 2.0 → normalized (2-1)/2 = 0.5
    assert abs(scores[0] - 0.5) < 0.01


def test_api_failure_returns_zero():
    """Failed API calls should return 0.0, not raise."""
    client = InaRISKClient()

    with patch.object(client.session, "get", side_effect=Exception("Connection error")):
        scores = client.get_risk_scores([(-7.5, 110.4)], DisasterType.FLOOD)

    assert scores == [0.0]


def test_no_features_returns_zero():
    """Empty features response should return 0.0."""
    client = InaRISKClient()
    mock_response = MagicMock()
    mock_response.json.return_value = {"features": []}
    mock_response.raise_for_status = MagicMock()

    with patch.object(client.session, "get", return_value=mock_response):
        scores = client.get_risk_scores([(-7.5, 110.4)], DisasterType.LANDSLIDE)

    assert scores == [0.0]


def test_batch_processing():
    """Multiple points should be processed in batches."""
    client = InaRISKClient(batch_size=3)
    points = [(-7.5 - i * 0.01, 110.4) for i in range(10)]

    call_count = 0
    def mock_get(url, params=None, timeout=None, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"features": [{"attributes": {"INDEKS_BAHAYA": 1.0}}]}
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    with patch.object(client.session, "get", side_effect=mock_get):
        with patch("time.sleep"):  # Skip rate limiting
            scores = client.get_risk_scores(points, DisasterType.EARTHQUAKE)

    assert len(scores) == 10
    # batch_size=3, 10 points → ceil(10/3) = 4 batches → 10 individual calls
    assert call_count == 10
