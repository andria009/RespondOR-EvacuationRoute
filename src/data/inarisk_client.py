"""
InaRISK API client for disaster risk index extraction.
API: https://gis.bnpb.go.id/server/rest/services/inarisk
Supports all 4 disaster types: earthquake, flood, volcano, landslide.
"""

import math
import time
import logging
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.models import DisasterType, RiskLayer

logger = logging.getLogger(__name__)

# InaRISK service endpoints and field names per disaster type
INARISK_CONFIG = {
    DisasterType.EARTHQUAKE: {
        "service": "INDEKS_BAHAYA_GEMPABUMI",
        "layer_id": 0,
        "field": "INDEKS_BAHAYA",
    },
    DisasterType.FLOOD: {
        "service": "INDEKS_BAHAYA_BANJIR",
        "layer_id": 0,
        "field": "INDEKS_BAHAYA",
    },
    DisasterType.VOLCANO: {
        "service": "INDEKS_BAHAYA_GUNUNGAPI",
        "layer_id": 0,
        "field": "INDEKS_BAHAYA",
    },
    DisasterType.LANDSLIDE: {
        "service": "INDEKS_BAHAYA_TANAHLONGSOR",
        "layer_id": 0,
        "field": "INDEKS_BAHAYA",
    },
}

BASE_URL = "https://gis.bnpb.go.id/server/rest/services/inarisk"
IDENTIFY_URL = "{base}/{service}/MapServer/{layer_id}/query"


def _latlon_to_mercator(lat: float, lon: float) -> Tuple[float, float]:
    """Convert WGS84 lat/lon to Web Mercator (EPSG:3857)."""
    x = lon * 20037508.34 / 180.0
    y = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
    y = y * 20037508.34 / 180.0
    return x, y


class InaRISKClient:
    """
    Client for querying disaster risk indices from InaRISK BNPB.
    Uses ArcGIS MapServer REST identify endpoint.
    """

    def __init__(
        self,
        batch_size: int = 20,
        rate_limit_s: float = 1.0,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        self.batch_size = batch_size
        self.rate_limit_s = rate_limit_s

        # Session with retry logic
        self.session = requests.Session()
        retry = Retry(total=max_retries, backoff_factor=1.0,
                      status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.timeout = timeout

    def get_risk_scores(
        self,
        points: List[Tuple[float, float]],    # list of (lat, lon)
        disaster_type: DisasterType,
    ) -> List[float]:
        """
        Get risk scores [0.0-1.0] for a list of (lat, lon) points.
        Returns list of same length as input; None values become 0.0.
        """
        if not points:
            return []

        config = INARISK_CONFIG.get(disaster_type)
        if config is None:
            logger.warning(f"No InaRISK config for {disaster_type}; returning zeros")
            return [0.0] * len(points)

        results = [0.0] * len(points)
        batches = [points[i:i + self.batch_size]
                   for i in range(0, len(points), self.batch_size)]

        for b_idx, batch in enumerate(batches):
            logger.debug(f"InaRISK batch {b_idx+1}/{len(batches)} ({len(batch)} points)")
            try:
                scores = self._query_batch(batch, config)
                start = b_idx * self.batch_size
                for i, score in enumerate(scores):
                    results[start + i] = score
            except Exception as e:
                logger.warning(f"InaRISK batch {b_idx} failed: {e} — using 0.0")

            if b_idx < len(batches) - 1:
                time.sleep(self.rate_limit_s)

        return results

    def get_risk_layers(
        self,
        points: List[Tuple[float, float]],
        disaster_type: DisasterType,
    ) -> List[RiskLayer]:
        """Return RiskLayer objects for each point."""
        scores = self.get_risk_scores(points, disaster_type)
        return [
            RiskLayer(
                lat=pt[0], lon=pt[1],
                disaster_type=disaster_type,
                risk_score=score
            )
            for pt, score in zip(points, scores)
        ]

    def get_all_risk_scores(
        self,
        points: List[Tuple[float, float]],
        disaster_types: Optional[List[DisasterType]] = None,
    ) -> Dict[str, List[float]]:
        """
        Query all disaster types. Returns dict: disaster_type -> [scores].
        """
        if disaster_types is None:
            disaster_types = list(DisasterType)

        all_scores = {}
        for dt in disaster_types:
            all_scores[dt.value] = self.get_risk_scores(points, dt)
        return all_scores

    # ------------------------------------------------------------------ #
    # Internal query methods
    # ------------------------------------------------------------------ #

    def _query_batch(
        self, points: List[Tuple[float, float]], config: dict
    ) -> List[float]:
        """
        Query InaRISK for a batch of points using spatial query.
        For each point, performs a point-in-polygon lookup.
        Returns list of risk scores [0.0-1.0].
        """
        scores = []
        for lat, lon in points:
            score = self._query_single_point(lat, lon, config)
            scores.append(score)
        return scores

    def _query_single_point(self, lat: float, lon: float, config: dict) -> float:
        """Query risk score for a single point."""
        x, y = _latlon_to_mercator(lat, lon)
        service = config["service"]
        layer_id = config["layer_id"]
        field = config["field"]

        url = IDENTIFY_URL.format(
            base=BASE_URL, service=service, layer_id=layer_id
        )

        # Small buffer around point for polygon intersection
        tolerance = 1000  # meters in mercator
        params = {
            "geometry": f"{x},{y}",
            "geometryType": "esriGeometryPoint",
            "spatialRel": "esriSpatialRelIntersects",
            "inSR": "102100",
            "outFields": field,
            "outSR": "4326",
            "f": "json",
            "returnGeometry": "false",
            "resultRecordCount": 1,
            "where": "1=1",
        }

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug(f"InaRISK query failed for ({lat},{lon}): {e}")
            return 0.0

        features = data.get("features", [])
        if not features:
            return 0.0

        attrs = features[0].get("attributes", {})
        raw_val = attrs.get(field, attrs.get("INDEKS_BAHAYA", None))

        if raw_val is None:
            return 0.0

        try:
            score = float(raw_val)
            # InaRISK values are typically 1-3 (low/medium/high)
            # Normalize to [0, 1]
            if score > 1.0:
                score = (score - 1.0) / 2.0  # 1→0.0, 2→0.5, 3→1.0
            return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------ #
    # Utility: assign risk to model objects
    # ------------------------------------------------------------------ #

    def enrich_villages_with_risk(self, villages, disaster_type: DisasterType) -> None:
        """In-place: add risk scores to village objects."""
        points = [(v.centroid_lat, v.centroid_lon) for v in villages]
        scores = self.get_risk_scores(points, disaster_type)
        for v, score in zip(villages, scores):
            v.risk_scores[disaster_type.value] = score

    def enrich_shelters_with_risk(self, shelters, disaster_type: DisasterType) -> None:
        """In-place: add risk scores to shelter objects."""
        points = [(s.centroid_lat, s.centroid_lon) for s in shelters]
        scores = self.get_risk_scores(points, disaster_type)
        for s, score in zip(shelters, scores):
            s.risk_scores[disaster_type.value] = score
