"""
InaRISK API client for disaster risk index extraction.
API: https://gis.bnpb.go.id/server/rest/services/inarisk

Supported hazard layers:
  Primary:
    earthquake   — INDEKS_BAHAYA_GEMPABUMI
    volcano      — INDEKS_BAHAYA_GUNUNGAPI
    flood        — INDEKS_BAHAYA_BANJIR
    landslide    — INDEKS_BAHAYA_TANAHLONGSOR
  Secondary / compound:
    tsunami      — INDEKS_BAHAYA_TSUNAMI
    liquefaction — INDEKS_BAHAYA_LIKUEFAKSI
    flash_flood  — INDEKS_BAHAYA_BANJIRBANDANG

Compound hazard scenarios (e.g. Palu: earthquake + tsunami + liquefaction)
are supported via hazard_layers config: a dict of {hazard_type: weight} that
queries each layer independently and combines scores into a single composite
risk score per point.
"""

import json
import math
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.data.models import DisasterType, RiskLayer

logger = logging.getLogger(__name__)

# InaRISK service endpoints and field names per disaster type
INARISK_CONFIG = {
    # Primary hazard layers
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
    # Secondary / compound hazard layers
    DisasterType.TSUNAMI: {
        "service": "INDEKS_BAHAYA_TSUNAMI",
        "layer_id": 0,
        "field": "INDEKS_BAHAYA",
    },
    DisasterType.LIQUEFACTION: {
        "service": "INDEKS_BAHAYA_LIKUEFAKSI",
        "layer_id": 0,
        "field": "INDEKS_BAHAYA",
    },
    DisasterType.FLASH_FLOOD: {
        "service": "INDEKS_BAHAYA_BANJIRBANDANG",
        "layer_id": 0,
        "field": "INDEKS_BAHAYA",
    },
}

BASE_URL = "https://gis.bnpb.go.id/server/rest/services/inarisk"
IDENTIFY_URL = "{base}/{service}/ImageServer/identify"


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

    def iter_risk_scores_batched(
        self,
        points: List[Tuple[float, float]],
        disaster_type: DisasterType,
    ):
        """
        Yield (batch_points, batch_scores) one batch at a time.
        Callers that need to persist intermediate results (e.g. large grids)
        can save after each yielded batch instead of waiting for all results.
        """
        if not points:
            return

        config = INARISK_CONFIG.get(disaster_type)
        if config is None:
            logger.warning(f"No InaRISK config for {disaster_type}; returning zeros")
            for i in range(0, len(points), self.batch_size):
                batch = points[i:i + self.batch_size]
                yield batch, [0.0] * len(batch)
            return

        batches = [points[i:i + self.batch_size]
                   for i in range(0, len(points), self.batch_size)]
        n_total = len(points)
        log_every = max(1, len(batches) // 10)   # log ~10 progress lines
        for b_idx, batch in enumerate(batches):
            try:
                scores = self._query_batch(batch, config)
            except Exception as e:
                logger.warning(f"InaRISK batch {b_idx+1}/{len(batches)} failed: {e} — using 0.0")
                scores = [0.0] * len(batch)
            yield batch, scores
            pts_done = min((b_idx + 1) * self.batch_size, n_total)
            if (b_idx + 1) % log_every == 0 or b_idx == len(batches) - 1:
                logger.info(f"  InaRISK [{disaster_type.value}]: {pts_done}/{n_total} points queried …")
            if b_idx < len(batches) - 1:
                time.sleep(self.rate_limit_s)

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

        results = [0.0] * len(points)
        idx = 0
        for batch_pts, batch_scores in self.iter_risk_scores_batched(points, disaster_type):
            for score in batch_scores:
                results[idx] = score
                idx += 1
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
        """Query risk score for a single point via ImageServer identify."""
        import json as _json
        x, y = _latlon_to_mercator(lat, lon)
        service = config["service"]

        url = IDENTIFY_URL.format(base=BASE_URL, service=service)

        params = {
            "geometry": _json.dumps({
                "x": x, "y": y,
                "spatialReference": {"wkid": 102100},
            }),
            "geometryType": "esriGeometryPoint",
            "returnGeometry": "false",
            "f": "json",
        }

        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.debug(f"InaRISK query failed for ({lat},{lon}): {e}")
            return 0.0

        # ImageServer identify returns {"value": "0.85", ...}
        raw_val = data.get("value")
        if raw_val is None or str(raw_val).lower() in ("", "null", "nodata"):
            return 0.0

        try:
            score = float(raw_val)
            # Values are already in [0, 1]; guard against legacy [1, 3] scale
            if score > 1.0:
                score = (score - 1.0) / 2.0  # 1→0.0, 2→0.5, 3→1.0
            return max(0.0, min(1.0, score))
        except (TypeError, ValueError):
            return 0.0

    # ------------------------------------------------------------------ #
    # Utility: assign risk to model objects
    # ------------------------------------------------------------------ #

    def _get_poi_scores_cached(
        self,
        points: List[Tuple[float, float]],
        disaster_type: DisasterType,
        cache: dict,
        cache_key: str,
        use_cache: bool = True,
        cache_path: Optional[Path] = None,
        grid_cache_path: Optional[Path] = None,
        grid_precision: int = 2,
    ) -> List[float]:
        """
        Wrapper around iter_risk_scores_batched with a file-backed in-memory cache dict.

        cache:           Shared dict loaded from / saved to a JSON file by the caller.
        cache_key:       Top-level key in cache (e.g. "villages_volcano").
        use_cache:       When False all points are treated as missing (force re-query),
                         but fresh results are still written back to cache.
        cache_path:      When provided, cache is saved to disk after every batch so
                         progress is preserved if the process is killed.
        grid_cache_path: Path to road_risk_cache.json (edge grid cache).  When provided,
                         missing POI points are first snapped to the grid and looked up
                         there — eliminating most API calls after the first pipeline run.
        grid_precision:  Decimal places used for grid snapping (2 → ~1.1 km cells).

        Cache entries: {cache_key: {"lat6,lon6": score, ...}}
        """
        cached = cache.setdefault(cache_key, {})
        point_keys = [f"{lat:.6f},{lon:.6f}" for lat, lon in points]

        if use_cache:
            missing_indices = [i for i, k in enumerate(point_keys) if k not in cached]
        else:
            missing_indices = list(range(len(points)))

        if missing_indices:
            # ---- Stage 1: grid-snap lookup (fast, no API calls) ----
            if grid_cache_path is not None:
                gcp = Path(grid_cache_path)
                if gcp.exists():
                    try:
                        with open(gcp) as f:
                            grid_cache = json.load(f)
                        grid_layer_key = f"edges_{disaster_type.value}"
                        grid_layer = grid_cache.get(grid_layer_key, {})
                        if grid_layer:
                            snapped = 0
                            still_missing = []
                            for i in missing_indices:
                                lat, lon = points[i]
                                snap_key = (
                                    f"{round(lat, grid_precision)},"
                                    f"{round(lon, grid_precision)}"
                                )
                                if snap_key in grid_layer:
                                    cached[point_keys[i]] = grid_layer[snap_key]
                                    snapped += 1
                                else:
                                    still_missing.append(i)
                            if snapped:
                                logger.info(
                                    f"  InaRISK [{cache_key}]: {snapped} points resolved "
                                    f"from grid cache (grid_precision={grid_precision})"
                                )
                            missing_indices = still_missing
                    except Exception as e:
                        logger.debug(f"  Grid cache load failed ({gcp}): {e}")

            # ---- Stage 2: direct API query for remaining misses ----
            if missing_indices:
                missing_pts = [points[i] for i in missing_indices]
                n_cached = len(points) - len(missing_pts) if use_cache else 0
                logger.info(
                    f"  InaRISK [{cache_key}]: querying {len(missing_pts)} points "
                    f"({n_cached} cached) via API …"
                )
                for batch_pts, batch_scores in self.iter_risk_scores_batched(missing_pts, disaster_type):
                    for pt, score in zip(batch_pts, batch_scores):
                        k = f"{pt[0]:.6f},{pt[1]:.6f}"
                        cached[k] = score
                    if cache_path is not None:
                        self._save_poi_cache(cache, cache_path)
            else:
                logger.info(f"  InaRISK [{cache_key}]: all {len(points)} points resolved (cache+grid)")
        else:
            logger.info(f"  InaRISK [{cache_key}]: all {len(points)} points cached")

        return [cached[k] for k in point_keys]

    @staticmethod
    def _load_poi_cache(cache_path: Optional[Path]) -> dict:
        if cache_path is not None:
            cp = Path(cache_path)
            if cp.exists():
                with open(cp) as f:
                    return json.load(f)
        return {}

    @staticmethod
    def _save_poi_cache(cache: dict, cache_path: Optional[Path]) -> None:
        if cache_path is not None:
            cp = Path(cache_path)
            cp.parent.mkdir(parents=True, exist_ok=True)
            with open(cp, "w") as f:
                json.dump(cache, f)

    def enrich_villages_with_risk(
        self,
        villages,
        disaster_type: DisasterType,
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
        grid_cache_path: Optional[Path] = None,
    ) -> None:
        """In-place: add single-layer risk scores to village objects."""
        points = [(v.centroid_lat, v.centroid_lon) for v in villages]
        cache = self._load_poi_cache(cache_path)
        scores = self._get_poi_scores_cached(
            points, disaster_type, cache, f"villages_{disaster_type.value}",
            use_cache=use_cache, cache_path=cache_path, grid_cache_path=grid_cache_path,
        )
        self._save_poi_cache(cache, cache_path)
        for v, score in zip(villages, scores):
            v.risk_scores[disaster_type.value] = score
            v.risk_scores["composite"] = round(score, 4)

    def enrich_shelters_with_risk(
        self,
        shelters,
        disaster_type: DisasterType,
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
        grid_cache_path: Optional[Path] = None,
    ) -> None:
        """In-place: add single-layer risk scores to shelter objects."""
        points = [(s.centroid_lat, s.centroid_lon) for s in shelters]
        cache = self._load_poi_cache(cache_path)
        scores = self._get_poi_scores_cached(
            points, disaster_type, cache, f"shelters_{disaster_type.value}",
            use_cache=use_cache, cache_path=cache_path, grid_cache_path=grid_cache_path,
        )
        self._save_poi_cache(cache, cache_path)
        for s, score in zip(shelters, scores):
            s.risk_scores[disaster_type.value] = score
            s.risk_scores["composite"] = round(score, 4)

    def enrich_with_compound_risk(
        self,
        objects: list,
        hazard_layers: Dict[DisasterType, float],
        aggregation: str = "weighted_sum",
        object_prefix: str = "poi",
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
        grid_cache_path: Optional[Path] = None,
    ) -> None:
        """
        Query multiple InaRISK hazard layers and combine into a single composite
        risk score stored on each object.

        Each layer is queried independently; per-layer scores are stored in
        obj.risk_scores[layer_name] for transparency. The combined score is
        stored as obj.risk_scores["composite"].

        Args:
            objects:        List of Village, Shelter, or any object with
                            centroid_lat, centroid_lon, risk_scores attributes.
            hazard_layers:  Dict mapping DisasterType → weight. Weights do not
                            need to sum to 1.0 — they are normalised internally.
            aggregation:    "weighted_sum" — linear combination of per-layer scores.
                            "max"          — worst-case (highest) score across layers.
            object_prefix:  Cache key prefix ("villages" or "shelters").
            cache_path:     Optional path to a JSON cache file.
            use_cache:      When False, re-queries all points even if cached.
        """
        if not objects or not hazard_layers:
            return

        points = [(o.centroid_lat, o.centroid_lon) for o in objects]
        total_weight = sum(hazard_layers.values()) or 1.0

        cache = self._load_poi_cache(cache_path)

        # Query each layer once (using cache, then grid snap, then API)
        layer_scores: Dict[DisasterType, List[float]] = {}
        for dt in hazard_layers:
            logger.info(f"  Compound risk: querying {dt.value} layer "
                        f"(weight={hazard_layers[dt]/total_weight:.2f})")
            layer_scores[dt] = self._get_poi_scores_cached(
                points, dt, cache, f"{object_prefix}_{dt.value}",
                use_cache=use_cache, cache_path=cache_path,
                grid_cache_path=grid_cache_path,
            )

        self._save_poi_cache(cache, cache_path)

        # Assign per-layer and composite scores
        for i, obj in enumerate(objects):
            per_layer = {dt.value: layer_scores[dt][i] for dt in hazard_layers}
            obj.risk_scores.update(per_layer)

            if aggregation == "max":
                composite = max(per_layer.values()) if per_layer else 0.0
            else:  # weighted_sum (normalised)
                composite = sum(
                    layer_scores[dt][i] * (w / total_weight)
                    for dt, w in hazard_layers.items()
                )
            obj.risk_scores["composite"] = round(composite, 4)

    def enrich_villages_compound(
        self,
        villages: list,
        hazard_layers: Dict[DisasterType, float],
        aggregation: str = "weighted_sum",
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
        grid_cache_path: Optional[Path] = None,
    ) -> None:
        """Compound risk enrichment for villages. Logs summary after enrichment."""
        self.enrich_with_compound_risk(
            villages, hazard_layers, aggregation,
            object_prefix="villages", cache_path=cache_path, use_cache=use_cache,
            grid_cache_path=grid_cache_path,
        )
        composites = [v.risk_scores.get("composite", 0.0) for v in villages]
        if composites:
            logger.info(
                f"  Village compound risk ({aggregation}): "
                f"min={min(composites):.3f} max={max(composites):.3f} "
                f"mean={sum(composites)/len(composites):.3f}"
            )

    def enrich_shelters_compound(
        self,
        shelters: list,
        hazard_layers: Dict[DisasterType, float],
        aggregation: str = "weighted_sum",
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
        grid_cache_path: Optional[Path] = None,
    ) -> None:
        """Compound risk enrichment for shelters. Logs summary after enrichment."""
        self.enrich_with_compound_risk(
            shelters, hazard_layers, aggregation,
            object_prefix="shelters", cache_path=cache_path, use_cache=use_cache,
            grid_cache_path=grid_cache_path,
        )
        composites = [s.risk_scores.get("composite", 0.0) for s in shelters]
        if composites:
            logger.info(
                f"  Shelter compound risk ({aggregation}): "
                f"min={min(composites):.3f} max={max(composites):.3f} "
                f"mean={sum(composites)/len(composites):.3f}"
            )

    def enrich_graph_edges(
        self,
        G,                                          # nx.Graph
        node_coords: Dict[int, Tuple[float, float]],  # node_id -> (lat, lon)
        hazard_layers: Dict[DisasterType, float],
        aggregation: str = "weighted_sum",
        risk_weight: float = 0.4,
        grid_precision: int = 2,
        cache_path: Optional[Path] = None,
        use_cache: bool = True,
    ) -> int:
        """
        Query InaRISK at each road segment's midpoint and update graph edge
        risk_score + composite weight in-place.

        Midpoints are snapped to a grid of `grid_precision` decimal places
        (~1.1 km at precision=2) to deduplicate API calls.  Results are
        persisted to cache_path (JSON) so subsequent runs skip already-queried
        cells.

        Args:
            G:              NetworkX graph built by EvacuationGraphBuilder.
            node_coords:    Mapping from node_id to (lat, lon).
            hazard_layers:  {DisasterType: weight} — same as used for villages.
            aggregation:    "weighted_sum" or "max".
            risk_weight:    Multiplier inside composite weight formula
                            (must match the value used in EvacuationGraphBuilder.build).
            grid_precision: Decimal places for grid dedup.
            cache_path:     Optional path to a JSON cache file.
        Returns:
            Number of edges whose risk_score was raised above 0.0.
        """
        if G is None or G.number_of_edges() == 0:
            return 0

        total_weight = sum(hazard_layers.values()) or 1.0

        # ---- Load cache ----
        cache: dict = {}
        if cache_path is not None and Path(cache_path).exists():
            with open(cache_path) as f:
                cache = json.load(f)

        # ---- Collect unique grid keys for all edge midpoints ----
        edge_grid_keys: Dict[Tuple[int, int], str] = {}
        all_grid_keys: set = set()
        for u, v in G.edges():
            if u not in node_coords or v not in node_coords:
                continue
            u_lat, u_lon = node_coords[u]
            v_lat, v_lon = node_coords[v]
            mid_lat = (u_lat + v_lat) / 2.0
            mid_lon = (u_lon + v_lon) / 2.0
            gk = f"{round(mid_lat, grid_precision)},{round(mid_lon, grid_precision)}"
            edge_grid_keys[(u, v)] = gk
            all_grid_keys.add(gk)

        logger.info(
            f"enrich_graph_edges: {G.number_of_edges()} edges → "
            f"{len(all_grid_keys)} unique grid cells"
        )

        # ---- Query each hazard layer; compute composite per grid cell ----
        composite_by_key: Dict[str, float] = {gk: 0.0 for gk in all_grid_keys}

        for dt, weight in hazard_layers.items():
            layer_key = f"edges_{dt.value}"
            layer_cached: dict = cache.get(layer_key, {})

            missing = [gk for gk in all_grid_keys
                       if gk not in layer_cached or not use_cache]
            if missing:
                pts = [
                    (float(gk.split(",")[0]), float(gk.split(",")[1]))
                    for gk in missing
                ]
                logger.info(
                    f"  Road risk [{dt.value}]: querying {len(pts)} grid cells "
                    f"({len(all_grid_keys) - len(missing)} cached) …"
                )
                scores = self.get_risk_scores(pts, dt)
                layer_cached.update(dict(zip(missing, scores)))
                cache[layer_key] = layer_cached
                if cache_path is not None:
                    cp = Path(cache_path)
                    cp.parent.mkdir(parents=True, exist_ok=True)
                    with open(cp, "w") as f:
                        json.dump(cache, f)
            else:
                logger.info(
                    f"  Road risk [{dt.value}]: all {len(all_grid_keys)} cells cached"
                )

            for gk in all_grid_keys:
                score = layer_cached.get(gk, 0.0)
                if aggregation == "max":
                    composite_by_key[gk] = max(composite_by_key[gk], score)
                else:  # weighted_sum (normalised)
                    composite_by_key[gk] += score * (weight / total_weight)

        # ---- Assign composite risk to graph edges; recompute weight ----
        updated = 0
        for (u, v), gk in edge_grid_keys.items():
            if not G.has_edge(u, v):
                continue
            new_risk = round(composite_by_key.get(gk, 0.0), 4)
            data = G[u][v]
            if new_risk > data.get("risk_score", 0.0):
                data["risk_score"] = new_risk
                data["weight"] = (
                    data["length_m"]
                    * data["quality_weight"]
                    * (1.0 + risk_weight * new_risk)
                )
                updated += 1

        all_composites = list(composite_by_key.values())
        logger.info(
            f"enrich_graph_edges: updated {updated}/{G.number_of_edges()} edges; "
            f"composite risk {min(all_composites):.3f}–{max(all_composites):.3f}"
        )
        return updated
