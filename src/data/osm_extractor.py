"""
OSM data extraction module.
Extracts road networks, villages, and shelters from OpenStreetMap.
Supports live extraction (osmnx) and loading pre-extracted files.
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

try:
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.cache_folder = "data/raw/osm_cache/http"
    HAS_OSMNX = True
except ImportError:
    HAS_OSMNX = False
    logging.warning("osmnx not available — OSM live extraction disabled")

from src.data.models import Village, Shelter, NetworkNode, NetworkEdge, RegionOfInterest, RegionType

logger = logging.getLogger(__name__)

# OSM highway types allowed for evacuation routing
EVACUATION_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "service", "living_street",
    "pedestrian", "footway", "path", "track"
}

# Speed limits by road type (km/h)
SPEED_BY_HIGHWAY = {
    "motorway": 100, "trunk": 80, "primary": 60, "secondary": 50,
    "tertiary": 40, "unclassified": 30, "residential": 30,
    "service": 20, "living_street": 15, "pedestrian": 10,
    "footway": 8, "path": 8, "track": 15
}


class OSMExtractor:
    """
    Extracts road network, villages, and shelters from OpenStreetMap.
    Results are cached to disk for reproducibility.
    """

    def __init__(self, cache_dir: str = "data/raw/osm_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def extract_road_network(
        self,
        region: RegionOfInterest,
        network_type: str = "all",
        road_types: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """
        Extract road network for the region.

        Args:
            network_type: osmnx network type (all|drive|walk).
            road_types:   Dict of {highway_type: {speed_kmh, capacity_veh_h}}.
                          Only highway types present in this dict are kept.
                          Speed and capacity values override OSM-derived defaults.
                          If None, all extracted edges are kept with OSM defaults.

        Returns (nodes, edges).
        """
        bbox = region.to_bbox()
        # Include road type fingerprint in cache key so changes invalidate cache
        import hashlib as _hl
        rt_sig = _hl.md5(str(sorted(road_types.keys()) if road_types else "all").encode()).hexdigest()[:8]
        cache_key = self._cache_key("network", bbox, f"{network_type}_{rt_sig}")
        cache_path = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached network from {cache_path}")
            return self._load_network_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        logger.info(f"Extracting road network for bbox {bbox}")
        south, west, north, east = bbox

        try:
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type=network_type,
                retain_all=True,
                simplify=True,
            )
        except Exception as e:
            logger.error(f"OSM network extraction failed: {e}")
            raise

        nodes, edges = self._oxgraph_to_models(G, road_types=road_types)

        self._save_network_cache(cache_path, nodes, edges)
        n_types = len(set(e.highway_type for e in edges))
        logger.info(
            f"Extracted {len(nodes)} nodes, {len(edges)} edges "
            f"({n_types} highway types"
            + (f", filtered to {list(road_types.keys())}" if road_types else "")
            + ")"
        )
        return nodes, edges

    def extract_villages(
        self,
        region: RegionOfInterest,
        admin_levels: List[int] = None,
        population_density_per_km2: float = 800.0,
        max_population_per_village: int = 50000,
        use_cache: bool = True,
        sources: List[str] = None,
        place_tags: List[str] = None,
        place_settings: Optional[dict] = None,
        place_radius_m: Optional[float] = None,
        cluster_eps_m: float = 300.0,
        cluster_min_buildings: int = 10,
        persons_per_dwelling: float = 4.0,
        building_persons: Optional[dict] = None,
    ) -> List[Village]:
        """
        Extract village population areas from one or more OSM sources.

        Sources are processed in order; each adds only settlements whose
        centroid is not already covered by a polygon from a previous source.

        Available sources (set via ``sources`` parameter):
          ``admin_boundary``
              OSM boundary=administrative polygons. admin_levels controls
              which levels are tried (9=desa, 8=kecamatan, 7=kabupaten).
              The first level that yields polygons is used.
          ``place_nodes``
              OSM place=village|hamlet|... point nodes. Each node is
              buffered to a synthetic circular polygon. Useful where admin
              boundaries are unmapped (remote islands, highlands).
          ``building_clusters``
              OSM building footprints grouped by DBSCAN. Each cluster
              becomes a synthetic convex-hull polygon. Population is
              cluster_building_count × persons_per_dwelling.

        Default sources: [``admin_boundary``, ``place_nodes``] — admin
        polygons first, place nodes fill remaining gaps.
        """
        if admin_levels is None:
            admin_levels = [9, 8, 7]
        if sources is None:
            sources = ["admin_boundary", "place_nodes"]
        if place_tags is None:
            place_tags = ["village", "hamlet", "town", "suburb", "quarter"]

        bbox = region.to_bbox()
        src_key = "_".join(sources)
        variant = f"src{src_key}_al{'_'.join(str(l) for l in admin_levels)}_d{int(population_density_per_km2)}"
        cache_key = self._cache_key("villages", bbox, variant)
        cache_path = self.cache_dir / f"{cache_key}.geojson"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached villages from {cache_path}")
            return self._load_villages_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        south, west, north, east = bbox
        ox_bbox = (west, south, east, north)  # osmnx 2.x: (left, bottom, right, top)

        all_villages: List[Village] = []

        for source in sources:
            if source == "admin_boundary":
                new = self._villages_from_admin_boundary(
                    ox_bbox, admin_levels, population_density_per_km2, max_population_per_village
                )
            elif source == "place_nodes":
                new = self._villages_from_place_nodes(
                    ox_bbox, place_tags, place_settings or {},
                    place_radius_m, population_density_per_km2,
                    max_population_per_village
                )
            elif source == "building_clusters":
                new = self._villages_from_building_clusters(
                    ox_bbox, cluster_eps_m, cluster_min_buildings,
                    persons_per_dwelling, building_persons or {},
                    max_population_per_village
                )
            else:
                logger.warning(f"Unknown village source '{source}' — skipping")
                continue

            if not new:
                logger.info(f"village source '{source}': no results")
                continue

            added = self._add_uncovered_villages(new, all_villages)
            logger.info(f"village source '{source}': {len(new)} found, {len(added)} added (not already covered)")
            all_villages.extend(added)

        if not all_villages:
            logger.warning(f"No villages found from any source ({sources}). Returning empty list.")
            return []

        all_villages = self._deduplicate_villages(all_villages)
        self._save_villages_cache(cache_path, all_villages)
        logger.info(
            f"Extracted {len(all_villages)} villages total "
            f"(sources={sources}, density={population_density_per_km2} p/km², "
            f"total_pop={sum(v.population for v in all_villages):,})"
        )
        return all_villages

    # ------------------------------------------------------------------
    # Village source implementations
    # ------------------------------------------------------------------

    def _villages_from_admin_boundary(
        self,
        ox_bbox: tuple,
        admin_levels: List[int],
        population_density_per_km2: float,
        max_population_per_village: int,
    ) -> List[Village]:
        """Extract villages from OSM boundary=administrative polygons."""
        for level in admin_levels:
            try:
                gdf = ox.features_from_bbox(
                    bbox=ox_bbox,
                    tags={"admin_level": str(level), "boundary": "administrative"}
                )
            except Exception as e:
                logger.warning(f"admin_level={level} query failed: {e}")
                continue

            if "admin_level" in gdf.columns:
                gdf = gdf[gdf["admin_level"].astype(str) == str(level)]

            poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
            gdf_poly = gdf[poly_mask].copy()

            if gdf_poly.empty:
                logger.info(f"admin_level={level}: no polygon boundaries found — trying next level")
                continue

            min_area_m2 = 100_000.0 if level >= 9 else 500_000.0
            try:
                _clon = float(gdf_poly.to_crs("EPSG:4326").geometry.unary_union.centroid.x)
                _clat = float(gdf_poly.to_crs("EPSG:4326").geometry.unary_union.centroid.y)
                _zone = int((_clon + 180) / 6) + 1
                _epsg = (32700 + _zone) if _clat < 0 else (32600 + _zone)
                gdf_utm = gdf_poly.to_crs(f"EPSG:{_epsg}")
                gdf_poly = gdf_poly[gdf_utm.geometry.area >= min_area_m2].copy()
            except Exception:
                pass

            if gdf_poly.empty:
                logger.info(f"admin_level={level}: all polygons below area threshold — trying next level")
                continue

            villages = self._gdf_to_villages(gdf_poly, admin_level=level)
            self._assign_population_from_area(villages, population_density_per_km2, max_population_per_village)
            logger.info(f"admin_level={level}: {len(villages)} boundary polygons")
            return villages

        return []

    def _villages_from_place_nodes(
        self,
        ox_bbox: tuple,
        place_tags: List[str],
        place_settings: dict,
        default_radius_m: Optional[float],
        population_density_per_km2: float,
        max_population_per_village: int,
    ) -> List[Village]:
        """
        Extract villages from OSM place=* point nodes, buffered to synthetic circles.

        Each node's radius and population density are looked up from
        place_settings by its ``place`` tag value. Tags not in place_settings
        fall back to default_radius_m (auto if None) and population_density_per_km2.
        """
        try:
            gdf = ox.features_from_bbox(
                bbox=ox_bbox,
                tags={"place": place_tags},
            )
        except Exception as e:
            logger.warning(f"place-node query failed: {e}")
            return []

        gdf_pts = gdf[gdf.geometry.geom_type == "Point"].copy()
        if gdf_pts.empty:
            return []

        import math
        import pyproj
        from shapely.geometry import Point as _Pt
        from shapely.ops import transform as _transform

        # Pre-compute fallback radius from global density if not given
        if default_radius_m is None:
            target_pop = 2000.0
            default_radius_m = math.sqrt(
                target_pop / max(population_density_per_km2, 1.0) / math.pi
            ) * 1000.0
            default_radius_m = max(200.0, min(default_radius_m, 2000.0))

        villages = []
        for idx, row in gdf_pts.iterrows():
            pt = row.geometry
            lon, lat = float(pt.x), float(pt.y)
            name = str(row.get("name", row.get("name:en", f"place_{idx}")))
            place_type = str(row.get("place", "") or "").strip().lower()

            # Per-tag radius and density; fall back to globals for unknown types
            tag_cfg  = place_settings.get(place_type, {})
            radius_m = float(tag_cfg.get("radius_m", default_radius_m))
            density  = float(tag_cfg.get("pop_density", population_density_per_km2))

            zone = int((lon + 180) / 6) + 1
            epsg = (32700 + zone) if lat < 0 else (32600 + zone)
            try:
                to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
                to_wgs = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
                pt_utm     = _transform(to_utm, _Pt(lon, lat))
                circle_utm = pt_utm.buffer(radius_m)
                circle_wgs = _transform(to_wgs, circle_utm)
                area_m2    = float(circle_utm.area)
                geom_wkt   = circle_wgs.wkt
            except Exception:
                area_m2  = math.pi * radius_m ** 2
                geom_wkt = None

            pop = max(1, min(int(area_m2 / 1_000_000 * density), max_population_per_village))
            villages.append(Village(
                village_id=str(idx),
                name=name,
                centroid_lat=lat,
                centroid_lon=lon,
                population=pop,
                area_m2=area_m2,
                admin_level=10,
                geometry_wkt=geom_wkt,
            ))

        # Summary: show radius range across place types used
        if villages:
            radii = sorted({
                float((place_settings.get(str(r.get("place", "") or ""), {})).get(
                    "radius_m", default_radius_m))
                for _, r in gdf_pts.iterrows()
            })
            logger.info(
                f"place-nodes: {len(villages)} settlements "
                f"(radius {radii[0]:.0f}–{radii[-1]:.0f} m, per-tag density)"
            )
        return villages

    def _villages_from_building_clusters(
        self,
        ox_bbox: tuple,
        eps_m: float,
        min_buildings: int,
        persons_per_dwelling: float,
        building_persons: dict,
        max_population_per_village: int,
    ) -> List[Village]:
        """
        Extract villages by DBSCAN-clustering OSM building footprints.
        Each cluster's convex hull becomes a synthetic village polygon.

        Population is the sum of per-building occupancy across the cluster.
        Occupancy per building is looked up from building_persons by the
        OSM ``building`` tag value; unknown types fall back to
        persons_per_dwelling. Types explicitly set to 0 (e.g. commercial,
        religious) are excluded from the population count.
        """
        try:
            gdf = ox.features_from_bbox(
                bbox=ox_bbox,
                tags={"building": True},
            )
        except Exception as e:
            logger.warning(f"building query failed: {e}")
            return []

        if gdf.empty:
            return []

        # Compute centroids in WGS84, reproject to local UTM for metric DBSCAN
        gdf_wgs = gdf.to_crs("EPSG:4326")
        centroids = gdf_wgs.geometry.centroid
        if centroids.empty:
            return []

        ref_lon = float(centroids.x.median())
        ref_lat = float(centroids.y.median())
        zone = int((ref_lon + 180) / 6) + 1
        epsg = (32700 + zone) if ref_lat < 0 else (32600 + zone)

        try:
            import pyproj
            from shapely.ops import transform as _transform
            to_utm = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform
            to_wgs = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True).transform
            gdf_utm = gdf_wgs.to_crs(f"EPSG:{epsg}")
        except Exception as e:
            logger.warning(f"building_clusters: UTM reproject failed: {e}")
            return []

        centroids_utm = gdf_utm.geometry.centroid
        coords = list(zip(centroids_utm.x, centroids_utm.y))
        if not coords:
            return []

        try:
            from sklearn.cluster import DBSCAN
            import numpy as np
            labels = DBSCAN(eps=eps_m, min_samples=min_buildings).fit_predict(
                np.array(coords)
            )
        except ImportError:
            logger.warning("building_clusters: scikit-learn not installed — skipping")
            return []

        villages = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue  # noise
            mask = labels == cluster_id
            cluster_pts_utm = [coords[i] for i, m in enumerate(mask) if m]
            cluster_rows = [i for i, m in enumerate(mask) if m]

            from shapely.geometry import MultiPoint as _MP, Point as _Pt
            from shapely.ops import transform as _transform
            hull_utm = _MP([_Pt(x, y) for x, y in cluster_pts_utm]).convex_hull
            # Degenerate hulls (Point or LineString) occur when all buildings are
            # collinear or only 2 buildings exist — buffer to a thin polygon
            if hull_utm.geom_type not in ("Polygon", "MultiPolygon"):
                hull_utm = hull_utm.buffer(max(eps_m * 0.1, 10.0))
            hull_wgs = _transform(to_wgs, hull_utm)

            centroid_utm = hull_utm.centroid
            centroid_wgs = _transform(to_wgs, centroid_utm)
            area_m2 = float(hull_utm.area)

            # Name from most common name tag in cluster; population from per-type occupancy
            name_counts: dict = {}
            total_persons = 0.0
            for i in cluster_rows:
                try:
                    row_wgs = gdf_wgs.iloc[i]
                    n = str(row_wgs.get("name", ""))
                    if n and n != "None":
                        name_counts[n] = name_counts.get(n, 0) + 1
                    btype = str(row_wgs.get("building", "yes") or "yes").strip().lower()
                    occupancy = building_persons.get(btype, persons_per_dwelling)
                    total_persons += occupancy
                except Exception:
                    total_persons += persons_per_dwelling
            name = max(name_counts, key=name_counts.get) if name_counts else f"cluster_{cluster_id}"

            pop = max(1, min(int(total_persons), max_population_per_village))
            villages.append(Village(
                village_id=f"bldg_cluster_{cluster_id}",
                name=name,
                centroid_lat=float(centroid_wgs.y),
                centroid_lon=float(centroid_wgs.x),
                population=pop,
                area_m2=area_m2,
                admin_level=11,
                geometry_wkt=hull_wgs.wkt,
            ))

        logger.info(
            f"building_clusters: {len(villages)} clusters "
            f"(eps={eps_m} m, min_buildings={min_buildings}, "
            f"persons_per_dwelling={persons_per_dwelling})"
        )
        return villages

    def _add_uncovered_villages(
        self,
        candidates: List[Village],
        existing: List[Village],
    ) -> List[Village]:
        """
        Return candidates whose centroid is NOT inside any existing village polygon.
        If existing is empty, all candidates are returned.
        """
        if not existing:
            return candidates

        from shapely.geometry import Point as _Pt
        from shapely.wkt import loads as _wkt_loads
        from shapely.ops import unary_union

        polys = []
        for v in existing:
            if v.geometry_wkt:
                try:
                    polys.append(_wkt_loads(v.geometry_wkt))
                except Exception:
                    pass
        if not polys:
            return candidates

        covered = unary_union(polys)
        added = []
        for c in candidates:
            pt = _Pt(c.centroid_lon, c.centroid_lat)
            if not covered.contains(pt):
                added.append(c)
        return added

    def _assign_population_from_area(
        self,
        villages: List[Village],
        density_per_km2: float,
        max_pop: int,
    ) -> None:
        """Assign population = area_km² × density, capped at max_pop."""
        for v in villages:
            if v.area_m2 > 0:
                area_km2 = v.area_m2 / 1_000_000.0
                v.population = max(1, min(int(area_km2 * density_per_km2), max_pop))
            else:
                v.population = 0  # no area — will be handled by PopulationLoader fallback

    def extract_shelters(
        self,
        region: RegionOfInterest,
        shelter_tags: Optional[Dict[str, Any]] = None,
        min_area_m2: float = 1000.0,
        m2_per_person: float = 2.0,
        use_cache: bool = True,
    ) -> List[Shelter]:
        """
        Extract shelter locations as boundary polygons only.

        Only features with a closed Polygon/MultiPolygon geometry AND
        area >= min_area_m2 are kept. Point nodes are discarded.
        Capacity is estimated from area / m2_per_person.

        Args:
            shelter_tags:  Dict of {osm_tag_key: value_or_list} to query.
                           Default: hospital + assembly_point only.
            min_area_m2:   Minimum polygon area to qualify as a shelter.
                           Default 1000 m² (≈ 32×32 m building footprint).
            m2_per_person: Floor area per occupant for capacity estimation.
                           Default 2.0 m²/person.
        """
        if shelter_tags is None:
            shelter_tags = {
                "amenity":    ["hospital"],
                "emergency":  ["assembly_point", "shelter"],
            }

        bbox = region.to_bbox()
        # Cache key encodes tag fingerprint + area threshold
        import hashlib as _hl
        tag_sig = _hl.md5(str(sorted(str(shelter_tags).split())).encode()).hexdigest()[:8]
        variant = f"poly_a{int(min_area_m2)}_t{tag_sig}"
        cache_key = self._cache_key("shelters", bbox, variant)
        cache_path = self.cache_dir / f"{cache_key}.geojson"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached shelters from {cache_path}")
            return self._load_shelters_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        south, west, north, east = bbox
        ox_bbox = (west, south, east, north)
        shelters: List[Shelter] = []

        for tag_key, tag_values in shelter_tags.items():
            for val in (tag_values if isinstance(tag_values, list) else [tag_values]):
                try:
                    gdf = ox.features_from_bbox(bbox=ox_bbox, tags={tag_key: val})
                except Exception as e:
                    logger.debug(f"No {tag_key}={val} found: {e}")
                    continue

                # Keep only closed polygons
                poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                gdf_poly = gdf[poly_mask].copy()
                if gdf_poly.empty:
                    logger.debug(f"{tag_key}={val}: no polygon features")
                    continue

                # Area filter in UTM for accuracy
                try:
                    gdf_utm = gdf_poly.to_crs("EPSG:32749")
                    area_mask = gdf_utm.geometry.area >= min_area_m2
                    gdf_poly = gdf_poly[area_mask].copy()
                    areas_m2 = gdf_utm.geometry.area[area_mask].values
                except Exception:
                    areas_m2 = None

                raw = self._gdf_to_shelters(gdf_poly, shelter_type=val,
                                            areas_m2=areas_m2,
                                            m2_per_person=m2_per_person)
                logger.info(f"  {tag_key}={val}: {len(raw)} polygon shelters "
                            f"(≥{min_area_m2:.0f} m²)")
                shelters += raw

        shelters = self._deduplicate_shelters(shelters)
        self._save_shelters_cache(cache_path, shelters)
        total_cap = sum(s.capacity for s in shelters)
        logger.info(
            f"Extracted {len(shelters)} shelter polygons "
            f"(min_area={min_area_m2:.0f} m², total_capacity={total_cap:,})"
        )
        return shelters

    # ------------------------------------------------------------------ #
    # Load pre-extracted files
    # ------------------------------------------------------------------ #

    def load_network_from_json(self, path: str) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """Load pre-extracted network from JSON (NetworkX adjacency format)."""
        with open(path) as f:
            data = json.load(f)
        return self._load_network_cache(path)

    def load_network_from_pycgr(self, path: str) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """
        Load network from PYCGR file (OsmToRoadGraph format).
        Format:
          Line 7: total_nodes
          Line 8: total_edges
          Nodes: node_id lat lon
          Edges: src_id tgt_id length highway max_speed bidirectional
        """
        nodes = []
        edges = []
        with open(path) as f:
            lines = f.readlines()

        # Find node/edge counts (skip comment lines starting with #)
        data_lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        idx = 0
        n_nodes = int(data_lines[idx]); idx += 1
        n_edges = int(data_lines[idx]); idx += 1

        for _ in range(n_nodes):
            parts = data_lines[idx].split(); idx += 1
            nodes.append(NetworkNode(
                node_id=int(parts[0]),
                lat=float(parts[1]),
                lon=float(parts[2])
            ))

        for _ in range(n_edges):
            parts = data_lines[idx].split(); idx += 1
            edges.append(NetworkEdge(
                source_id=int(parts[0]),
                target_id=int(parts[1]),
                length_m=float(parts[2]),
                highway_type=parts[3] if len(parts) > 3 else "residential",
                max_speed_kmh=float(parts[4]) if len(parts) > 4 else 30.0,
                bidirectional=bool(int(parts[5])) if len(parts) > 5 else True,
            ))

        logger.info(f"Loaded PYCGR: {len(nodes)} nodes, {len(edges)} edges from {path}")
        return nodes, edges

    def load_villages_from_geojson(self, path: str) -> List[Village]:
        """Load pre-extracted villages from GeoJSON."""
        return self._load_villages_cache(path)

    def load_shelters_from_geojson(self, path: str) -> List[Shelter]:
        """Load pre-extracted shelters from GeoJSON."""
        return self._load_shelters_cache(path)

    # ------------------------------------------------------------------ #
    # Internal conversion helpers
    # ------------------------------------------------------------------ #

    def _oxgraph_to_models(
        self,
        G,
        road_types: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """
        Convert osmnx graph to NetworkNode/NetworkEdge lists.

        road_types: if provided, only edges whose highway type is a key in this
        dict are kept. Speed and capacity values from road_types override OSM data.
        """
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        node_map = {}
        nodes = []
        for seq_id, (osm_id, row) in enumerate(nodes_gdf.iterrows()):
            node_map[osm_id] = seq_id
            nodes.append(NetworkNode(
                node_id=seq_id,
                lat=float(row["y"]),
                lon=float(row["x"]),
            ))

        edges = []
        seen = set()
        skipped = 0
        for (u, v, _), row in edges_gdf.iterrows():
            if u not in node_map or v not in node_map:
                continue
            src = node_map[u]
            tgt = node_map[v]
            key = (min(src, tgt), max(src, tgt))
            if key in seen:
                continue
            seen.add(key)

            highway = row.get("highway", "unclassified")
            if isinstance(highway, list):
                highway = highway[0]
            highway = str(highway).strip()

            # Filter to configured road types if provided
            if road_types is not None and highway not in road_types:
                skipped += 1
                continue

            # Speed: use configured value if provided, else parse OSM, else default
            if road_types is not None and highway in road_types:
                cfg_speed = road_types[highway].get("speed_kmh")
            else:
                cfg_speed = None

            speed_raw = row.get("maxspeed", None)
            if isinstance(speed_raw, list):
                speed_raw = speed_raw[0]
            try:
                speed = float(str(speed_raw).replace(" mph","").replace(" kmh","").strip())
                if speed != speed or speed <= 0:
                    raise ValueError("invalid speed")
            except (TypeError, ValueError):
                speed = cfg_speed or SPEED_BY_HIGHWAY.get(highway, 30.0)

            # If OSM had a valid speed but config overrides it, use config
            if cfg_speed is not None:
                speed = float(cfg_speed)

            lanes_raw = row.get("lanes", 1)
            try:
                lanes = int(lanes_raw) if lanes_raw else 1
            except (TypeError, ValueError):
                lanes = 1

            edges.append(NetworkEdge(
                source_id=src,
                target_id=tgt,
                length_m=float(row.get("length", 0.0)),
                highway_type=highway,
                max_speed_kmh=speed,
                bidirectional=not row.get("oneway", False),
                lanes=lanes,
            ))

        if skipped:
            logger.info(f"  Filtered out {skipped} edges (highway type not in road_types config)")
        return nodes, edges

    def _gdf_to_villages(self, gdf: gpd.GeoDataFrame, admin_level: int = 9) -> List[Village]:
        # Reproject to local UTM for accurate m² area (auto-select zone from centroid)
        try:
            centroid_lon = float(gdf.to_crs("EPSG:4326").geometry.unary_union.centroid.x)
            centroid_lat = float(gdf.to_crs("EPSG:4326").geometry.unary_union.centroid.y)
            zone = int((centroid_lon + 180) / 6) + 1
            epsg = (32700 + zone) if centroid_lat < 0 else (32600 + zone)
            gdf_utm = gdf.to_crs(f"EPSG:{epsg}")
        except Exception:
            gdf_utm = gdf

        # Area cap scales with admin level: kecamatan (level 8) can be up to 500 km²
        area_cap_m2 = 100_000_000.0 if admin_level <= 8 else 100_000_000.0  # 100 km² for desa, 500 km² for kecamatan
        if admin_level <= 8:
            area_cap_m2 = 500_000_000.0  # 500 km²

        villages = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid if hasattr(geom, 'centroid') else geom
            # Use UTM geometry for area; fall back to 0 for non-polygon features (nodes)
            try:
                geom_utm = gdf_utm.loc[idx].geometry
                area_m2 = float(geom_utm.area) if hasattr(geom_utm, 'area') else 0.0
            except Exception:
                area_m2 = 0.0
            area_m2 = min(area_m2, area_cap_m2)
            name = str(row.get("name", row.get("name:en", f"village_{idx}")))
            villages.append(Village(
                village_id=str(idx),
                name=name,
                centroid_lat=float(centroid.y),
                centroid_lon=float(centroid.x),
                population=0,       # filled by population loader
                area_m2=area_m2,
                admin_level=admin_level,
                geometry_wkt=geom.wkt,
            ))
        return villages

    def _gdf_to_shelters(
        self,
        gdf: gpd.GeoDataFrame,
        shelter_type: str,
        areas_m2=None,          # pre-computed UTM areas (numpy array, same row order)
        m2_per_person: float = 2.0,
    ) -> List[Shelter]:
        shelters = []
        for i, (idx, row) in enumerate(gdf.iterrows()):
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid if hasattr(geom, "centroid") else geom
            # Use pre-computed UTM area when available (accurate); fall back to 0
            if areas_m2 is not None and i < len(areas_m2):
                area_m2 = float(areas_m2[i])
            else:
                area_m2 = 0.0
            capacity = max(10, int(area_m2 / m2_per_person)) if area_m2 > 0 else 0
            name = str(row.get("name", row.get("name:en", f"shelter_{idx}")))
            shelters.append(Shelter(
                shelter_id=str(idx),
                name=name,
                centroid_lat=float(centroid.y),
                centroid_lon=float(centroid.x),
                capacity=capacity,
                shelter_type=shelter_type,
                area_m2=area_m2,
                geometry_wkt=geom.wkt,
            ))
        return shelters

    def _deduplicate_villages(self, villages: List[Village], threshold_m: float = 100.0) -> List[Village]:
        """Remove duplicate villages by proximity."""
        seen = {}
        result = []
        for v in villages:
            key = f"{round(v.centroid_lat, 4)}_{round(v.centroid_lon, 4)}"
            if key not in seen:
                seen[key] = True
                result.append(v)
        return result

    def _deduplicate_shelters(self, shelters: List[Shelter], threshold_m: float = 50.0) -> List[Shelter]:
        seen = {}
        result = []
        for s in shelters:
            key = f"{round(s.centroid_lat, 4)}_{round(s.centroid_lon, 4)}"
            if key not in seen:
                seen[key] = True
                result.append(s)
        return result

    # ------------------------------------------------------------------ #
    # Cache helpers
    # ------------------------------------------------------------------ #

    def _cache_key(self, prefix: str, bbox: tuple, variant: str) -> str:
        raw = f"{prefix}_{bbox}_{variant}"
        return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:12]}"

    def _save_network_cache(self, path: Path, nodes: List[NetworkNode], edges: List[NetworkEdge]):
        data = {
            "nodes": [{"id": n.node_id, "lat": n.lat, "lon": n.lon} for n in nodes],
            "edges": [{"src": e.source_id, "tgt": e.target_id, "len": e.length_m,
                       "hw": e.highway_type, "spd": e.max_speed_kmh,
                       "bi": e.bidirectional, "lanes": e.lanes} for e in edges]
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def _load_network_cache(self, path) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        with open(path) as f:
            data = json.load(f)
        nodes = [NetworkNode(node_id=n["id"], lat=n["lat"], lon=n["lon"])
                 for n in data["nodes"]]
        edges = [NetworkEdge(
            source_id=e["src"], target_id=e["tgt"], length_m=e["len"],
            highway_type=e.get("hw", "residential"),
            max_speed_kmh=e.get("spd", 30.0),
            bidirectional=e.get("bi", True),
            lanes=e.get("lanes", 1),
        ) for e in data["edges"]]
        return nodes, edges

    def _save_villages_cache(self, path: Path, villages: List[Village]):
        import json as _json
        from shapely.wkt import loads as _wkt_loads
        from shapely.geometry import mapping as _mapping
        features = []
        for v in villages:
            # Prefer full polygon geometry; fall back to point centroid
            if v.geometry_wkt:
                try:
                    geom = _mapping(_wkt_loads(v.geometry_wkt))
                except Exception:
                    geom = {"type": "Point", "coordinates": [v.centroid_lon, v.centroid_lat]}
            else:
                geom = {"type": "Point", "coordinates": [v.centroid_lon, v.centroid_lat]}
            features.append({
                "type": "Feature",
                "properties": {
                    "village_id":  v.village_id,
                    "name":        v.name,
                    "population":  v.population,
                    "area_m2":     v.area_m2,
                    "admin_level": v.admin_level,
                },
                "geometry": geom,
            })
        with open(path, "w") as f:
            _json.dump({"type": "FeatureCollection", "features": features}, f)

    def _load_villages_cache(self, path) -> List[Village]:
        with open(path) as f:
            data = json.load(f)
        villages = []
        for i, feat in enumerate(data.get("features", [])):
            props = feat.get("properties", {})
            geom  = feat.get("geometry", {})
            gtype = geom.get("type", "Point")
            coords = geom.get("coordinates", [0, 0])
            # Extract centroid from any geometry type
            if gtype == "Point":
                lon, lat = float(coords[0]), float(coords[1])
            elif gtype == "Polygon":
                ring = coords[0]
                lon = sum(c[0] for c in ring) / len(ring)
                lat = sum(c[1] for c in ring) / len(ring)
            elif gtype == "MultiPolygon":
                ring = max(coords, key=lambda p: len(p[0]))[0]
                lon = sum(c[0] for c in ring) / len(ring)
                lat = sum(c[1] for c in ring) / len(ring)
            else:
                lon, lat = 0.0, 0.0
            from shapely.geometry import shape as _shape
            try:
                wkt = _shape(geom).wkt
            except Exception:
                wkt = None
            villages.append(Village(
                village_id=str(props.get("village_id", i)),
                name=str(props.get("name", f"village_{i}")),
                centroid_lat=lat,
                centroid_lon=lon,
                population=int(props.get("population", 0)),
                area_m2=float(props.get("area_m2", 0.0)),
                admin_level=int(props.get("admin_level", 9)),
                geometry_wkt=wkt,
            ))
        return villages

    def _save_shelters_cache(self, path: Path, shelters: List[Shelter]):
        from shapely.wkt import loads as _wkt_loads
        from shapely.geometry import mapping as _mapping
        features = []
        for s in shelters:
            if s.geometry_wkt:
                try:
                    geom = _mapping(_wkt_loads(s.geometry_wkt))
                except Exception:
                    geom = {"type": "Point", "coordinates": [s.centroid_lon, s.centroid_lat]}
            else:
                geom = {"type": "Point", "coordinates": [s.centroid_lon, s.centroid_lat]}
            features.append({
                "type": "Feature",
                "properties": {
                    "shelter_id":   s.shelter_id,
                    "name":         s.name,
                    "capacity":     s.capacity,
                    "area_m2":      s.area_m2,
                    "shelter_type": s.shelter_type,
                },
                "geometry": geom,
            })
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    def _load_shelters_cache(self, path) -> List[Shelter]:
        with open(path) as f:
            data = json.load(f)
        shelters = []
        for i, feat in enumerate(data.get("features", [])):
            props = feat.get("properties", {})
            geom  = feat.get("geometry", {})
            gtype = geom.get("type", "Point")
            coords = geom.get("coordinates", [0, 0])
            if gtype == "Point":
                lon, lat = float(coords[0]), float(coords[1])
            elif gtype == "Polygon":
                ring = coords[0]
                lon = sum(c[0] for c in ring) / len(ring)
                lat = sum(c[1] for c in ring) / len(ring)
            elif gtype == "MultiPolygon":
                ring = max(coords, key=lambda p: len(p[0]))[0]
                lon = sum(c[0] for c in ring) / len(ring)
                lat = sum(c[1] for c in ring) / len(ring)
            else:
                lon, lat = 0.0, 0.0
            from shapely.geometry import shape as _shape
            try:
                wkt = _shape(geom).wkt
            except Exception:
                wkt = None
            shelters.append(Shelter(
                shelter_id=str(props.get("shelter_id", i)),
                name=str(props.get("name", f"shelter_{i}")),
                centroid_lat=lat,
                centroid_lon=lon,
                capacity=int(props.get("capacity", 0)),
                shelter_type=str(props.get("shelter_type", "shelter")),
                area_m2=float(props.get("area_m2", 0.0)),
                geometry_wkt=wkt,
            ))
        return shelters

    # ------------------------------------------------------------------ #
    # Legacy POI CSV format (RespondOR v1 compatibility)
    # Format: name,type,latitude,longitude[,node_id[,type_node_id]]
    # type values: village, shelter, depot, airport, warehouse, ...
    # ------------------------------------------------------------------ #

    # POI types that map to Village
    _VILLAGE_TYPES = {"village", "hamlet", "suburb", "neighbourhood"}
    # POI types that map to Shelter (everything else is treated as shelter)
    _SHELTER_TYPES = {
        "shelter", "depot", "airport", "warehouse",
        "deployment", "assembly_point", "hospital", "school",
    }

    def load_pois_from_csv(
        self,
        path: str,
    ) -> "Tuple[List[Village], List[Shelter]]":
        """
        Load villages and shelters from a legacy POI CSV file.

        Expected format (no header):
            name,type,latitude,longitude[,node_id[,...]]

        type = 'village' → Village
        type = anything else (shelter/depot/airport/…) → Shelter

        Supports both headered and headerless files.
        """
        import pandas as pd

        df = pd.read_csv(path, header=None)
        # Auto-detect header: if first row's 2nd column is a known type, no header
        first_val = str(df.iloc[0, 1]).strip().lower()
        known_types = self._VILLAGE_TYPES | self._SHELTER_TYPES
        if first_val not in known_types:
            # First row looks like a header — re-read with header
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
        else:
            df.columns = (["name", "type", "latitude", "longitude"]
                          + [f"extra_{i}" for i in range(len(df.columns) - 4)])

        villages: List[Village] = []
        shelters: List[Shelter] = []

        cols = set(df.columns)

        for i, row in df.iterrows():
            name     = str(row["name"]).strip()
            poi_type = str(row["type"]).strip().lower()
            lat      = float(row["latitude"])
            lon      = float(row["longitude"])
            area_m2  = float(row["area_m2"]) if "area_m2" in cols else 0.0

            if poi_type in self._VILLAGE_TYPES:
                pop = 0
                if "population" in cols:
                    try: pop = int(float(row["population"]))
                    except (ValueError, TypeError): pass
                if pop == 0 and "value" in cols:
                    try: pop = int(float(row["value"]))
                    except (ValueError, TypeError): pass
                vid = str(row["id"]).strip() if "id" in cols else f"poi_v_{i}"
                villages.append(Village(
                    village_id=vid,
                    name=name,
                    centroid_lat=lat,
                    centroid_lon=lon,
                    population=pop,
                    area_m2=area_m2,
                ))
            else:
                cap = 0
                if "capacity" in cols:
                    try: cap = int(float(row["capacity"]))
                    except (ValueError, TypeError): pass
                if cap == 0 and "value" in cols:
                    try: cap = int(float(row["value"]))
                    except (ValueError, TypeError): pass
                sid = str(row["id"]).strip() if "id" in cols else f"poi_s_{i}"
                shelters.append(Shelter(
                    shelter_id=sid,
                    name=name,
                    centroid_lat=lat,
                    centroid_lon=lon,
                    capacity=cap,
                    shelter_type=poi_type,
                    area_m2=area_m2,
                ))

        logger.info(
            f"Loaded POI CSV: {len(villages)} villages, {len(shelters)} shelters from {path}"
        )
        return villages, shelters

    def export_to_pycgr(
        self,
        nodes: "List[NetworkNode]",
        edges: "List[NetworkEdge]",
        path: str,
    ) -> None:
        """
        Export road network to PYCGR text format (RespondOR v1 / OsmToRoadGraph).

        Format:
          # comment lines
          n_nodes
          n_edges
          node_id lat lon          (one per line)
          src tgt length highway max_speed bidirectional   (one per line)
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# Road Graph File — exported by RespondOR-EvacuationRoute\n")
            f.write(f"# nodes: {len(nodes)}  edges: {len(edges)}\n")
            f.write(f"{len(nodes)}\n")
            f.write(f"{len(edges)}\n")
            for n in nodes:
                f.write(f"{n.node_id} {n.lat} {n.lon}\n")
            for e in edges:
                bidir = 1 if e.bidirectional else 0
                spd = int(e.max_speed_kmh) if e.max_speed_kmh == e.max_speed_kmh else 30
                f.write(
                    f"{e.source_id} {e.target_id} "
                    f"{e.length_m:.2f} {e.highway_type} {spd} {bidir}\n"
                )
        logger.info(f"Exported {len(nodes)} nodes, {len(edges)} edges to {path}")

    def export_pois_to_csv(
        self,
        villages: "List[Village]",
        shelters: "List[Shelter]",
        path: str,
    ) -> None:
        """
        Export villages and shelters to legacy POI CSV format.

        Output: name,type,latitude,longitude  (no header, no node_id column)
        """
        import pandas as pd

        rows = (
            [{"name": v.name, "type": "village",
              "latitude": v.centroid_lat, "longitude": v.centroid_lon}
             for v in villages]
            + [{"name": s.name, "type": s.shelter_type or "shelter",
                "latitude": s.centroid_lat, "longitude": s.centroid_lon}
               for s in shelters]
        )
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, header=False)
        logger.info(f"Exported {len(villages)} villages + {len(shelters)} shelters to {path}")
