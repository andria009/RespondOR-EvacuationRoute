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

        _prev_oxcache = ox.settings.use_cache
        if not use_cache:
            ox.settings.use_cache = False
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
        finally:
            ox.settings.use_cache = _prev_oxcache

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

    def extract_road_network_gdf(
        self,
        region: RegionOfInterest,
        network_type: str = "all",
        use_cache: bool = True,
    ) -> Tuple["gpd.GeoDataFrame", "gpd.GeoDataFrame"]:
        """
        Return raw (nodes_gdf, edges_gdf) from osmnx without model conversion.
        Useful for shapefile export or other consumers that need line geometries.
        Uses osmnx's built-in HTTP cache under self.cache_dir/http.
        """
        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        ox.settings.use_cache = use_cache
        ox.settings.cache_folder = str(self.cache_dir / "http")

        south, west, north, east = region.to_bbox()
        logger.info(f"Extracting road network GDF for bbox ({south:.4f},{west:.4f},{north:.4f},{east:.4f})")
        G = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type=network_type,
            retain_all=True,
            simplify=True,
        )
        return ox.graph_to_gdfs(G)

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
        cluster_max_area_km2: float = 25.0,
        persons_per_dwelling: float = 4.0,
        building_persons: Optional[dict] = None,
        fill_uncovered_l9: bool = False,
    ) -> List[Village]:
        """
        Extract village population areas from one or more OSM sources.

        Sources are processed in order; each adds only settlements whose
        centroid is not already covered by a polygon from a previous source.

        Available sources (set via ``sources`` parameter):
          ``wilayah_db``
              Official Indonesian administrative boundaries from the local
              PostGIS wilayah database (cahayadsn-wilayah* data).
              Requires Docker Compose to be running (see docker-compose.yml).
              Provides the most accurate official boundaries and names.
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
        fill_suffix = "_fill" if fill_uncovered_l9 else ""
        variant = f"src{src_key}_al{'_'.join(str(l) for l in admin_levels)}_d{int(population_density_per_km2)}_eps{int(cluster_eps_m)}{fill_suffix}"
        cache_key = self._cache_key("villages", bbox, variant)
        cache_path = self.cache_dir / f"{cache_key}.geojson"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached villages from {cache_path}")
            return self._load_villages_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        south, west, north, east = bbox
        ox_bbox = (west, south, east, north)  # osmnx 2.x: (left, bottom, right, top)

        # ── Incremental / resume support ──────────────────────────────────
        # Partial cache stores raw villages per completed source.
        # If killed mid-way, we resume from where we left off.
        partial_path = cache_path.with_suffix(".partial.json")
        completed_sources: set = set()
        all_villages: List[Village] = []

        if use_cache and partial_path.exists():
            try:
                with open(partial_path) as f:
                    partial = json.load(f)
                completed_sources = set(partial.get("completed_sources", []))
                all_villages = self._load_villages_from_records(partial.get("villages", []))
                logger.info(
                    f"Resuming village extraction: {completed_sources} already done, "
                    f"{len(all_villages)} villages loaded from partial cache"
                )
            except Exception as e:
                logger.warning(f"Could not load partial village cache ({e}) — starting fresh")
                completed_sources = set()
                all_villages = []

        for source in sources:
            if source in completed_sources:
                logger.info(f"village source '{source}': (cached — skipping)")
                continue

            if source == "wilayah_db":
                new = self._villages_from_wilayah_db(
                    bbox, admin_levels, population_density_per_km2, max_population_per_village
                )
            elif source == "admin_boundary":
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
                    max_population_per_village,
                    cluster_max_area_km2,
                    use_cache=use_cache,
                )
            else:
                logger.warning(f"Unknown village source '{source}' — skipping")
                completed_sources.add(source)
                continue

            if not new:
                logger.info(f"village source '{source}': no results")
            else:
                added = self._add_uncovered_villages(new, all_villages)
                logger.info(f"village source '{source}': {len(new)} found, {len(added)} added (not already covered)")
                all_villages.extend(added)

            completed_sources.add(source)
            self._save_village_partial(partial_path, completed_sources, all_villages)

        if not all_villages:
            logger.warning(f"No villages found from any source ({sources}). Returning empty list.")
            if partial_path.exists():
                partial_path.unlink()
            return []

        all_villages = self._deduplicate_villages(all_villages)

        if fill_uncovered_l9:
            artificial = self._fill_uncovered_l9(
                bbox=bbox,
                existing_villages=all_villages,
                house_persons=float((building_persons or {}).get("house", persons_per_dwelling)),
            )
            if artificial:
                all_villages.extend(artificial)

        self._save_villages_cache(cache_path, all_villages)
        # Remove partial cache — full cache is now written
        if partial_path.exists():
            partial_path.unlink()
        logger.info(
            f"Extracted {len(all_villages)} villages total "
            f"(sources={sources}, density={population_density_per_km2} p/km², "
            f"total_pop={sum(v.population for v in all_villages):,})"
        )
        return all_villages

    # ------------------------------------------------------------------
    # Village source implementations
    # ------------------------------------------------------------------

    def _villages_from_wilayah_db(
        self,
        bbox: tuple,
        admin_levels: List[int],
        population_density_per_km2: float,
        max_population_per_village: int,
    ) -> List[Village]:
        """
        Load official Indonesian administrative boundaries from the local
        PostGIS wilayah database.  Returns an empty list (with a warning) if
        the database is unreachable.

        Levels are queried finest-first (9→8→7): each coarser level only
        fills areas not already covered by a finer one, so kabupaten
        boundaries never overlap existing kelurahan polygons.

        bbox: (south, west, north, east)
        """
        from src.data.wilayah_loader import WilayahLoader
        try:
            loader = WilayahLoader()
            south, west, north, east = bbox
            combined: List[Village] = []
            # Sort finest to coarsest (highest admin_level number = finest)
            for level in sorted(admin_levels, reverse=True):
                candidates = loader.load_villages(
                    bbox=(south, west, north, east),
                    admin_levels=[level],
                    population_density_per_km2=population_density_per_km2,
                    max_population_per_village=max_population_per_village,
                )
                added = self._add_uncovered_villages(candidates, combined)
                if added:
                    logger.info(
                        f"wilayah_db L{level}: {len(candidates)} found, "
                        f"{len(added)} added"
                    )
                combined.extend(added)
            loader.close()
            return combined
        except Exception as exc:
            logger.warning(
                f"wilayah_db source failed (is Docker running?): {exc} — skipping"
            )
            return []

    def _fill_uncovered_l9(
        self,
        bbox: tuple,
        existing_villages: List[Village],
        house_persons: float = 4.5,
        min_radius_m: float = 75.0,
        max_radius_m: float = 200.0,
    ) -> List[Village]:
        """
        For each L9 kelurahan that contains no existing village centroid,
        create one synthetic circular cluster at the kelurahan centroid.

        Radius = equivalent circle radius of the smallest existing cluster,
                 clamped to [min_radius_m, max_radius_m].
        Population = max(1, int(house_persons))  (one household unit).
        Requires the wilayah PostGIS DB to be running.
        """
        import math
        from pyproj import Transformer
        from shapely.ops import transform as shp_transform
        from shapely.strtree import STRtree
        from shapely.wkt import loads as wkt_loads
        from src.data.wilayah_loader import WilayahLoader

        try:
            with WilayahLoader() as loader:
                l9_villages = loader.load_villages(bbox=bbox, admin_levels=[9])
        except Exception as exc:
            logger.warning(f"fill_uncovered_l9: wilayah DB unavailable: {exc}")
            return []

        if not l9_villages:
            return []

        # Compute fill radius from smallest real cluster
        real_areas = [v.area_m2 for v in existing_villages if v.area_m2 > 0]
        if real_areas:
            fill_radius_m = math.sqrt(min(real_areas) / math.pi)
        else:
            fill_radius_m = min_radius_m
        fill_radius_m = max(min_radius_m, min(fill_radius_m, max_radius_m))

        # Spatial join: find which L9s have an existing village inside them
        l9_with_geom = [
            (v, wkt_loads(v.geometry_wkt))
            for v in l9_villages if v.geometry_wkt
        ]
        if not l9_with_geom:
            return []

        tree = STRtree([g for _, g in l9_with_geom])
        covered_ids: set = set()
        for ev in existing_villages:
            pt = Point(ev.centroid_lon, ev.centroid_lat)
            for idx in tree.query(pt):
                lv, g = l9_with_geom[idx]
                if g.contains(pt):
                    covered_ids.add(lv.village_id)
                    break

        # Build synthetic circular Village for each uncovered L9
        pop = max(1, int(house_persons))
        area_m2 = math.pi * fill_radius_m ** 2
        artificial: List[Village] = []

        for lv in l9_villages:
            if lv.village_id in covered_ids:
                continue

            lat, lon = lv.centroid_lat, lv.centroid_lon
            zone = int((lon + 180) / 6) + 1
            epsg_utm = f"{'326' if lat >= 0 else '327'}{zone:02d}"
            to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy=True)
            to_wgs = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy=True)

            pt_utm = shp_transform(to_utm.transform, Point(lon, lat))
            buf_wgs = shp_transform(to_wgs.transform, pt_utm.buffer(fill_radius_m))

            artificial.append(Village(
                village_id=f"artificial_l9_{lv.village_id}",
                name=lv.name,
                centroid_lat=lat,
                centroid_lon=lon,
                population=pop,
                area_m2=area_m2,
                admin_level=9,
                geometry_wkt=buf_wgs.wkt,
            ))

        logger.info(
            f"fill_uncovered_l9: {len(artificial)} synthetic clusters added "
            f"(radius={fill_radius_m:.0f} m, pop={pop} each)"
        )
        return artificial

    def _villages_from_admin_boundary(
        self,
        ox_bbox: tuple,
        admin_levels: List[int],
        population_density_per_km2: float,
        max_population_per_village: int,
    ) -> List[Village]:
        """
        Extract villages from OSM boundary=administrative polygons.

        All admin levels are queried independently.  Results are merged
        from finest to coarsest: each coarser level only fills areas not
        already covered by a finer level (centroid-outside check).
        This ensures desa (L9) polygons are kept wherever they exist, with
        kecamatan (L8) or kabupaten (L7) filling remaining gaps.

        No minimum area filter is applied — any valid polygon is accepted.
        OSM boundary polygons, however small, represent a real administrative
        unit that a community identifies with and should be routed independently.
        """
        from shapely.wkt import loads as _wkt_loads
        from shapely.ops import unary_union
        from shapely.geometry import Point as _Pt

        all_villages: List[Village] = []

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
                logger.info(f"admin_level={level}: no polygon boundaries found")
                continue

            new_villages = self._gdf_to_villages(gdf_poly, admin_level=level)
            self._assign_population_from_area(new_villages, population_density_per_km2, max_population_per_village)

            # Only add villages whose centroid is not already covered by a finer level
            if all_villages:
                polys = []
                for v in all_villages:
                    if v.geometry_wkt:
                        try:
                            polys.append(_wkt_loads(v.geometry_wkt))
                        except Exception:
                            pass
                covered = unary_union(polys) if polys else None
                added = []
                for v in new_villages:
                    if covered is None or not covered.contains(_Pt(v.centroid_lon, v.centroid_lat)):
                        added.append(v)
            else:
                added = new_villages

            logger.info(f"admin_level={level}: {len(new_villages)} polygons found, {len(added)} added")
            all_villages.extend(added)

        return all_villages

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

        # Deduplicate place-nodes by area: sort smallest first, reject any candidate
        # whose polygon overlaps an already-accepted polygon by >overlap_threshold.
        # This means a hamlet circle inside a town circle wins — keeping the finest
        # granularity available and discarding the larger synthetic area.
        villages.sort(key=lambda v: v.area_m2)
        # Deduplicate place-nodes by clipping each candidate to its uncovered area.
        # Candidates are processed smallest-first so finer-grained settlements
        # take priority. Each candidate's circle is clipped against the union of
        # already-accepted polygons; the clipped polygon and proportionally scaled
        # population are stored. A candidate is dropped only if less than 10% of
        # its circle remains uncovered after clipping.
        villages.sort(key=lambda v: v.area_m2)
        accepted: List[Village] = []
        from shapely.wkt import loads as _wkt_loads2
        from shapely.ops import unary_union as _unary_union2
        accepted_polys = []
        min_remaining_fraction = 0.10  # drop if <10% of circle is uncovered

        for v in villages:
            if v.geometry_wkt:
                try:
                    cand_poly = _wkt_loads2(v.geometry_wkt)
                    orig_area = cand_poly.area
                    if accepted_polys and orig_area > 0:
                        covered = _unary_union2(accepted_polys)
                        clipped  = cand_poly.difference(covered)
                        remaining_frac = clipped.area / orig_area
                        if remaining_frac < min_remaining_fraction:
                            continue  # almost entirely covered — skip
                        if remaining_frac < 0.99:
                            # Partially clipped — update geometry and scale population
                            v.geometry_wkt = clipped.wkt
                            v.area_m2      = float(clipped.area)
                            v.population   = max(1, int(v.population * remaining_frac))
                            accepted_polys.append(clipped)
                        else:
                            accepted_polys.append(cand_poly)
                    else:
                        accepted_polys.append(cand_poly)
                    accepted.append(v)
                    continue
                except Exception:
                    pass
            accepted.append(v)  # no geometry — keep as-is

        if villages:
            radii = sorted({
                float((place_settings.get(str(r.get("place", "") or ""), {})).get(
                    "radius_m", default_radius_m))
                for _, r in gdf_pts.iterrows()
            })
            logger.info(
                f"place-nodes: {len(villages)} raw → {len(accepted)} after clip-dedup "
                f"(radius {radii[0]:.0f}–{radii[-1]:.0f} m, per-tag density)"
            )
        return accepted

    def _villages_from_building_clusters(
        self,
        ox_bbox: tuple,
        eps_m: float,
        min_buildings: int,
        persons_per_dwelling: float,
        building_persons: dict,
        max_population_per_village: int,
        max_area_km2: float = 25.0,
        use_cache: bool = True,
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
        logger.info(f"building_clusters: querying OSM building footprints …")
        _prev_oxcache = ox.settings.use_cache
        if not use_cache:
            ox.settings.use_cache = False
        try:
            gdf = ox.features_from_bbox(
                bbox=ox_bbox,
                tags={"building": True},
            )
        except Exception as e:
            logger.warning(f"building query failed: {e}")
            return []
        finally:
            ox.settings.use_cache = _prev_oxcache

        if gdf.empty:
            logger.info("building_clusters: no buildings found in bbox")
            return []

        logger.info(f"building_clusters: {len(gdf):,} buildings fetched — projecting to UTM …")

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
        except ImportError:
            logger.warning("building_clusters: scikit-learn not installed — skipping")
            return []

        from shapely.geometry import Point as _Pt
        from shapely.ops import transform as _transform, unary_union as _unary_union

        def _cluster_polygon(c_pts: list, eps: float):
            """
            Build a non-overlapping polygon for a cluster by unioning per-building
            buffers of radius eps/2.

            Why eps/2: DBSCAN guarantees any two buildings in *different* clusters
            are more than eps apart. Buffering each building by eps/2 therefore
            produces buffers that cannot overlap across cluster boundaries —
            non-overlap is guaranteed by construction, not by post-processing.
            The union of same-cluster buffers produces a connected shape that
            follows the actual settlement footprint rather than a convex envelope.
            """
            buf_r = eps / 2.0
            polys = [_Pt(x, y).buffer(buf_r) for x, y in c_pts]
            return _unary_union(polys)

        def _clusters_to_villages(
            pts_utm: list,
            row_indices: list,
            eps: float,
            id_prefix: str,
        ) -> List[Village]:
            """
            Run DBSCAN(eps, min_samples=1) on pts_utm.
            Clusters exceeding max_area_km2 are recursively re-clustered at
            eps/2, down to a minimum eps of 10 m (single-building clusters).
            Returns a flat list of Village objects.
            """
            labels = DBSCAN(eps=eps, min_samples=1).fit_predict(np.array(pts_utm))
            result: List[Village] = []

            for cid in set(labels):
                mask = [l == cid for l in labels]
                c_pts  = [pts_utm[i]    for i, m in enumerate(mask) if m]
                c_rows = [row_indices[i] for i, m in enumerate(mask) if m]

                poly_utm = _cluster_polygon(c_pts, eps)
                area_m2  = float(poly_utm.area)

                # Recursively subdivide oversized clusters by halving eps
                if max_area_km2 > 0 and area_m2 > max_area_km2 * 1e6:
                    new_eps = eps / 2.0
                    if new_eps >= 10.0:
                        result.extend(_clusters_to_villages(
                            c_pts, c_rows, new_eps, f"{id_prefix}_{cid}"
                        ))
                        continue
                    # eps floor reached — accept as-is to avoid infinite recursion
                    logger.debug(
                        f"building_clusters: cluster {id_prefix}_{cid} "
                        f"area={area_m2/1e6:.1f} km² — eps floor reached, keeping"
                    )

                poly_wgs     = _transform(to_wgs, poly_utm)
                centroid_wgs = _transform(to_wgs, poly_utm.centroid)

                name_counts: dict = {}
                total_persons = 0.0
                for i in c_rows:
                    try:
                        row_wgs = gdf_wgs.iloc[i]
                        n = str(row_wgs.get("name", ""))
                        if n and n not in ("None", "nan"):
                            name_counts[n] = name_counts.get(n, 0) + 1
                        btype = str(row_wgs.get("building", "yes") or "yes").strip().lower()
                        total_persons += building_persons.get(btype, persons_per_dwelling)
                    except Exception:
                        total_persons += persons_per_dwelling

                name = max(name_counts, key=name_counts.get) if name_counts else f"cluster_{id_prefix}_{cid}"
                pop  = max(1, min(int(total_persons), max_population_per_village))

                result.append(Village(
                    village_id=f"bldg_cluster_{id_prefix}_{cid}",
                    name=name,
                    centroid_lat=float(centroid_wgs.y),
                    centroid_lon=float(centroid_wgs.x),
                    population=pop,
                    area_m2=area_m2,
                    admin_level=11,
                    geometry_wkt=poly_wgs.wkt,
                ))
            return result

        logger.info(f"building_clusters: running DBSCAN on {len(coords):,} building centroids "
                    f"(eps={eps_m} m, min_buildings={min_buildings}) …")
        all_indices = list(range(len(coords)))
        villages = _clusters_to_villages(coords, all_indices, eps_m, "0")

        logger.info(
            f"building_clusters: {len(villages)} clusters "
            f"(eps={eps_m} m, min_samples=1, recursive subdivision to 10 m, "
            f"persons_per_dwelling={persons_per_dwelling})"
        )
        return villages

    def _add_uncovered_villages(
        self,
        candidates: List[Village],
        existing: List[Village],
        overlap_threshold: float = 0.3,
    ) -> List[Village]:
        """
        Return candidates that do NOT significantly overlap any existing village polygon.

        A candidate is rejected if the intersection of its polygon with the union
        of existing polygons exceeds ``overlap_threshold`` × candidate area.
        Default threshold 0.3 means a candidate is dropped if more than 30% of
        its area is already covered — preventing double-counting while still
        allowing place-node circles that extend beyond admin-boundary edges.

        Falls back to centroid-in-polygon for candidates without geometry_wkt,
        preserving original behaviour for node-only records.

        If existing is empty, all candidates are returned unchanged.
        """
        if not existing:
            return candidates

        from shapely.wkt import loads as _wkt_loads
        from shapely.geometry import Point as _Pt
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
            if c.geometry_wkt:
                try:
                    cand_poly = _wkt_loads(c.geometry_wkt)
                    inter_area = cand_poly.intersection(covered).area
                    cand_area  = cand_poly.area
                    if cand_area > 0 and (inter_area / cand_area) > overlap_threshold:
                        continue   # too much overlap — already covered
                except Exception:
                    pass   # fall through to centroid check
            # Fallback: centroid check for candidates without valid geometry
            if covered.contains(_Pt(c.centroid_lon, c.centroid_lat)):
                continue
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
        cluster_eps_m: float = 250.0,
        cluster_min_shelters: int = 1,
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
        # Cache key encodes tag fingerprint + area + cluster params
        import hashlib as _hl
        tag_sig = _hl.md5(str(sorted(str(shelter_tags).split())).encode()).hexdigest()[:8]
        variant = f"poly_a{int(min_area_m2)}_t{tag_sig}_eps{int(cluster_eps_m)}_min{cluster_min_shelters}"
        cache_key = self._cache_key("shelters", bbox, variant)
        cache_path = self.cache_dir / f"{cache_key}.geojson"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached shelters from {cache_path}")
            return self._load_shelters_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        south, west, north, east = bbox
        ox_bbox = (west, south, east, north)

        # ── Incremental / resume support ──────────────────────────────────
        # Partial cache stores raw (pre-cluster) shelters per completed tag.
        # If killed mid-way, we resume from where we left off.
        partial_path = cache_path.with_suffix(".partial.json")
        completed_tags: set = set()
        shelters: List[Shelter] = []

        if use_cache and partial_path.exists():
            try:
                with open(partial_path) as f:
                    partial = json.load(f)
                completed_tags = set(partial.get("completed_tags", []))
                shelters = self._load_shelters_from_records(partial.get("shelters", []))
                logger.info(
                    f"Resuming shelter extraction: {len(completed_tags)} tags already done, "
                    f"{len(shelters)} shelters loaded from partial cache"
                )
            except Exception as e:
                logger.warning(f"Could not load partial shelter cache ({e}) — starting fresh")
                completed_tags = set()
                shelters = []

        n_tags = sum(len(v) if isinstance(v, list) else 1 for v in shelter_tags.values())
        logger.info(f"Querying {n_tags} shelter tag combinations from OSM …")
        tag_idx = 0
        for tag_key, tag_values in shelter_tags.items():
            for val in (tag_values if isinstance(tag_values, list) else [tag_values]):
                tag_idx += 1
                tag_label = f"{tag_key}={val}"
                if tag_label in completed_tags:
                    logger.info(f"  [{tag_idx}/{n_tags}] {tag_label} … (cached)")
                    continue
                logger.info(f"  [{tag_idx}/{n_tags}] {tag_label} …")
                try:
                    gdf = ox.features_from_bbox(bbox=ox_bbox, tags={tag_key: val})
                except Exception as e:
                    logger.debug(f"No {tag_label} found: {e}")
                    completed_tags.add(tag_label)
                    self._save_shelter_partial(partial_path, completed_tags, shelters)
                    continue

                # Keep only closed polygons
                poly_mask = gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
                gdf_poly = gdf[poly_mask].copy()
                if gdf_poly.empty:
                    logger.debug(f"{tag_label}: no polygon features")
                    completed_tags.add(tag_label)
                    self._save_shelter_partial(partial_path, completed_tags, shelters)
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
                logger.info(f"  {tag_label}: {len(raw)} polygon shelters "
                            f"(≥{min_area_m2:.0f} m²)")
                shelters += raw
                completed_tags.add(tag_label)
                self._save_shelter_partial(partial_path, completed_tags, shelters)

        shelters = self._deduplicate_shelters(shelters)
        shelters = self._cluster_shelters(shelters, cluster_eps_m, cluster_min_shelters)
        self._save_shelters_cache(cache_path, shelters)
        # Remove partial cache — full cache is now written
        if partial_path.exists():
            partial_path.unlink()
        total_cap = sum(s.capacity for s in shelters)
        logger.info(
            f"Extracted {len(shelters)} shelter clusters "
            f"(min_area={min_area_m2:.0f} m², eps={cluster_eps_m:.0f} m, "
            f"total_capacity={total_cap:,})"
        )
        return shelters

    def _cluster_shelters(
        self,
        shelters: List[Shelter],
        eps_m: float,
        min_samples: int,
    ) -> List[Shelter]:
        """
        DBSCAN-cluster nearby shelters into a single shelter destination.
        Clustered shelters share a merged polygon (union), summed capacity,
        and centroid. Noise points (unclustered) are kept as-is.
        """
        if len(shelters) < 2:
            return shelters

        import math as _math
        import numpy as np
        from sklearn.cluster import DBSCAN
        from pyproj import Transformer
        from shapely.geometry import Point as _ShapelyPoint, Polygon as _ShapelyPolygon
        from shapely.ops import unary_union
        from shapely.wkt import loads as wkt_loads

        def _circle_wkt_utm(cx_utm: float, cy_utm: float, area_m2: float,
                             to_wgs: Transformer) -> str:
            """
            Create a circular polygon WKT in WGS84 centered at a UTM point,
            with area equivalent to area_m2.  Resolution = 64 segments.
            """
            radius_m = _math.sqrt(max(area_m2, 1.0) / _math.pi)
            circle_utm = _ShapelyPoint(cx_utm, cy_utm).buffer(radius_m, resolution=64)
            # Transform exterior ring from UTM → WGS84
            wgs_pts = [to_wgs.transform(x, y) for x, y in circle_utm.exterior.coords]
            # to_wgs returns (lon, lat); Shapely Polygon takes (x=lon, y=lat)
            return _ShapelyPolygon(wgs_pts).wkt

        # Project centroids to UTM for metric distance
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32749", always_xy=True)
        coords = np.array([
            transformer.transform(s.centroid_lon, s.centroid_lat)
            for s in shelters
        ])

        labels = DBSCAN(
            eps=eps_m, min_samples=min_samples, metric="euclidean"
        ).fit_predict(coords)

        to_wgs = Transformer.from_crs("EPSG:32749", "EPSG:4326", always_xy=True)
        clustered: List[Shelter] = []

        for cid in set(labels):
            members = [s for s, l in zip(shelters, labels) if l == cid]

            # Summed capacity and area (applies to both single and merged)
            total_cap = sum(s.capacity for s in members)
            total_area = sum(s.area_m2 for s in members)

            if len(members) == 1:
                s = members[0]
                # Circularize single-shelter geometry too — OSM buildings are rectangles
                cx_utm, cy_utm = transformer.transform(s.centroid_lon, s.centroid_lat)
                circle_wkt = _circle_wkt_utm(cx_utm, cy_utm, s.area_m2, to_wgs)
                s.geometry_wkt = circle_wkt
                clustered.append(s)
                continue

            # Merged cluster: centroid is mean of member centroids in UTM
            cx_utm = sum(coords[i][0] for i, l in enumerate(labels) if l == cid) / len(members)
            cy_utm = sum(coords[i][1] for i, l in enumerate(labels) if l == cid) / len(members)
            clon, clat = to_wgs.transform(cx_utm, cy_utm)

            # Circular geometry of total area
            circle_wkt = _circle_wkt_utm(cx_utm, cy_utm, total_area, to_wgs)

            # Representative name: most common non-empty name among members
            names = [s.name for s in members if s.name]
            name = max(set(names), key=names.count) if names else members[0].shelter_id

            # Representative type: most common type
            types = [s.shelter_type for s in members]
            shelter_type = max(set(types), key=types.count)

            clustered.append(Shelter(
                shelter_id=f"shelter_cluster_{cid}",
                name=name,
                centroid_lat=clat,
                centroid_lon=clon,
                capacity=total_cap,
                shelter_type=shelter_type,
                area_m2=total_area,
                geometry_wkt=circle_wkt,
            ))

        n_orig = len(shelters)
        n_clus = len(clustered)
        if n_clus < n_orig:
            logger.info(f"  Shelter clustering: {n_orig} → {n_clus} clusters (eps={eps_m:.0f} m)")
        return clustered

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

    def _village_to_record(self, v: Village) -> dict:
        """Serialize a Village to a plain dict for partial cache storage."""
        return {
            "village_id":   v.village_id,
            "name":         v.name,
            "centroid_lat": v.centroid_lat,
            "centroid_lon": v.centroid_lon,
            "population":   v.population,
            "area_m2":      v.area_m2,
            "admin_level":  v.admin_level,
            "geometry_wkt": v.geometry_wkt,
        }

    def _load_villages_from_records(self, records: list) -> List[Village]:
        """Deserialize Village list from plain dicts (partial cache)."""
        villages = []
        for i, r in enumerate(records):
            villages.append(Village(
                village_id=str(r.get("village_id", i)),
                name=str(r.get("name", f"village_{i}")),
                centroid_lat=float(r.get("centroid_lat", 0.0)),
                centroid_lon=float(r.get("centroid_lon", 0.0)),
                population=int(r.get("population", 0)),
                area_m2=float(r.get("area_m2", 0.0)),
                admin_level=int(r.get("admin_level", 9)),
                geometry_wkt=r.get("geometry_wkt"),
            ))
        return villages

    def _save_village_partial(self, path: Path, completed_sources: set, villages: List[Village]):
        """Save incremental village progress to a partial cache file."""
        try:
            with open(path, "w") as f:
                json.dump({
                    "completed_sources": sorted(completed_sources),
                    "villages": [self._village_to_record(v) for v in villages],
                }, f)
        except Exception as e:
            logger.warning(f"Could not write partial village cache: {e}")

    def _shelter_to_record(self, s: Shelter) -> dict:
        """Serialize a Shelter to a plain dict for partial cache storage."""
        return {
            "shelter_id":   s.shelter_id,
            "name":         s.name,
            "centroid_lat": s.centroid_lat,
            "centroid_lon": s.centroid_lon,
            "capacity":     s.capacity,
            "area_m2":      s.area_m2,
            "shelter_type": s.shelter_type,
            "geometry_wkt": s.geometry_wkt,
        }

    def _load_shelters_from_records(self, records: list) -> List[Shelter]:
        """Deserialize Shelter list from plain dicts (partial cache)."""
        shelters = []
        for i, r in enumerate(records):
            shelters.append(Shelter(
                shelter_id=str(r.get("shelter_id", i)),
                name=str(r.get("name", f"shelter_{i}")),
                centroid_lat=float(r.get("centroid_lat", 0.0)),
                centroid_lon=float(r.get("centroid_lon", 0.0)),
                capacity=int(r.get("capacity", 0)),
                shelter_type=str(r.get("shelter_type", "shelter")),
                area_m2=float(r.get("area_m2", 0.0)),
                geometry_wkt=r.get("geometry_wkt"),
            ))
        return shelters

    def _save_shelter_partial(self, path: Path, completed_tags: set, shelters: List[Shelter]):
        """Save incremental shelter progress to a partial cache file."""
        try:
            with open(path, "w") as f:
                json.dump({
                    "completed_tags": sorted(completed_tags),
                    "shelters": [self._shelter_to_record(s) for s in shelters],
                }, f)
        except Exception as e:
            logger.warning(f"Could not write partial shelter cache: {e}")

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

