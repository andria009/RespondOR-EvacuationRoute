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
        use_cache: bool = True,
    ) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """
        Extract road network for the region.
        Returns (nodes, edges).
        """
        bbox = region.to_bbox()
        cache_key = self._cache_key("network", bbox, network_type)
        cache_path = self.cache_dir / f"{cache_key}.json"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached network from {cache_path}")
            return self._load_network_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        logger.info(f"Extracting road network for bbox {bbox}")
        south, west, north, east = bbox

        try:
            # osmnx 2.x: bbox = (left, bottom, right, top) = (west, south, east, north)
            G = ox.graph_from_bbox(
                bbox=(west, south, east, north),
                network_type=network_type,
                retain_all=True,
                simplify=True,
            )
        except Exception as e:
            logger.error(f"OSM network extraction failed: {e}")
            raise

        nodes, edges = self._oxgraph_to_models(G)

        # Cache result
        self._save_network_cache(cache_path, nodes, edges)
        logger.info(f"Extracted {len(nodes)} nodes, {len(edges)} edges")
        return nodes, edges

    def extract_villages(
        self,
        region: RegionOfInterest,
        admin_level: int = 9,
        use_cache: bool = True,
    ) -> List[Village]:
        """Extract village/population area polygons."""
        bbox = region.to_bbox()
        cache_key = self._cache_key("villages", bbox, str(admin_level))
        cache_path = self.cache_dir / f"{cache_key}.geojson"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached villages from {cache_path}")
            return self._load_villages_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        south, west, north, east = bbox
        # osmnx 2.x bbox = (left, bottom, right, top) = (west, south, east, north)
        ox_bbox = (west, south, east, north)

        villages = []
        try:
            # Admin boundaries
            gdf = ox.features_from_bbox(
                bbox=ox_bbox,
                tags={"admin_level": str(admin_level), "boundary": "administrative"}
            )
            villages += self._gdf_to_villages(gdf)
        except Exception as e:
            logger.warning(f"Admin boundary extraction failed: {e}")

        try:
            # Places
            gdf_places = ox.features_from_bbox(
                bbox=ox_bbox,
                tags={"place": ["village", "hamlet", "suburb", "neighbourhood"]}
            )
            villages += self._gdf_to_villages(gdf_places)
        except Exception as e:
            logger.warning(f"Place extraction failed: {e}")

        villages = self._deduplicate_villages(villages)
        self._save_villages_cache(cache_path, villages)
        logger.info(f"Extracted {len(villages)} villages")
        return villages

    def extract_shelters(
        self,
        region: RegionOfInterest,
        shelter_tags: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> List[Shelter]:
        """Extract candidate shelter locations."""
        bbox = region.to_bbox()
        cache_key = self._cache_key("shelters", bbox, "default")
        cache_path = self.cache_dir / f"{cache_key}.geojson"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached shelters from {cache_path}")
            return self._load_shelters_cache(cache_path)

        if not HAS_OSMNX:
            raise RuntimeError("osmnx required for live OSM extraction")

        if shelter_tags is None:
            shelter_tags = {
                "emergency": ["assembly_point", "shelter"],
                "amenity": ["community_centre", "hospital", "clinic",
                            "place_of_worship", "school", "university"],
                "building": ["public", "civic", "government"]
            }

        south, west, north, east = bbox
        # osmnx 2.x bbox = (left, bottom, right, top) = (west, south, east, north)
        ox_bbox = (west, south, east, north)
        shelters = []

        for tag_key, tag_values in shelter_tags.items():
            for val in (tag_values if isinstance(tag_values, list) else [tag_values]):
                try:
                    gdf = ox.features_from_bbox(
                        bbox=ox_bbox,
                        tags={tag_key: val}
                    )
                    shelters += self._gdf_to_shelters(gdf, shelter_type=val)
                except Exception as e:
                    logger.debug(f"No {tag_key}={val} found: {e}")

        shelters = self._deduplicate_shelters(shelters)
        self._save_shelters_cache(cache_path, shelters)
        logger.info(f"Extracted {len(shelters)} shelter candidates")
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

    def _oxgraph_to_models(self, G) -> Tuple[List[NetworkNode], List[NetworkEdge]]:
        """Convert osmnx graph to NetworkNode/NetworkEdge lists."""
        nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
        node_map = {}  # osm_id -> sequential int id
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
        for (u, v, _), row in edges_gdf.iterrows():
            if u not in node_map or v not in node_map:
                continue
            src = node_map[u]
            tgt = node_map[v]
            key = (min(src, tgt), max(src, tgt))
            if key in seen:
                continue
            seen.add(key)

            highway = row.get("highway", "residential")
            if isinstance(highway, list):
                highway = highway[0]

            speed_raw = row.get("maxspeed", None)
            if isinstance(speed_raw, list):
                speed_raw = speed_raw[0]
            try:
                speed = float(str(speed_raw).replace(" mph", "").replace(" kmh", "").strip())
                # float("nan") succeeds in Python — guard against NaN/invalid values
                if speed != speed or speed <= 0:
                    raise ValueError("invalid speed")
            except (TypeError, ValueError):
                speed = SPEED_BY_HIGHWAY.get(str(highway), 30.0)

            lanes_raw = row.get("lanes", 1)
            try:
                lanes = int(lanes_raw) if lanes_raw else 1
            except (TypeError, ValueError):
                lanes = 1

            edges.append(NetworkEdge(
                source_id=src,
                target_id=tgt,
                length_m=float(row.get("length", 0.0)),
                highway_type=str(highway),
                max_speed_kmh=speed,
                bidirectional=not row.get("oneway", False),
                lanes=lanes,
            ))

        return nodes, edges

    def _gdf_to_villages(self, gdf: gpd.GeoDataFrame) -> List[Village]:
        # Reproject to UTM zone 49S (EPSG:32749) for accurate m² area on Java/Sumatra
        try:
            gdf_utm = gdf.to_crs("EPSG:32749")
        except Exception:
            gdf_utm = gdf

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
            # Sanity cap: max village area = 100 km² (larger = likely wrong admin level)
            area_m2 = min(area_m2, 100_000_000.0)
            name = str(row.get("name", row.get("name:en", f"village_{idx}")))
            villages.append(Village(
                village_id=str(idx),
                name=name,
                centroid_lat=float(centroid.y),
                centroid_lon=float(centroid.x),
                population=0,       # filled by population loader
                area_m2=area_m2,
                geometry_wkt=geom.wkt,
            ))
        return villages

    def _gdf_to_shelters(self, gdf: gpd.GeoDataFrame, shelter_type: str) -> List[Shelter]:
        shelters = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            centroid = geom.centroid if hasattr(geom, 'centroid') else geom
            area = geom.area if hasattr(geom, 'area') else 0.0
            name = str(row.get("name", row.get("name:en", f"shelter_{idx}")))
            shelters.append(Shelter(
                shelter_id=str(idx),
                name=name,
                centroid_lat=float(centroid.y),
                centroid_lon=float(centroid.x),
                capacity=0,         # filled by capacity estimator
                shelter_type=shelter_type,
                area_m2=float(area) * 1e10,
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
        features = []
        for v in villages:
            features.append({
                "type": "Feature",
                "properties": {
                    "village_id": v.village_id, "name": v.name,
                    "population": v.population, "area_m2": v.area_m2,
                },
                "geometry": {"type": "Point", "coordinates": [v.centroid_lon, v.centroid_lat]}
            })
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    def _load_villages_cache(self, path) -> List[Village]:
        with open(path) as f:
            data = json.load(f)
        villages = []
        for i, feat in enumerate(data.get("features", [])):
            props = feat.get("properties", {})
            coords = feat["geometry"]["coordinates"]
            villages.append(Village(
                village_id=str(props.get("village_id", i)),
                name=str(props.get("name", f"village_{i}")),
                centroid_lat=float(coords[1]),
                centroid_lon=float(coords[0]),
                population=int(props.get("population", 0)),
                area_m2=float(props.get("area_m2", 0.0)),
            ))
        return villages

    def _save_shelters_cache(self, path: Path, shelters: List[Shelter]):
        features = []
        for s in shelters:
            features.append({
                "type": "Feature",
                "properties": {
                    "shelter_id": s.shelter_id, "name": s.name,
                    "capacity": s.capacity, "area_m2": s.area_m2,
                    "shelter_type": s.shelter_type,
                },
                "geometry": {"type": "Point", "coordinates": [s.centroid_lon, s.centroid_lat]}
            })
        with open(path, "w") as f:
            json.dump({"type": "FeatureCollection", "features": features}, f)

    def _load_shelters_cache(self, path) -> List[Shelter]:
        with open(path) as f:
            data = json.load(f)
        shelters = []
        for i, feat in enumerate(data.get("features", [])):
            props = feat.get("properties", {})
            coords = feat["geometry"]["coordinates"]
            shelters.append(Shelter(
                shelter_id=str(props.get("shelter_id", i)),
                name=str(props.get("name", f"shelter_{i}")),
                centroid_lat=float(coords[1]),
                centroid_lon=float(coords[0]),
                capacity=int(props.get("capacity", 0)),
                shelter_type=str(props.get("shelter_type", "shelter")),
                area_m2=float(props.get("area_m2", 0.0)),
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

        for i, row in df.iterrows():
            name = str(row["name"]).strip()
            poi_type = str(row["type"]).strip().lower()
            lat = float(row["latitude"])
            lon = float(row["longitude"])

            if poi_type in self._VILLAGE_TYPES:
                villages.append(Village(
                    village_id=f"poi_v_{i}",
                    name=name,
                    centroid_lat=lat,
                    centroid_lon=lon,
                    population=0,       # filled by PopulationLoader
                    area_m2=0.0,
                ))
            else:
                shelters.append(Shelter(
                    shelter_id=f"poi_s_{i}",
                    name=name,
                    centroid_lat=lat,
                    centroid_lon=lon,
                    capacity=0,         # filled by ShelterCapacityLoader
                    shelter_type=poi_type,
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
