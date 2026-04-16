"""
WilayahLoader — query the wilayah database and return Village objects
with official Indonesian administrative boundaries.

Supports two backends (auto-detected, or set explicitly):

  SQLite (portable, no server):
    WilayahLoader(sqlite_path="data/wilayah.db")
    # or set env: WILAYAH_SQLITE_PATH=data/wilayah.db
    # or place file at data/wilayah.db — auto-discovered

  PostgreSQL+PostGIS (Docker, production):
    WilayahLoader()   # uses WILAYAH_DB_* env vars or defaults

Build the SQLite file:
    python -m docker.import_wilayah_sqlite
Build the PostgreSQL DB:
    docker compose up -d && python docker/import_wilayah.py

Usage in extract_villages() pipeline:
    sources: [wilayah_db, building_clusters]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default SQLite path relative to project root (auto-discovered if it exists)
_DEFAULT_SQLITE = Path(__file__).parent.parent.parent / "data" / "wilayah.db"


@dataclass
class WilayahDBConfig:
    host:     str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_HOST", "localhost"))
    port:     int = field(default_factory=lambda: int(os.environ.get("WILAYAH_DB_PORT", "5432")))
    dbname:   str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_NAME", "wilayah"))
    user:     str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_USER", "respondor"))
    password: str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_PASS", "respondor"))


class WilayahLoader:
    """
    Load kelurahan/desa boundaries from the wilayah database and return
    them as Village objects.

    Backend selection (first match wins):
      1. ``sqlite_path`` constructor argument
      2. ``WILAYAH_SQLITE_PATH`` environment variable
      3. ``data/wilayah.db`` if that file exists  ← portable default
      4. PostgreSQL (docker-compose) via ``WilayahDBConfig``

    Population strategy:
      Population is NOT estimated from the DB — wilayah_penduduk only has
      kab-level totals, so per-kelurahan estimates would be meaningless
      uniform averages. All returned villages have population=0.
      Population should be set by the building-cluster source.
    """

    def __init__(
        self,
        sqlite_path: Optional[str | Path] = None,
        config: Optional[WilayahDBConfig] = None,
    ):
        # Resolve SQLite path (env var > constructor arg > default location)
        env_path = os.environ.get("WILAYAH_SQLITE_PATH")
        if sqlite_path is not None:
            self._sqlite_path = Path(sqlite_path)
        elif env_path:
            self._sqlite_path = Path(env_path)
        elif _DEFAULT_SQLITE.exists():
            self._sqlite_path = _DEFAULT_SQLITE
        else:
            self._sqlite_path = None

        self._pg_config = config or WilayahDBConfig()
        self._pg_conn = None

        if self._sqlite_path:
            logger.debug(f"WilayahLoader: SQLite backend → {self._sqlite_path}")
        else:
            logger.debug("WilayahLoader: PostgreSQL backend")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load_villages(
        self,
        bbox: Tuple[float, float, float, float],
        admin_levels: Optional[List[int]] = None,
        population_density_per_km2: float = 2000.0,  # unused — kept for API compat
        max_population_per_village: int = 50_000,     # unused — kept for API compat
        max_area_km2: float = 100.0,
    ) -> List:
        """
        Return Village objects whose boundaries intersect *bbox*.

        bbox: (min_lat, min_lon, max_lat, max_lon)
        admin_levels: None/[9] → kelurahan (kode len=13)
                      [8]      → kecamatan (kode len=8)
                      [7]      → kabupaten (kode len=5)
        max_area_km2: discard boundaries larger than this.
        """
        max_area_m2 = max_area_km2 * 1e6
        if self._sqlite_path:
            villages = self._load_sqlite(bbox, admin_levels, max_area_m2)
        else:
            villages = self._load_postgres(bbox, admin_levels, max_area_m2)
        logger.info(f"WilayahLoader: loaded {len(villages)} villages (bbox={bbox})")
        return villages

    # ------------------------------------------------------------------ #
    # SQLite backend
    # ------------------------------------------------------------------ #

    def _load_sqlite(self, bbox, admin_levels, max_area_m2) -> List:
        import sqlite3
        from shapely.geometry import box as shapely_box
        from shapely.wkt import loads as wkt_loads
        from src.data.models import Village

        min_lat, min_lon, max_lat, max_lon = bbox

        level_map = {9: 13, 8: 8, 7: 5}
        if admin_levels:
            lengths = [level_map[al] for al in admin_levels if al in level_map]
        else:
            lengths = [13]

        bbox_poly = shapely_box(min_lon, min_lat, max_lon, max_lat)
        # Expand centroid filter by 0.5° to catch polygons whose centroid
        # sits outside the bbox but whose geometry still intersects it.
        buf = 0.5

        conn = sqlite3.connect(self._sqlite_path)
        villages = []
        try:
            for kode_len in lengths:
                rows = conn.execute(
                    """
                    SELECT kode, nama, lat, lng, area_m2, geom_wkt
                    FROM wilayah_boundaries
                    WHERE LENGTH(kode) = ?
                      AND lat BETWEEN ? AND ?
                      AND lng BETWEEN ? AND ?
                    ORDER BY kode
                    """,
                    (kode_len,
                     min_lat - buf, max_lat + buf,
                     min_lon - buf, max_lon + buf),
                ).fetchall()

                for kode, nama, lat, lng, area_m2, geom_wkt in rows:
                    if not geom_wkt:
                        continue
                    if max_area_m2 and area_m2 and area_m2 > max_area_m2:
                        continue

                    try:
                        geom = wkt_loads(geom_wkt)
                        if not geom.intersects(bbox_poly):
                            continue
                    except Exception:
                        continue

                    if lat is None or lng is None:
                        try:
                            c = geom.centroid
                            lat, lng = c.y, c.x
                        except Exception:
                            continue

                    villages.append(Village(
                        village_id=f"wilayah_{kode}",
                        name=nama or kode,
                        centroid_lat=lat,
                        centroid_lon=lng,
                        population=0,
                        area_m2=area_m2 or 0.0,
                        admin_level=self._kode_to_admin_level(kode),
                        geometry_wkt=geom_wkt,
                    ))
        finally:
            conn.close()

        return villages

    # ------------------------------------------------------------------ #
    # PostgreSQL backend
    # ------------------------------------------------------------------ #

    def _load_postgres(self, bbox, admin_levels, max_area_m2) -> List:
        from src.data.models import Village

        conn = self._get_pg_conn()
        cur = conn.cursor()

        min_lat, min_lon, max_lat, max_lon = bbox
        bbox_wkt = f"ST_MakeEnvelope({min_lon},{min_lat},{max_lon},{max_lat},4326)"

        level_map = {9: 13, 8: 8, 7: 5}
        if admin_levels:
            length_filter = " OR ".join(
                f"LENGTH(b.kode) = {level_map.get(al, 13)}"
                for al in admin_levels if al in level_map
            )
            length_clause = f"AND ({length_filter})"
        else:
            length_clause = "AND LENGTH(b.kode) = 13"

        sql = f"""
            SELECT
                b.kode,
                b.nama,
                b.lat,
                b.lng,
                ST_Area(b.geom::geography) AS area_m2,
                ST_AsText(ST_Multi(b.geom)) AS geom_wkt
            FROM wilayah_boundaries b
            WHERE b.geom IS NOT NULL
              AND ST_Intersects(b.geom, {bbox_wkt})
              AND ST_Area(b.geom::geography) <= {max_area_m2}
              {length_clause}
            ORDER BY b.kode
        """
        cur.execute(sql)
        rows = cur.fetchall()

        villages = []
        for kode, nama, lat, lng, area_m2, geom_wkt in rows:
            if lat is None or lng is None:
                cur.execute(
                    "SELECT ST_Y(ST_Centroid(geom)), ST_X(ST_Centroid(geom)) "
                    "FROM wilayah_boundaries WHERE kode = %s",
                    (kode,),
                )
                c = cur.fetchone()
                if c:
                    lat, lng = c[0], c[1]
                else:
                    continue

            villages.append(Village(
                village_id=f"wilayah_{kode}",
                name=nama or kode,
                centroid_lat=lat,
                centroid_lon=lng,
                population=0,
                area_m2=area_m2 or 0.0,
                admin_level=self._kode_to_admin_level(kode),
                geometry_wkt=geom_wkt,
            ))

        cur.close()
        return villages

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _get_pg_conn(self):
        if self._pg_conn is None or self._pg_conn.closed:
            try:
                import psycopg2
            except ImportError:
                raise RuntimeError(
                    "psycopg2 not installed. Run: pip install psycopg2-binary\n"
                    "Or use the SQLite backend: python -m docker.import_wilayah_sqlite"
                )
            cfg = self._pg_config
            self._pg_conn = psycopg2.connect(
                host=cfg.host, port=cfg.port,
                dbname=cfg.dbname, user=cfg.user, password=cfg.password,
            )
            self._pg_conn.autocommit = True
        return self._pg_conn

    @staticmethod
    def _kode_to_admin_level(kode: str) -> int:
        n = len(kode)
        if n == 2:  return 4   # provinsi
        if n == 5:  return 7   # kabupaten/kota
        if n == 8:  return 8   # kecamatan
        return 9               # kelurahan/desa

    def close(self):
        if self._pg_conn and not self._pg_conn.closed:
            self._pg_conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
