"""
WilayahLoader — query the PostGIS wilayah database and return Village objects
with official Indonesian administrative boundaries and population estimates.

The database is populated by running:
    docker compose up -d
    python docker/import_wilayah.py

Connection defaults match docker-compose.yml (localhost:5432, db=wilayah,
user/password=respondor).  Override via environment variables or constructor.

Usage in extract_villages() pipeline:
    sources: [wilayah_db, building_clusters]
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class WilayahDBConfig:
    host:     str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_HOST", "localhost"))
    port:     int = field(default_factory=lambda: int(os.environ.get("WILAYAH_DB_PORT", "5432")))
    dbname:   str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_NAME", "wilayah"))
    user:     str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_USER", "respondor"))
    password: str = field(default_factory=lambda: os.environ.get("WILAYAH_DB_PASS", "respondor"))


class WilayahLoader:
    """
    Load kelurahan/desa boundaries from the local PostGIS wilayah database
    and return them as Village objects.

    Population strategy (best available, in priority order):
    1. wilayah_penduduk (kab-level total) divided evenly among its kelurahan.
    2. Fallback: area_km2 * population_density_per_km2.
    """

    def __init__(self, config: Optional[WilayahDBConfig] = None):
        self.config = config or WilayahDBConfig()
        self._conn = None

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

        Population is NOT estimated from the DB — wilayah_penduduk only has
        kab-level totals, so per-kelurahan estimates would be meaningless
        uniform averages.  All returned villages have population=0.
        Population should be set by the building-cluster source.

        bbox: (min_lat, min_lon, max_lat, max_lon)
        admin_levels: kode-length filter.
            None / [9] → kelurahan only (kode len=13)
            [8]        → kecamatan (kode len=8)
            [7]        → kabupaten (kode len=5)
        max_area_km2: discard boundaries larger than this (default 100 km²).
        """
        from src.data.models import Village

        conn = self._get_conn()
        cur = conn.cursor()

        min_lat, min_lon, max_lat, max_lon = bbox
        # Build a bbox envelope in PostGIS (note: x=lon, y=lat)
        bbox_wkt = (
            f"ST_MakeEnvelope({min_lon},{min_lat},{max_lon},{max_lat},4326)"
        )

        # Resolve kode length filter
        # L4 kelurahan kode = "33.22.01.2001" (13 chars)
        # L3 kecamatan kode = "33.22.01"      (8 chars)
        # L2 kabupaten kode = "33.22"          (5 chars)
        level_map = {9: 13, 8: 8, 7: 5}
        if admin_levels:
            length_filter = " OR ".join(
                f"LENGTH(b.kode) = {level_map.get(al, 13)}"
                for al in admin_levels
                if al in level_map
            )
            length_clause = f"AND ({length_filter})"
        else:
            length_clause = "AND LENGTH(b.kode) = 13"

        max_area_m2 = max_area_km2 * 1e6
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

            v = Village(
                village_id=f"wilayah_{kode}",
                name=nama or kode,
                centroid_lat=lat,
                centroid_lon=lng,
                population=0,       # no per-kelurahan population in DB
                area_m2=area_m2 or 0.0,
                admin_level=self._kode_to_admin_level(kode),
                geometry_wkt=geom_wkt,
            )
            villages.append(v)

        cur.close()
        logger.info(f"WilayahLoader: loaded {len(villages)} villages from DB (bbox={bbox})")
        return villages

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            try:
                import psycopg2
            except ImportError:
                raise RuntimeError(
                    "psycopg2 not installed. Run: pip install psycopg2-binary"
                )
            cfg = self.config
            self._conn = psycopg2.connect(
                host=cfg.host, port=cfg.port,
                dbname=cfg.dbname, user=cfg.user, password=cfg.password,
            )
            self._conn.autocommit = True
        return self._conn

    @staticmethod
    def _kode_to_admin_level(kode: str) -> int:
        n = len(kode)
        if n == 2:  return 4   # provinsi
        if n == 5:  return 7   # kabupaten/kota
        if n == 8:  return 8   # kecamatan
        return 9               # kelurahan/desa

    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
