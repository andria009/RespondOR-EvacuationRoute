#!/usr/bin/env python3
"""
Import all cahayadsn-wilayah* MySQL SQL dumps into a portable SQLite database.

No Docker, no PostGIS, no server required — just Python + shapely.

Usage:
    python -m docker.import_wilayah_sqlite
    python -m docker.import_wilayah_sqlite --output data/wilayah.db
    python -m docker.import_wilayah_sqlite --skip-boundaries  # fast, no geometry

Output: data/wilayah.db (SQLite, ~300 MB with boundaries)

The SQLite schema mirrors the PostGIS schema, except:
  - geom is stored as WKT text (no spatial index)
  - area_m2 is pre-computed and stored (no ST_Area)
  - bbox queries in WilayahLoader use centroid lat/lng columns + Shapely intersection

Dependencies:
    shapely  (already in requirements.txt)
"""

from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
import time
from pathlib import Path

# Reuse all SQL parsing + geometry helpers from the PostGIS importer
from docker.import_wilayah import (
    BOUNDARIES_DIR,
    BATCH_SIZE,
    KODEPOS_DIR,
    WILAYAH_DIR,
    iter_insert_rows,
    path_to_multipolygon_wkt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("import_wilayah_sqlite")

DEFAULT_OUT = Path(__file__).parent.parent / "data" / "wilayah.db"


# ------------------------------------------------------------------ #
# Schema
# ------------------------------------------------------------------ #

SCHEMA = """
CREATE TABLE IF NOT EXISTS wilayah (
    kode TEXT NOT NULL PRIMARY KEY,
    nama TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS wilayah_nama_idx ON wilayah (nama);

CREATE TABLE IF NOT EXISTS wilayah_penduduk (
    kode   TEXT NOT NULL PRIMARY KEY,
    nama   TEXT NOT NULL,
    pria   INTEGER NOT NULL DEFAULT 0,
    wanita INTEGER NOT NULL DEFAULT 0,
    total  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wilayah_luas (
    kode TEXT NOT NULL PRIMARY KEY,
    nama TEXT NOT NULL,
    luas REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS wilayah_kodepos (
    kode    TEXT NOT NULL PRIMARY KEY,
    kodepos TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wilayah_boundaries (
    kode    TEXT NOT NULL PRIMARY KEY,
    nama    TEXT,
    lat     REAL,
    lng     REAL,
    area_m2 REAL,
    geom_wkt TEXT
);
CREATE INDEX IF NOT EXISTS wilayah_boundaries_lat_idx
    ON wilayah_boundaries (lat);
CREATE INDEX IF NOT EXISTS wilayah_boundaries_lng_idx
    ON wilayah_boundaries (lng);
CREATE INDEX IF NOT EXISTS wilayah_boundaries_kode_len_idx
    ON wilayah_boundaries (LENGTH(kode));
"""


# ------------------------------------------------------------------ #
# Area computation (Shapely only, no pyproj required)
# ------------------------------------------------------------------ #

def _approx_area_m2(wkt: str, centroid_lat: float) -> float:
    """
    Approximate geodetic area from a WKT MultiPolygon.
    Uses Shapely's planar area (degrees²) scaled by metres-per-degree.
    Accurate to ~5% for small polygons (< 1,000 km²).
    """
    try:
        from shapely.wkt import loads
        geom = loads(wkt)
        lat_m = 111_320.0                              # metres per degree latitude
        lon_m = 111_320.0 * math.cos(math.radians(centroid_lat))
        return geom.area * lat_m * lon_m
    except Exception:
        return 0.0


# ------------------------------------------------------------------ #
# Importers
# ------------------------------------------------------------------ #

def _bulk_insert(conn: sqlite3.Connection, table: str, cols: list[str], rows: list):
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"
    for i in range(0, len(rows), BATCH_SIZE):
        conn.executemany(sql, rows[i: i + BATCH_SIZE])


def import_wilayah(conn):
    sql_file = WILAYAH_DIR / "wilayah.sql"
    log.info(f"  {sql_file.name} …")
    rows = list(iter_insert_rows(sql_file.read_text(encoding="utf-8")))
    _bulk_insert(conn, "wilayah", ["kode", "nama"], rows)
    log.info(f"    → {len(rows):,} rows")


def import_wilayah_penduduk(conn):
    sql_file = WILAYAH_DIR / "wilayah_penduduk.sql"
    log.info(f"  {sql_file.name} …")
    rows = list(iter_insert_rows(sql_file.read_text(encoding="utf-8")))
    clean = []
    for r in rows:
        if len(r) >= 5:
            clean.append([r[0], r[1], int(r[2] or 0), int(r[3] or 0), int(r[4] or 0)])
    _bulk_insert(conn, "wilayah_penduduk", ["kode", "nama", "pria", "wanita", "total"], clean)
    log.info(f"    → {len(clean):,} rows")


def import_wilayah_luas(conn):
    sql_file = WILAYAH_DIR / "wilayah_luas.sql"
    log.info(f"  {sql_file.name} …")
    rows = list(iter_insert_rows(sql_file.read_text(encoding="utf-8")))
    clean = [[r[0], r[1], float(r[2] or 0)] for r in rows if len(r) >= 3]
    _bulk_insert(conn, "wilayah_luas", ["kode", "nama", "luas"], clean)
    log.info(f"    → {len(clean):,} rows")


def import_wilayah_kodepos(conn):
    sql_file = KODEPOS_DIR / "wilayah_kodepos.sql"
    log.info(f"  {sql_file.name} …")
    rows = list(iter_insert_rows(sql_file.read_text(encoding="utf-8")))
    clean = [[r[0], str(r[1])] for r in rows if len(r) >= 2]
    _bulk_insert(conn, "wilayah_kodepos", ["kode", "kodepos"], clean)
    log.info(f"    → {len(clean):,} rows")


def import_wilayah_boundaries(conn):
    sql_files = sorted(BOUNDARIES_DIR.rglob("*.sql"))
    sql_files = [f for f in sql_files if "ddl_" not in f.name]

    total = 0
    for sql_file in sql_files:
        text = sql_file.read_text(encoding="utf-8")
        rows_raw = list(iter_insert_rows(text))
        if not rows_raw:
            continue

        batch = []
        for r in rows_raw:
            while len(r) < 6:
                r.append(None)
            kode, nama, lat, lng, path_json, _status = r[:6]
            wkt = path_to_multipolygon_wkt(
                path_json if isinstance(path_json, str) else None
            )
            area = _approx_area_m2(wkt, lat or 0.0) if wkt else 0.0
            batch.append([kode, nama, lat, lng, area, wkt])

        _bulk_insert(conn, "wilayah_boundaries",
                     ["kode", "nama", "lat", "lng", "area_m2", "geom_wkt"], batch)
        total += len(batch)
        log.info(f"    {sql_file.name}: {len(batch):,} rows (total {total:,})")

    log.info(f"    → {total:,} boundary rows total")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Import wilayah data into a portable SQLite database"
    )
    parser.add_argument("--output", default=str(DEFAULT_OUT),
                        help=f"Output SQLite path (default: {DEFAULT_OUT})")
    parser.add_argument("--skip-boundaries", action="store_true",
                        help="Skip boundary import (fast, no geometry)")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log.info(f"Removing existing DB: {out_path}")
        out_path.unlink()

    log.info(f"Creating SQLite DB: {out_path}")
    conn = sqlite3.connect(out_path)
    conn.executescript(SCHEMA)

    t_start = time.perf_counter()

    conn.execute("BEGIN")
    log.info("Importing wilayah …")
    import_wilayah(conn)

    log.info("Importing wilayah_penduduk …")
    import_wilayah_penduduk(conn)

    log.info("Importing wilayah_luas …")
    import_wilayah_luas(conn)

    log.info("Importing wilayah_kodepos …")
    import_wilayah_kodepos(conn)
    conn.execute("COMMIT")

    if not args.skip_boundaries:
        log.info("Importing wilayah_boundaries (this may take a few minutes) …")
        conn.execute("BEGIN")
        import_wilayah_boundaries(conn)
        conn.execute("COMMIT")
        log.info("Optimising …")
        conn.execute("ANALYZE")
    else:
        log.warning("Skipping boundaries import (--skip-boundaries)")

    conn.close()

    size_mb = out_path.stat().st_size / 1_048_576
    elapsed = time.perf_counter() - t_start
    log.info(f"Done in {elapsed:.1f}s  |  {out_path}  ({size_mb:.0f} MB)")
    log.info("")
    log.info("Use with WilayahLoader:")
    log.info(f"  WilayahLoader(sqlite_path='{out_path}')")
    log.info(f"  # or set env: WILAYAH_SQLITE_PATH={out_path}")


if __name__ == "__main__":
    main()
