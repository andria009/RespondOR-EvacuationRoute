#!/usr/bin/env python3
"""
Import all cahayadsn-wilayah* MySQL SQL dumps into the PostgreSQL+PostGIS
wilayah database running in Docker.

Usage:
    python docker/import_wilayah.py [--host localhost] [--port 5432]

The script:
  1. Truncates all target tables (idempotent — safe to re-run).
  2. Parses each MySQL SQL file, strips MySQL-specific syntax.
  3. For wilayah_boundaries: converts the JSON path (depth-3 Polygon /
     depth-4 MultiPolygon, [lat,lon] order) to PostGIS WKT (lon,lat).
  4. Bulk-inserts via psycopg2 executemany with 2000-row batches.

Dependencies:
    psycopg2-binary  (already in requirements.txt after this commit)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("import_wilayah")

# ------------------------------------------------------------------ #
# Paths (relative to project root)
# ------------------------------------------------------------------ #
MISC = Path(__file__).parent.parent / "misc"
WILAYAH_DIR       = MISC / "cahayadsn-wilayah" / "db"
BOUNDARIES_DIR    = MISC / "cahayadsn-wilayah_boundaries" / "db"
KODEPOS_DIR       = MISC / "cahayadsn-wilayah_kodepos" / "db"

BATCH_SIZE = 2000


# ------------------------------------------------------------------ #
# SQL value-row parser
# ------------------------------------------------------------------ #

def _unescape(s: str) -> str:
    """Remove outer quotes and unescape MySQL string literals."""
    if (s.startswith("'") and s.endswith("'")) or \
       (s.startswith('"') and s.endswith('"')):
        inner = s[1:-1]
        inner = inner.replace("\\'", "'").replace('\\"', '"')
        inner = inner.replace("\\\\", "\\").replace("\\n", "\n")
        return inner
    if s.upper() == "NULL":
        return None
    return s


def _parse_value_token(token: str):
    """Parse a single SQL value token to a Python value."""
    token = token.strip()
    if token.upper() == "NULL":
        return None
    if (token.startswith("'") and token.endswith("'")) or \
       (token.startswith('"') and token.endswith('"')):
        return _unescape(token)
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _split_row(row_str: str) -> list:
    """
    Split a single VALUES row like ('a','b',1.0,'...') into a list of tokens,
    respecting nested quotes so that embedded JSON strings are kept intact.
    """
    # Strip outer parentheses
    row_str = row_str.strip()
    if row_str.startswith("("):
        row_str = row_str[1:]
    if row_str.endswith(")"):
        row_str = row_str[:-1]

    tokens = []
    depth = 0  # parenthesis depth inside a quoted string
    in_quote = False
    quote_char = None
    current = []
    i = 0
    while i < len(row_str):
        c = row_str[i]
        if not in_quote:
            if c in ("'", '"'):
                in_quote = True
                quote_char = c
                current.append(c)
            elif c == "," and depth == 0:
                tokens.append("".join(current).strip())
                current = []
            else:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                current.append(c)
        else:
            if c == "\\" and i + 1 < len(row_str):
                current.append(c)
                current.append(row_str[i + 1])
                i += 2
                continue
            elif c == quote_char:
                in_quote = False
            current.append(c)
        i += 1
    if current:
        tokens.append("".join(current).strip())
    return [_parse_value_token(t) for t in tokens]


def iter_insert_rows(sql_text: str) -> Iterator[list]:
    """
    Yield value-row lists from all INSERT statements in a SQL text.
    Handles multi-row INSERT INTO … VALUES (…),(…),(…);
    """
    # Find INSERT … VALUES block
    for m in re.finditer(
        r"INSERT INTO\s+[`\"\w]+\s*\([^)]+\)\s*VALUES\s*(.*?);",
        sql_text,
        re.DOTALL | re.IGNORECASE,
    ):
        values_block = m.group(1).strip()
        # Split into individual rows: find top-level parenthesised groups
        row_pat = re.compile(r"\((?:[^()']|'(?:[^'\\]|\\.)*')*\)", re.DOTALL)
        for row_m in row_pat.finditer(values_block):
            row_str = row_m.group(0)
            yield _split_row(row_str)


# ------------------------------------------------------------------ #
# Geometry conversion: JSON path → PostGIS WKT MultiPolygon
# Coordinates in source are [lat, lon]; PostGIS wants (lon lat).
# ------------------------------------------------------------------ #

def _ring_wkt(ring: list) -> str:
    """[[lat,lon], ...] → 'lon lat, ...' (auto-closes unclosed rings)."""
    pts = list(ring)
    if pts and pts[0] != pts[-1]:
        pts.append(pts[0])  # PostGIS requires closed rings
    return ", ".join(f"{pt[1]} {pt[0]}" for pt in pts)


def _polygon_wkt(poly_rings: list) -> str:
    """
    [exterior_ring, *holes] → '((lon lat,...), (hole_wkt))'
    WKT MULTIPOLYGON needs: MULTIPOLYGON( ((ring),(hole)), ((ring)) )
    so each polygon is wrapped in outer (), each ring in inner ().
    """
    rings_wkt = ", ".join(f"({_ring_wkt(r)})" for r in poly_rings)
    return f"({rings_wkt})"


def path_to_multipolygon_wkt(path_json: str | None) -> str | None:
    """
    Convert the wilayah_boundaries path JSON to a WKT MultiPolygon.

    Depth-3  [[[lat,lon], ...]]              → single-ring Polygon → MultiPolygon
    Depth-4  [[[[lat,lon], ...]], ...]        → MultiPolygon
    """
    if not path_json:
        return None
    try:
        data = json.loads(path_json)
    except (json.JSONDecodeError, TypeError):
        return None
    if not data:
        return None

    # Detect depth by examining data[0][0][0]:
    #   depth-3  [[[lat,lon],...]]         → data[0][0][0] is float (a coord)
    #   depth-4  [[[[lat,lon],...],...]]   → data[0][0][0] is [lat,lon] (a point list)
    try:
        probe = data[0][0][0]
    except (IndexError, TypeError):
        return None

    if isinstance(probe, (int, float)):
        # depth-3: data = [ring1, ring2, ...] — rings of a single Polygon
        polygons = [data]
    elif isinstance(probe, list) and isinstance(probe[0], (int, float)):
        # depth-4: data = [polygon1, polygon2, ...] where each polygon = [ring, ...]
        polygons = data
    else:
        # depth-2: data is a bare ring [[lat,lon],...] — wrap it
        polygons = [[data]]

    poly_parts = []
    for poly_rings in polygons:
        if not poly_rings:
            continue
        poly_parts.append(_polygon_wkt(poly_rings))

    if not poly_parts:
        return None
    return "MULTIPOLYGON(" + ", ".join(poly_parts) + ")"


# ------------------------------------------------------------------ #
# Importers
# ------------------------------------------------------------------ #

def import_wilayah(cur):
    sql_file = WILAYAH_DIR / "wilayah.sql"
    log.info(f"  importing {sql_file.name} …")
    text = sql_file.read_text(encoding="utf-8")
    rows = list(iter_insert_rows(text))
    _bulk_insert(cur, "wilayah", ["kode", "nama"], rows)
    log.info(f"    → {len(rows):,} rows")


def import_wilayah_penduduk(cur):
    sql_file = WILAYAH_DIR / "wilayah_penduduk.sql"
    log.info(f"  importing {sql_file.name} …")
    text = sql_file.read_text(encoding="utf-8")
    rows = list(iter_insert_rows(text))
    # penduduk values may be quoted integers — coerce
    clean = []
    for r in rows:
        if len(r) >= 5:
            clean.append([r[0], r[1], int(r[2] or 0), int(r[3] or 0), int(r[4] or 0)])
    _bulk_insert(cur, "wilayah_penduduk", ["kode", "nama", "pria", "wanita", "total"], clean)
    log.info(f"    → {len(clean):,} rows")


def import_wilayah_luas(cur):
    sql_file = WILAYAH_DIR / "wilayah_luas.sql"
    log.info(f"  importing {sql_file.name} …")
    text = sql_file.read_text(encoding="utf-8")
    rows = list(iter_insert_rows(text))
    clean = [[r[0], r[1], float(r[2] or 0)] for r in rows if len(r) >= 3]
    _bulk_insert(cur, "wilayah_luas", ["kode", "nama", "luas"], clean)
    log.info(f"    → {len(clean):,} rows")


def import_wilayah_kodepos(cur):
    sql_file = KODEPOS_DIR / "wilayah_kodepos.sql"
    log.info(f"  importing {sql_file.name} …")
    text = sql_file.read_text(encoding="utf-8")
    rows = list(iter_insert_rows(text))
    clean = [[r[0], str(r[1])] for r in rows if len(r) >= 2]
    _bulk_insert(cur, "wilayah_kodepos", ["kode", "kodepos"], clean)
    log.info(f"    → {len(clean):,} rows")


def import_wilayah_boundaries(cur):
    """
    Walk all SQL files under BOUNDARIES_DIR and import them.
    Converts path JSON → PostGIS MultiPolygon geometry.
    """
    sql_files = sorted(BOUNDARIES_DIR.rglob("*.sql"))
    # Skip the DDL file
    sql_files = [f for f in sql_files if "ddl_" not in f.name]

    total = 0
    cols = ["kode", "nama", "lat", "lng", "path", "status", "geom"]

    for sql_file in sql_files:
        text = sql_file.read_text(encoding="utf-8")
        rows_raw = list(iter_insert_rows(text))
        if not rows_raw:
            continue

        batch = []
        for r in rows_raw:
            # Pad short rows (some files omit status)
            while len(r) < 6:
                r.append(None)
            kode, nama, lat, lng, path_json, status = r[:6]
            wkt = path_to_multipolygon_wkt(path_json if isinstance(path_json, str) else None)
            batch.append([kode, nama, lat, lng, path_json, status, wkt])

        _bulk_insert_geom(cur, batch)
        total += len(batch)

    log.info(f"    → {total:,} boundary rows total")


# ------------------------------------------------------------------ #
# Bulk insert helpers
# ------------------------------------------------------------------ #

def _bulk_insert(cur, table: str, cols: list[str], rows: list):
    if not rows:
        return
    placeholders = ", ".join(["%s"] * len(cols))
    col_list = ", ".join(cols)
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) ON CONFLICT (kode) DO NOTHING"
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        cur.executemany(sql, batch)


def _bulk_insert_geom(cur, rows: list):
    """Insert boundary rows; last column is WKT or None for geom."""
    if not rows:
        return
    sql = """
        INSERT INTO wilayah_boundaries (kode, nama, lat, lng, path, status, geom)
        VALUES (
            %s, %s, %s, %s, %s, %s,
            CASE WHEN %s IS NOT NULL
                 THEN ST_Multi(ST_GeomFromText(%s, 4326))
                 ELSE NULL END
        )
        ON CONFLICT (kode) DO NOTHING
    """
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i: i + BATCH_SIZE]
        # Each row: [kode, nama, lat, lng, path, status, wkt]
        # SQL needs wkt twice (for IS NOT NULL check + ST_GeomFromText)
        params = [r[:6] + [r[6], r[6]] for r in batch]
        cur.executemany(sql, params)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Import wilayah data into PostgreSQL+PostGIS")
    parser.add_argument("--host",     default="localhost")
    parser.add_argument("--port",     type=int, default=5432)
    parser.add_argument("--dbname",   default="wilayah")
    parser.add_argument("--user",     default="respondor")
    parser.add_argument("--password", default="respondor")
    parser.add_argument("--skip-boundaries", action="store_true",
                        help="Skip the large boundary import (useful for quick testing)")
    args = parser.parse_args()

    try:
        import psycopg2
    except ImportError:
        log.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    log.info(f"Connecting to {args.host}:{args.port}/{args.dbname} …")
    conn = psycopg2.connect(
        host=args.host, port=args.port,
        dbname=args.dbname, user=args.user, password=args.password,
    )
    conn.autocommit = False
    cur = conn.cursor()

    try:
        tables = ["wilayah", "wilayah_penduduk", "wilayah_luas",
                  "wilayah_kodepos", "wilayah_boundaries"]
        log.info("Truncating tables …")
        # Truncate in reverse FK dependency order (none here, but good practice)
        for t in reversed(tables):
            cur.execute(f"TRUNCATE TABLE {t} RESTART IDENTITY CASCADE")
        conn.commit()

        t_start = time.perf_counter()

        log.info("Importing wilayah …")
        import_wilayah(cur)
        conn.commit()

        log.info("Importing wilayah_penduduk …")
        import_wilayah_penduduk(cur)
        conn.commit()

        log.info("Importing wilayah_luas …")
        import_wilayah_luas(cur)
        conn.commit()

        log.info("Importing wilayah_kodepos …")
        import_wilayah_kodepos(cur)
        conn.commit()

        if not args.skip_boundaries:
            log.info("Importing wilayah_boundaries (this may take a few minutes) …")
            import_wilayah_boundaries(cur)
            conn.commit()
            log.info("Refreshing spatial index …")
            # VACUUM must run outside a transaction block
            conn.autocommit = True
            cur.execute("VACUUM ANALYZE wilayah_boundaries")
            conn.autocommit = False
        else:
            log.warning("Skipping boundaries import (--skip-boundaries set)")

        elapsed = time.perf_counter() - t_start
        log.info(f"Done in {elapsed:.1f}s")

    except Exception as exc:
        conn.rollback()
        log.error(f"Import failed: {exc}")
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
