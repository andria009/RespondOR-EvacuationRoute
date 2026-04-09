-- RespondOR-EvacuationRoute: Wilayah database schema
-- Initialised automatically by PostgreSQL on first container start.

CREATE EXTENSION IF NOT EXISTS postgis;

-- ------------------------------------------------------------------ --
-- Wilayah hierarchy (kode = administrative code, L1–L4)
-- kode format: "11" (prov) | "11.01" (kab) | "11.01.01" (kec) | "11.01.01.2001" (kel)
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS wilayah (
    kode  VARCHAR(13) NOT NULL PRIMARY KEY,
    nama  VARCHAR(100) NOT NULL
);
CREATE INDEX IF NOT EXISTS wilayah_nama_idx ON wilayah (nama);

-- ------------------------------------------------------------------ --
-- Population by kabupaten (kab-level only, 553 rows)
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS wilayah_penduduk (
    kode    VARCHAR(13) NOT NULL PRIMARY KEY,
    nama    VARCHAR(100) NOT NULL,
    pria    INTEGER NOT NULL DEFAULT 0,
    wanita  INTEGER NOT NULL DEFAULT 0,
    total   INTEGER NOT NULL DEFAULT 0
);

-- ------------------------------------------------------------------ --
-- Area by kabupaten (km², BIG 2024 data, ~552 rows)
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS wilayah_luas (
    kode  VARCHAR(13) NOT NULL PRIMARY KEY,
    nama  VARCHAR(100) NOT NULL,
    luas  DOUBLE PRECISION NOT NULL DEFAULT 0
);

-- ------------------------------------------------------------------ --
-- Postal codes by kelurahan
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS wilayah_kodepos (
    kode    VARCHAR(13) NOT NULL PRIMARY KEY,
    kodepos VARCHAR(5)  NOT NULL
);
CREATE INDEX IF NOT EXISTS wilayah_kodepos_kodepos_idx ON wilayah_kodepos (kodepos);

-- ------------------------------------------------------------------ --
-- Administrative boundaries with PostGIS geometry
-- path (TEXT): original JSON from source  —  depth-3 = Polygon,
--              depth-4 = MultiPolygon, coordinates [lat, lon]
-- geom: PostGIS MultiPolygon(4326) converted from path (lon, lat)
-- ------------------------------------------------------------------ --
CREATE TABLE IF NOT EXISTS wilayah_boundaries (
    kode    VARCHAR(13) NOT NULL PRIMARY KEY,
    nama    VARCHAR(100),
    lat     DOUBLE PRECISION,
    lng     DOUBLE PRECISION,
    path    TEXT,
    status  SMALLINT,
    geom    GEOMETRY(MultiPolygon, 4326)
);
CREATE INDEX IF NOT EXISTS wilayah_boundaries_geom_idx
    ON wilayah_boundaries USING GIST (geom);
CREATE INDEX IF NOT EXISTS wilayah_boundaries_kode_prefix_idx
    ON wilayah_boundaries (LEFT(kode, 5));  -- fast kab-level prefix filter

-- ------------------------------------------------------------------ --
-- Convenience view: kelurahan with centroid, area (km²), population
-- (joined from wilayah + penduduk + luas where kode length = 13)
-- ------------------------------------------------------------------ --
CREATE OR REPLACE VIEW v_kelurahan AS
SELECT
    b.kode,
    b.nama,
    b.lat,
    b.lng,
    b.geom,
    ST_Area(b.geom::geography) / 1e6                AS area_km2,
    ST_Centroid(b.geom)                             AS centroid,
    -- Population: kelurahan-level not available, use kab total / kab kel count
    -- (populated by WilayahLoader at query time)
    NULL::INTEGER                                   AS population_estimate
FROM wilayah_boundaries b
WHERE LENGTH(b.kode) = 13;
