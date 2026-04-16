# RespondOR-EvacuationRoute

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research-grade disaster evacuation route optimization system for Indonesia. Integrates OpenStreetMap road networks, InaRISK BNPB hazard data, and multi-objective routing to maximize the number of people evacuated to safe shelters across multiple disaster types and compound hazard scenarios.

---

## Overview

Given a disaster event configuration, the system:

1. **Extracts** road network, settlement clusters, and shelters from OpenStreetMap — using configurable multi-source village extraction (admin boundaries, place nodes, DBSCAN building clusters) and optional wilayah DB for official kelurahan/desa boundaries
2. **Scores hazard risk** via the InaRISK BNPB API for each village, shelter, and road segment — with grid-snap caching that eliminates redundant API calls on subsequent runs
3. **Builds** a weighted evacuation graph with composite edge weights (distance × road quality × risk penalty); supports compound hazard scenarios (multiple weighted InaRISK layers)
4. **Computes** optimized evacuation routes from each village to all reachable shelters using Dijkstra's algorithm, scored by a composite metric that penalises distance, risk, road quality, travel time, and proximity to the disaster centre
5. **Assigns** village populations to shelters using greedy optimisation (or optional LP), maximising total evacuated population subject to shelter capacity constraints
6. **Exports** results as interactive HTML maps, route CSVs, graph statistics, SHP files (for QGIS / GAMA Platform), and GAMA ABM simulation inputs

---

## Repository Structure

```
RespondOR-EvacuationRoute/
├── configs/                            # One YAML per scenario
│   ├── banjarnegara_landslide_2021.yaml
│   ├── cianjur_earthquake_2022.yaml
│   ├── demak_flood_2024.yaml
│   ├── lewotobi_eruption_2024.yaml
│   ├── merapi_eruption_2023.yaml
│   ├── palu_earthquake_2018.yaml
│   ├── sinabung_eruption_2016.yaml
│   ├── sumedang_landslide_2021.yaml
│   ├── sukabumi_landslide_2024.yaml
│   └── tuban_earthquake_2024.yaml
├── data/
│   └── raw/
│       ├── osm_cache/                  # OSM network + village + shelter cache (bbox-keyed)
│       └── inarisk_cache/              # InaRISK POI + road + hazard grid cache
├── docker/
│   ├── import_wilayah.py               # Populate wilayah PostGIS DB from shapefiles
│   ├── import_wilayah_sqlite.py        # Portable SQLite alternative (no Docker needed)
│   └── init/                           # DB initialisation SQL
├── docker-compose.yml                  # PostGIS wilayah DB (for L9 kelurahan boundaries)
├── experiments/
│   ├── prerun_validation.py            # Pre-run check: config, cache, API budget, time estimate
│   ├── preview_region.py               # Preview OSM extraction region (villages, roads, admin)
│   ├── preview_hazard.py               # Preview InaRISK hazard grid for a scenario
│   ├── export_shp.py                   # Export villages/shelters/roads as SHP + GeoJSON
│   ├── prepare_gama_inputs.py          # Write GAMA simulation inputs (CSV + SHP + JSON)
│   ├── build_legacy_input.py           # Build preloaded/legacy input files from OSM cache
│   ├── compare_routes.py               # Compare route outputs across scenarios
│   ├── benchmark_all.py                # Full pipeline benchmark: all scenarios × all execution modes
│   └── benchmark_extraction.py         # Extraction-only benchmark: OSM + InaRISK isolation runs
├── helper/
│   └── benchmark_compare.py            # Print/compare benchmark_results.json from the terminal
├── simulation/
│   └── models/
│       └── EvacuationModel.gaml        # GAMA agent-based evacuation model (4 modes, BPR, batch sweep)
├── output/                             # Pipeline outputs (auto-generated, per scenario)
│   └── <scenario_id>/
├── src/
│   ├── config/config_loader.py         # YAML config loader + AppConfig dataclasses
│   ├── data/
│   │   ├── inarisk_client.py           # InaRISK BNPB API client (grid-snap + batched cache)
│   │   ├── models.py                   # Core dataclasses (Village, Shelter, Route, ...)
│   │   ├── osm_extractor.py            # OSM extraction + multi-source villages + circular shelters
│   │   ├── population_loader.py        # Population and shelter capacity estimation
│   │   └── wilayah_loader.py           # Wilayah DB loader — SQLite or PostgreSQL+PostGIS
│   ├── graph/graph_builder.py          # Weighted NetworkX evacuation graph + InaRISK enrichment
│   ├── hpc/
│   │   ├── naive_runner.py             # Sequential single-process pipeline
│   │   ├── parallel_runner.py          # ThreadPool (I/O) + ProcessPool (routing)
│   │   ├── distributed_runner.py       # Hybrid MPI × multiprocessing runner (mpi4py; fallback: ProcessPool)
│   │   └── runner_utils.py             # Shared helpers: MemoryTracker, apply_risk_parallel, resolve_hazard_layers
│   ├── routing/
│   │   ├── heuristic_optimizer.py      # Dijkstra-based route computation and scoring
│   │   └── assignment.py               # Greedy / LP population-to-shelter assignment
│   ├── utils/
│   │   └── logging_setup.py            # File + console logging to logs/
│   ├── visualization/visualizer.py     # Folium interactive map + matplotlib charts
│   └── main.py                         # CLI entry point
└── tests/
```

---

## Scenarios

| Scenario | Config | Disaster | Center | Radius |
|---|---|---|---|---|
| Banjarnegara Landslide | `banjarnegara_landslide_2021.yaml` | Landslide, Feb 2021 | -7.37, 109.68 | 10 km |
| Sukabumi Landslide | `sukabumi_landslide_2024.yaml` | Landslide, Dec 2024 | -7.12, 106.73 | 18 km |
| Merapi Eruption | `merapi_eruption_2023.yaml` | Volcano, 2023 | -7.54, 110.44 | 15 km |
| Sumedang Landslide | `sumedang_landslide_2021.yaml` | Landslide, Jan 2021 | -6.85, 107.92 | 10 km |
| Cianjur Earthquake | `cianjur_earthquake_2022.yaml` | Earthquake, Nov 2022 | -6.86, 107.05 | 20 km |
| Sinabung Eruption | `sinabung_eruption_2016.yaml` | Volcano, 2016 | 3.17, 98.39 | 15 km |
| Palu Earthquake | `palu_earthquake_2018.yaml` | Compound (earthquake + tsunami + liquefaction), Sep 2018 | -0.90, 119.87 | 20 km |
| Demak Flood | `demak_flood_2024.yaml` | Flood, Feb 2024 | -6.89, 110.63 | 25 km |
| Tuban Earthquake | `tuban_earthquake_2024.yaml` | Earthquake, Mar 2024 | -6.54, 112.05 | 20 km |
| Lewotobi Eruption | `lewotobi_eruption_2024.yaml` | Volcano, Nov 2024 | -8.50, 122.78 | 15 km |

---

## Algorithm

The routing pipeline has two stages: **path finding** (Dijkstra on edge weights) and **path ranking** (composite score). They are complementary — Dijkstra finds the best road path to each shelter, the composite score picks the best shelter.

### Stage 1 — Path Finding: Edge Weight (Dijkstra)

For each village, a single-source Dijkstra traversal finds the least-cost road path to every reachable shelter. The edge weight used by Dijkstra is:

```
weight(e) = length_m(e) × quality_weight(e) × (1.0 + risk_weight × risk_score(e))
```

- `length_m` — physical length of the road segment
- `quality_weight` — road type factor; faster roads get lower weight (motorway → 0.5, tertiary → 1.5, footway → 2.5), so Dijkstra prefers higher-quality roads even at equal distance
- `risk_score` — InaRISK composite hazard index [0, 1]; multiplied in as a penalty so Dijkstra naturally avoids high-risk segments while never making them impassable — all road-connected shelters remain reachable

The result is the geometrically optimal path (following the actual road network) from the village to each shelter.

### Stage 2 — Path Ranking: Composite Route Score (lower = better)

Once paths are found, each village-to-shelter route is scored to allow comparison across different shelters and to drive population assignment:

```
score = w_dist     × (total_distance / 50 km)
      + w_risk     × avg_edge_risk
      + w_quality  × (worst_edge_quality / 3)
      + w_time     × (total_travel_time / 2 h)
      + w_disaster × (1 − shelter_dist_from_disaster / max_shelter_dist)
```

The five weights are set per scenario in the YAML config under `routing:`:

| Weight | Config field | What it controls |
|---|---|---|
| `w_dist` | `weight_distance` | Penalises longer routes (normalised by 50 km) |
| `w_risk` | `weight_risk` | Penalises higher average InaRISK hazard score along the path |
| `w_quality` | `weight_road_quality` | Penalises poor road surface (normalised by worst quality factor of 3) |
| `w_time` | `weight_time` | Penalises longer travel time (normalised by 2 hours) |
| `w_disaster` | `weight_disaster_distance` | Penalises shelters close to the disaster centre — routes moving away from the hazard zone are preferred |

Weights do not need to sum to 1. Default values: `w_dist=0.25, w_risk=0.30, w_quality=0.20, w_time=0.10, w_disaster=0.15`.

Routes are ranked per village by composite score (ascending). The top-ranked route is the primary evacuation route; the next `min_routes_per_village − 1` are alternatives.

### Stage 3 — Population Assignment

Greedy algorithm: sort all (village, shelter, route) candidates by composite score, assign greedily while respecting remaining shelter capacity. Optional LP solver (`assignment_method: lp`) via SciPy HiGHS.

---

## Setup

### Requirements

- Python 3.14 (via pyenv)
- GAMA Platform 2025+ (for simulation, optional)
- OpenMPI or Intel MPI (for HPC mode, optional)
- Docker (for PostgreSQL+PostGIS wilayah backend — optional; SQLite is the portable default)

### Installation

```bash
pyenv virtualenv 3.14.2 respondor-evroute
pyenv local respondor-evroute
pip install -r requirements.txt

# Optional: MPI support
# macOS:  brew install open-mpi && pip install mpi4py
# Linux:  module load openmpi/4.1 && pip install mpi4py
```

### Wilayah DB (optional, for L9/L8 admin context in tooltips)

The wilayah PostGIS database ([Wilayah DB](https://github.com/cahyadsn/wilayah)) provides official Indonesian kelurahan/desa (L9) and kecamatan (L8) boundaries. Used to:
- Assign structured names (`C_33.10.03.2001_2`, `S_33.10.03.2001_1`) to building clusters
- Show L9/L8 breadcrumbs in map tooltips
- Fill uncovered L9 with synthetic clusters (`village_fill_uncovered_l9: true`)

**Option A — SQLite (portable, no Docker required):**

```bash
python -m docker.import_wilayah_sqlite   # outputs data/wilayah.db (~430 MB, ~40s)
```

`WilayahLoader` auto-discovers `data/wilayah.db` — no config needed. Override the path with `WILAYAH_SQLITE_PATH` env var or `WilayahLoader(sqlite_path=...)`.

**Option B — PostgreSQL+PostGIS (Docker):**

```bash
docker compose up -d
python docker/import_wilayah.py
```

---

## Usage

### Pre-run validation

```bash
python -m experiments.prerun_validation --config configs/banjarnegara_landslide_2021.yaml
```

Checks config validity, OSM/InaRISK cache coverage, estimates village count, API budget, and expected routing time before committing to a full run.

### Run a scenario

```bash
# Parallel (default)
python -m src.main --config configs/banjarnegara_landslide_2021.yaml

# Override workers
python -m src.main --config configs/merapi_eruption_2023.yaml --mode parallel --workers 12

# LP assignment (optimal, ~20 ms overhead vs greedy)
python -m src.main --config configs/banjarnegara_landslide_2021.yaml --assignment-method lp

# Limit villages for quick testing
python -m src.main --config configs/merapi_eruption_2023.yaml --village-limit 100

# HPC/MPI (multi-node)
srun --mpi=pmix -n 32 python -m src.main --config configs/palu_earthquake_2018.yaml --mode hpc --workers 32

# HPC/MPI local test
mpirun -n 2 python -m src.main --config configs/lewotobi_eruption_2024.yaml --mode hpc --workers 4
```

### Preview region and hazard before running

```bash
# OSM extraction preview — shows admin boundaries, building clusters, shelters, roads
python -m experiments.preview_region --config configs/banjarnegara_landslide_2021.yaml

# InaRISK hazard grid preview
python -m experiments.preview_hazard --config configs/banjarnegara_landslide_2021.yaml
```

### Export shapefiles

```bash
python -m experiments.export_shp --config configs/banjarnegara_landslide_2021.yaml
```

Outputs to `output/<scenario_id>/gis/`: `villages.shp`, `shelters.shp`, `roads.shp`, `villages.geojson`, `shelters.geojson`.

### Prepare GAMA simulation inputs

```bash
python -m experiments.prepare_gama_inputs --config configs/banjarnegara_landslide_2021.yaml
```

Outputs to `output/<scenario_id>/gama_inputs/`:

| File | Description |
|---|---|
| `all_routes.csv` | All ranked routes with WKT waypoints + village coordinates (pipe-delimited) |
| `villages.csv` | Village centroids, population, admin level, source, risk (pipe-delimited) |
| `shelters.csv` | Shelter centroids, capacity, type, risk; IDs as `way_XXXXXXX` (pipe-delimited) |
| `hazard_grid.csv` | InaRISK scores clipped to scenario bbox (pipe-delimited) |
| `region.shp` | Circular region boundary — loaded into GAMA world extent |
| `roads.shp` | Road network for display in GAMA |
| `villages.shp` | Village polygons for display in GAMA |
| `shelters.shp` | Shelter polygons for display in GAMA |
| `scenario.json` | Disaster coordinates, BPR parameters, simulation config |

All CSV files use `|` (pipe) as delimiter to avoid conflicts with WKT coordinate commas.

### Build preloaded / legacy input files

```bash
python -m experiments.build_legacy_input --config configs/banjarnegara_landslide_2021.yaml
```

Generates a `scenario_preloaded.yaml` that skips all OSM + InaRISK extraction on re-runs.

---

## Pipeline Outputs

All outputs written to `output/<scenario_id>/`:

| File | Description |
|---|---|
| `evacuation_map.html` | Interactive Folium map — village/shelter polygons, road-following routes, hazard grid, LayerControl with Primary/Alternative routes, per-shelter filter panel |
| `evacuation_results.csv` | Per-village assignment: shelter, population, distance, travel time, risk |
| `optimization_summary.json` | Aggregate KPIs: evacuation ratio, avg risk, avg distance, avg time, runtime, shelter utilisation |
| `routes.csv` | All candidate routes ranked per village (village × shelter × rank) |
| `routes_summary.json` | Same as routes.csv in JSON |
| `graph_stats.json` | Edge count, risk distribution histogram (zero / very_low / … / very_high) |
| `evacuation_summary.png` | Coverage pie chart + shelter utilisation bar chart |
| `timings_naive.json` | Stage-by-stage runtime + per-phase memory delta (naive mode) |
| `timings_parallel_Nw.json` | Stage-by-stage runtime + per-phase memory delta (parallel mode, N workers) |
| `timings_hpc_Nr_Mw.json` | Stage-by-stage runtime + per-phase memory delta (HPC/MPI mode, N ranks × M workers) |
| `gama_inputs/` | GAMA simulation inputs — generated separately by `prepare_gama_inputs.py` |
| `gama_shp/` | SHP exports — generated separately by `export_shp.py` |
| `route_comparison/` | Side-by-side OSM vs legacy comparison — generated by `compare_routes.py` |

---

## Interactive Map (`evacuation_map.html`)

The map includes a persistent LayerControl (always expanded) with independent toggles for:

- **Building Clusters** — real DBSCAN settlement polygons, coloured by population quintile
- **Synthetic Clusters** — L9 kelurahan with no OSM buildings (centroid circles)
- **Shelters** — equivalent-area circles coloured by capacity quintile
- **Primary Routes** — rank-1 routes (one per village), green
- **Alternative Routes** — rank 2–3 routes, orange / red
- **Hazard Grid** — InaRISK rectangles (~1 km grid, 5-colour ramp, off by default)
- **Risk Heatmap** — village risk heatmap layer (off by default)

A **Shelter Filter Panel** (bottom-right, minimisable) provides per-shelter checkboxes that simultaneously hide/show:
1. That shelter's polygon on the map
2. All primary and alternative routes leading to that shelter

The panel is composable with the LayerControl — filtering a shelter via the panel and then toggling "Primary Routes" off/on via the LayerControl behaves correctly.

Tooltips show:
- **Villages**: `C_/S_[L9-kode]_N` display name, kelurahan / kecamatan breadcrumb, area (m²), population, risk
- **Shelters**: name, type, kelurahan / kecamatan breadcrumb, area (m²), capacity, risk
- **Routes**: rank label, village → shelter, distance, travel time, avg risk, composite score

---

## InaRISK API and Caching

Hazard data is sourced from BNPB's [InaRISK](https://inarisk.bnpb.go.id) service via the ImageServer identify endpoint.

**Supported hazard layers:**

| Layer | InaRISK service |
|---|---|
| `volcano` | INDEKS_BAHAYA_GUNUNGAPI |
| `landslide` | INDEKS_BAHAYA_TANAHLONGSOR |
| `flood` | INDEKS_BAHAYA_BANJIR |
| `tsunami` | INDEKS_BAHAYA_TSUNAMI |
| `liquefaction` | INDEKS_BAHAYA_LIKUEFAKSI |
| `flash_flood` | INDEKS_BAHAYA_BANJIRBANDANG |
| `earthquake` | INDEKS_BAHAYA_GEMPABUMI (currently down) |

**Compound hazard** scenarios (e.g. Palu) configure multiple layers with weights:

```yaml
routing:
  hazard_layers:
    earthquake:   0.5
    tsunami:      0.3
    liquefaction: 0.2
  hazard_aggregation: weighted_sum   # or max
```

**Grid-snap caching** eliminates most API calls on subsequent runs. Road edges are enriched at ~1.1 km grid precision; village/shelter POI points snap to the same grid before any API call is attempted. On Banjarnegara (1 159 villages), this reduces 1 159 API calls to ~16 after the first run.

Cache files:

| File | Contents |
|---|---|
| `data/raw/inarisk_cache/road_risk_cache.json` | `{layer_key: {"lat2,lon2": score}}` — edge grid |
| `data/raw/inarisk_cache/poi_risk_cache.json` | `{cache_key: {"lat6,lon6": score}}` — village/shelter POI |
| `data/raw/inarisk_cache/hazard_grid_cache.json` | `{hazard_type: {"lat6,lon6": score}}` — preview grid |

Use `skip_inarisk: true` in config for fully offline testing (all risk scores set to 0.0).

---

## Village Extraction Sources

Villages can be extracted from multiple OSM sources, composable in any order:

| Source | What it uses | Best for |
|---|---|---|
| `admin_boundary` | `boundary=administrative` closed polygons at configured admin levels | Java, Sumatra — well-mapped admin data |
| `place_nodes` | `place=village\|hamlet\|...` point nodes → synthetic circles | Remote islands, highlands with sparse admin data |
| `building_clusters` | DBSCAN-grouped building footprints → convex hull polygons | Dense areas with good building traces but no boundaries |
| `wilayah_db` | Official L9 kelurahan polygons from wilayah DB (SQLite or PostGIS) | Complete coverage with structured naming |

**Synthetic clusters** (`village_fill_uncovered_l9: true`): when a kelurahan has no OSM building data, a single synthetic circular cluster is placed at the kelurahan centroid, named `S_[L9-kode]_1`.

**Shelter clustering**: nearby shelter polygons are merged into a single circular destination (`shelter_cluster_eps_m` diameter). All shelter geometries are stored as equivalent-area circles (in UTM, reprojected to WGS84) regardless of original OSM shape.

---

## Configuration Reference

Full annotated examples: [configs/](configs/)

### Key fields

| Field | Default | Description |
|---|---|---|
| `disaster.type` | — | `volcano` `earthquake` `flood` `landslide` `tsunami` `liquefaction` `flash_flood` |
| `region.type` | `circle` | `circle` or `bbox` |
| `region.radius_km` | — | Radius for circle region |
| `skip_inarisk` | `false` | Set `true` to bypass all InaRISK API calls (offline testing) |
| `extraction.use_cached_osm` | `true` | Skip OSM HTTP download if cache exists |
| `extraction.use_cached_inarisk` | `true` | Skip InaRISK API if grid/POI cache covers all points |
| `extraction.village_sources` | `[building_clusters]` | Ordered village extraction sources |
| `extraction.village_cluster_eps_m` | `300` | DBSCAN cluster diameter in metres (use `100` for highland Java) |
| `extraction.village_cluster_max_area_km2` | `25.0` | Skip degenerate large clusters |
| `extraction.village_fill_uncovered_l9` | `false` | Add synthetic clusters for uncovered L9 (requires wilayah DB) |
| `extraction.shelter_cluster_eps_m` | `250` | Shelter merge diameter in metres (use `200`) |
| `extraction.village_pop_density` | `800` | Fallback population density (persons/km²) |
| `routing.weight_distance` | `0.25` | Distance term weight in composite score |
| `routing.weight_risk` | `0.30` | Risk term weight |
| `routing.weight_road_quality` | `0.20` | Road quality term weight |
| `routing.weight_time` | `0.10` | Travel time term weight |
| `routing.weight_disaster_distance` | `0.15` | Penalty for shelters near disaster centre |
| `routing.max_routes_per_village` | `5` | Hard cap on routes per village |
| `routing.min_routes_per_village` | `3` | Guaranteed floor (1 primary + 2 alternatives) |
| `routing.hazard_layers` | `{}` | Compound hazard: `{volcano: 0.6, earthquake: 0.4}` |
| `routing.hazard_aggregation` | `weighted_sum` | `weighted_sum` or `max` |
| `routing.assignment_method` | `greedy` | `greedy` (fast) or `lp` (optimal, requires scipy); overridable via `--assignment-method` |
| `execution.mode` | `parallel` | `naive` `parallel` `hpc`; overridable via `--mode` |
| `execution.n_workers` | `4` | Worker count for parallel mode; overridable via `--workers` |
| `preloaded_villages_geojson` | — | Skip extraction — use pre-extracted GeoJSON |
| `preloaded_shelters_geojson` | — | Skip extraction — use pre-extracted GeoJSON |
| `preloaded_network_json` | — | Skip extraction — use pre-extracted network JSON |
| `benchmark_village_limit` | `0` | Limit villages for speed testing (`0` = no limit) |

---

## HPC Execution (MPI + SLURM)

```bash
# Local test — 2 MPI ranks × 4 workers = 8 cores
mpirun -n 2 python -m src.main --config configs/demak_flood_2024.yaml --mode hpc --workers 4

# SLURM cluster
sbatch hpc/slurm_job.sh configs/demak_flood_2024.yaml

# SLURM with hybrid parallelism (4 nodes × 32 cores)
srun --mpi=pmix -n 4 python -m src.main --config configs/palu_earthquake_2018.yaml --mode hpc --workers 32
```

- **Rank 0**: loads OSM data, builds weighted graph, queries InaRISK, broadcasts to all ranks
- **Ranks 1..N**: receive a village partition (round-robin), compute Dijkstra routes independently
- **Intra-rank parallelism**: each rank processes its partition using `ProcessPoolExecutor` with `--workers` processes (spawn method — avoids MPI + fork deadlocks on Linux)
- **Gather**: all routes collected at rank 0 for assignment and output

**Total cores**: `n MPI ranks × --workers processes per rank`. Example: `mpirun -n 4 --workers 32` = 128 cores.

**Fallback:** If `mpi4py` is not installed, HPC mode automatically uses `ProcessPoolExecutor` on a single node.

---

## Benchmarking

Two benchmark scripts measure pipeline performance across execution modes and scenarios.

### Full pipeline benchmark (`benchmark_all.py`)

Runs all scenarios × all execution modes, recording per-phase timing and peak memory consumption.

```bash
# All scenarios, all modes (warm cache recommended — run benchmark_extraction first)
python -m experiments.benchmark_all

# Specific scenarios only
python -m experiments.benchmark_all --scenarios banjarnegara_landslide_2021 merapi_eruption_2023

# Custom mode subset
python -m experiments.benchmark_all --modes naive parallel_4w parallel_8w hpc_2r_8w

# Quick test — limit villages per scenario
python -m experiments.benchmark_all --scenarios banjarnegara_landslide_2021 \
    --parallel-workers 2 4 8 --hpc-ranks 2 --hpc-workers 4 8 --village-limit 100

# Resume interrupted run (skip completed entries)
python -m experiments.benchmark_all --resume

# Dry-run — print commands without executing
python -m experiments.benchmark_all --dry-run
```

**Default execution matrix** (13 modes × N scenarios):

| Category | Variants |
|---|---|
| Naive | 1 worker |
| Parallel | 2, 4, 8, 16, 32, 64 workers |
| HPC (MPI) | ranks ∈ {2, 4} × workers ∈ {8, 16, 64} |

**Outputs** (written to `output/`):

| File | Description |
|---|---|
| `benchmark_results.json` | All run entries: timings, memory, mode metadata |
| `benchmark_results.csv` | Flat CSV for analysis (one row per run) |

### Extraction-only benchmark (`benchmark_extraction.py`)

Benchmarks OSM extraction and InaRISK risk scoring in isolation — skips graph build, routing, and assignment. Use this to pre-warm caches on a new server before running `benchmark_all`.

```bash
# Pre-warm all caches for all scenarios (fresh download)
python -m experiments.benchmark_extraction --no-cache --skip-inarisk-edges

# Also pre-warm InaRISK road edge cache
python -m experiments.benchmark_extraction --no-cache

# Compare sequential vs parallel OSM extraction
python -m experiments.benchmark_extraction --osm-modes sequential parallel

# Compare InaRISK thread counts (use --no-cache-inarisk for real API throughput)
python -m experiments.benchmark_extraction --inarisk-threads 1 2 4 8 --no-cache-inarisk

# Full matrix: 2 OSM modes × 4 thread counts
python -m experiments.benchmark_extraction \
    --osm-modes sequential parallel --inarisk-threads 1 2 4 8 --no-cache-inarisk
```

**Output:** `output/benchmark_extraction.json`

### Recommended workflow for a new server

```bash
# 1. Pre-warm OSM + InaRISK POI caches for all scenarios
python -m experiments.benchmark_extraction --no-cache --skip-inarisk-edges

# 2. Run full benchmark (InaRISK edge cache warmed by the first scenario's first run)
python -m experiments.benchmark_all --resume
```

---

## GAMA Simulation

```bash
# 1. Run main pipeline (or use cached outputs)
python -m src.main --config configs/banjarnegara_landslide_2021.yaml

# 2. (Optional) Export road shapefile for GAMA road-following display
python -m experiments.export_shp --config configs/banjarnegara_landslide_2021.yaml

# 3. Generate all GAMA inputs (reads cache, no new API calls)
python -m experiments.prepare_gama_inputs --config configs/banjarnegara_landslide_2021.yaml
```

Open `simulation/models/EvacuationModel.gaml` in GAMA Platform 2025+ and set `inputs_dir` to the path printed by `prepare_gama_inputs`. The model implements four evacuation strategies selectable via a dropdown in the GUI experiment:

| Mode | Label | Strategy |
|---|---|---|
| 1 | Nearest Shelter | Road-graph Dijkstra to nearest shelter (static target) |
| 2 | Aware Evacuation | Road-graph routing with hazard-zone rerouting when within awareness buffer |
| 3 | Pipeline Best Route | Follow pre-computed rank-1 waypoints from the Python pipeline |
| 4 | Pipeline Random Route | Pick rank 1, 2, or 3 route randomly per agent at initialisation |

**Key model features:**
- **BPR congestion**: road speed adjusted per segment as `base_speed / (1 + α × (flow/capacity)^β)`; α=0.15, β=4.0
- **Hazard propagation**: circular zone expands from disaster centre at `hazard_speed_kmh`; configurable start delay
- **Shelter closure**: shelters engulfed by the hazard zone stop accepting arrivals; agents redirect to nearest safe shelter or die
- **Evacuation time tracking**: per-agent arrival time recorded; avg and worst-case reported at termination
- **Batch experiment** (`BatchCompareModes`): sweeps modes 1–4 × hazard speeds 1 / 2 / 5 km/h automatically

**Verified working** (Banjarnegara scenario): 31 shelters, 5,214 route rows, 1,054 evacuee groups, 131,747 total population.

---

## References

- [InaRISK BNPB](https://inarisk.bnpb.go.id) — Indonesia national hazard risk data
- [osmnx](https://github.com/gboeing/osmnx) — OpenStreetMap network extraction
- [GAMA Platform](https://gama-platform.org) — Agent-based modelling for evacuation simulation
- [OsmToRoadGraph / PYCGR format](https://github.com/AndGem/OsmToRoadGraph) — Legacy network file format
- [BPR function](https://en.wikipedia.org/wiki/Bureau_of_Public_Roads) — Traffic congestion model
- [Wilayah DB](https://github.com/cahyadsn/wilayah) - Wilayah Indonesia
