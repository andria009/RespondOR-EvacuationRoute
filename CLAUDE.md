# RespondOR-EvacuationRoute — Project Context for Claude

This file is read automatically by Claude Code on every session. It contains all architectural context, design decisions, known bugs, and workflow instructions needed to resume work on this project without asking the user to re-explain.

---

## Project Overview

**RespondOR-EvacuationRoute** is a disaster evacuation route optimization system for Indonesia. It:
1. Extracts settlement clusters (villages) and shelters from OpenStreetMap
2. Builds a road network graph with composite edge weights (distance × quality × risk)
3. Enriches edges and POIs with InaRISK BNPB hazard risk scores
4. Computes optimized evacuation routes (primary + alternatives) using Dijkstra on the road graph
5. Assigns village populations to shelters using a greedy optimizer
6. Outputs route CSVs, summary JSONs, and an interactive Folium HTML map

**Research goal:** Academic publication on multi-hazard evacuation route optimization for Indonesian disaster scenarios.

**Python environment:** pyenv, Python 3.14. Run with `python -m src.main --config configs/<scenario>.yaml`

---

## Directory Structure

```
src/
  config/config_loader.py      — AppConfig dataclass + YAML loader
  data/
    models.py                  — Village, Shelter, EvacuationRoute, DisasterType dataclasses
    osm_extractor.py           — OSMnx-based road/village/shelter extraction with partial-cache resume
    inarisk_client.py          — InaRISK BNPB API client with grid-snap + batched caching
    wilayah_loader.py          — Local wilayah DB loader for L9 kelurahan reference
    population_loader.py       — CSV population override + density-based estimation
  graph/graph_builder.py       — NetworkX graph build + InaRISK edge enrichment
  routing/
    heuristic_optimizer.py     — Dijkstra-based route computation (parallel + sequential)
    assignment.py              — Greedy/LP population-to-shelter assignment
  hpc/
    naive_runner.py            — Sequential (single-process) pipeline runner
    parallel_runner.py         — Multiprocessing pipeline runner (default)
    distributed_runner.py      — MPI/HPC distributed runner
  visualization/visualizer.py  — Folium interactive map + PNG chart + CSV export
  utils/logging_setup.py       — File+console logging to logs/ directory
  main.py                      — CLI entrypoint

configs/                       — One YAML per scenario (see Scenarios section)
experiments/
  prerun_validation.py         — Pre-run check: config, cache coverage, API budget, time estimate
  export_shp.py                — Export villages/shelters to SHP + GeoJSON for QGIS
  prepare_gama_inputs.py       — Generate GAMA ABM simulation input files
  build_legacy_input.py        — Build RespondOR v1 legacy CSV format
  compare_routes.py            — Side-by-side route comparison across scenarios
  preview_hazard.py            — Interactive hazard grid map for a scenario
  preview_region.py            — Preview OSM extraction bounding box

data/
  raw/osm_cache/               — OSMnx + OSM query cache files (bbox-keyed JSON)
  raw/inarisk_cache/           — poi_risk_cache.json + road_risk_cache.json

output/<scenario_id>/          — Pipeline outputs per scenario
logs/                          — Timestamped log files (auto-created)
```

---

## Key Algorithms & Design Decisions

### Edge Weight Formula
```
weight = length_m × quality_weight × (1.0 + risk_weight × risk_score)
```
- `quality_weight`: ratio of motorway_speed / actual_speed (faster roads = lower weight)
- `risk_score`: InaRISK composite [0, 1] from one or more hazard layers
- `risk_weight`: defaults to 0.4 in `EvacuationGraphBuilder`

### Composite Route Score (lower = better route)
```
score = w_dist    × (dist / 50_000)
      + w_risk    × avg_edge_risk
      + w_quality × (worst_quality / 3.0)
      + w_time    × (time_s / 7200)
      + w_disaster × (1 - shelter_dist_from_disaster / max_shelter_dist)
```
`weight_disaster_distance` penalizes shelters close to the disaster center — routes moving *away* from the hazard zone are preferred.

### DBSCAN Cluster Geometry
- `village_cluster_eps_m` = cluster diameter (not radius). A single building cluster has area ≈ π×(eps/2)²
- At eps=100m: single-building cluster area ≈ 7,854 m² (~0.8 ha). At eps=150m: 17,671 m².
- Recommended: `village_cluster_eps_m: 100`, `shelter_cluster_eps_m: 200`

### InaRISK Grid-Snap Enrichment
- Road edges are enriched at `grid_precision=2` decimal places (~1.1 km grid), results cached in `road_risk_cache.json`
- Village/shelter POI enrichment **first checks `road_risk_cache.json`** (snapped to same grid) before making API calls — eliminates API calls on second runs for the same region
- If grid doesn't cover a point, falls back to direct per-point API query
- API rate: ~1 req/s per point; use `skip_inarisk: true` for offline testing

### Min Routes Guarantee
- `min_routes_per_village: 3` = 1 primary + 2 alternatives, guaranteed
- If strict risk threshold yields fewer routes, it relaxes to 0.90 → 0.95 → 1.0 progressively

### Resumability (Partial Cache)
All long-running extraction steps save partial progress so a killed process can resume:
- **Shelter extraction**: saves `.partial.json` after each OSM tag query
- **Village extraction**: saves after each source (building_clusters, admin_boundary, etc.)
- **InaRISK POI**: saves after every API batch (20 points by default)

---

## Configuration Reference

Key fields in `ExtractionConfig`:
| Field | Default | Notes |
|-------|---------|-------|
| `village_cluster_eps_m` | 300 | DBSCAN diameter; use **100** for highland Java |
| `shelter_cluster_eps_m` | 250 | Use **200** |
| `village_fill_uncovered_l9` | false | Requires local wilayah PostgreSQL DB |
| `use_cached_osm` | true | Set false to force re-query after OSM changes |
| `use_cached_inarisk` | true | Set false to force re-query InaRISK API |

Key fields in `RoutingConfig`:
| Field | Default | Notes |
|-------|---------|-------|
| `max_routes_per_village` | 5 | Hard cap |
| `min_routes_per_village` | 3 | Guaranteed floor; relaxes risk threshold if needed |
| `weight_disaster_distance` | 0.05 | Penalty for shelters near disaster center |
| `hazard_layers` | {} | Multi-hazard: `{volcano: 0.6, earthquake: 0.4}` |

Key fields in `ExecutionConfig`:
| Field | Default | Notes |
|-------|---------|-------|
| `mode` | parallel | `naive` / `parallel` / `hpc` |
| `n_workers` | 4 | Ignored in naive and hpc modes |

Top-level config:
| Field | Notes |
|-------|-------|
| `skip_inarisk` | true = all risk scores 0.0, skip all API calls |
| `benchmark_village_limit` | Limit villages for speed testing (0 = no limit) |

---

## Scenarios

| File | Disaster | Type | Radius | Status |
|------|----------|------|--------|--------|
| `banjarnegara_landslide_2021.yaml` | Banjarnegara Landslide Feb 2021 | landslide | 10 km | ✅ Completed |
| `sukabumi_landslide_2024.yaml` | Sukabumi/Bogor Landslides Dec 2024 | landslide | 18 km | ✅ Completed |
| `merapi_eruption_2023.yaml` | Merapi Eruption 2023 | volcano | 15 km | ⚠️ Routing slow (use parallel, n_workers≥10) |
| `sumedang_landslide_2021.yaml` | Sumedang Landslide Jan 2021 | landslide | 10 km | Not yet run |
| `cianjur_earthquake_2022.yaml` | Cianjur Earthquake Nov 2022 | earthquake | 20 km | Not yet run |
| `palu_earthquake_2018.yaml` | Palu Earthquake+Tsunami 2018 | compound | 20 km | Not yet run |
| `sinabung_eruption_2016.yaml` | Sinabung Eruption 2016 | volcano | 15 km | Not yet run |
| `tuban_earthquake_2024.yaml` | Tuban Earthquake 2024 | earthquake | 20 km | Not yet run |
| `demak_flood_2024.yaml` | Demak Flood 2024 | flood | 25 km | Not yet run |
| `lewotobi_eruption_2024.yaml` | Lewotobi Eruption 2024 | volcano | 15 km | Not yet run |

---

## How to Run

```bash
# Pre-run validation (check config, cache, API budget)
python -m experiments.prerun_validation --config configs/<scenario>.yaml

# Full pipeline
python -m src.main --config configs/<scenario>.yaml

# Override mode/workers
python -m src.main --config configs/<scenario>.yaml --mode parallel --workers 8

# Visualization only (from existing outputs)
python -m src.main --config configs/<scenario>.yaml --visualize-only

# Hazard grid preview
python -m experiments.preview_hazard --config configs/<scenario>.yaml

# Export SHP/GeoJSON for QGIS
python -m experiments.export_shp --config configs/<scenario>.yaml
```

---

## Known Bugs Fixed (do not re-introduce)

### NaN Speed Bug
`float("nan")` bypasses try/except in speed parsing. `max(nan, x)` returns `nan`. Guard explicitly in speed parsing and any model property that reads speed. See `src/data/models.py`.

### Stale .pyc Bytecode
After editing `src/graph/graph_builder.py` or other core files, Python may use cached bytecode. If a fix doesn't seem to apply, delete `src/graph/__pycache__/graph_builder*.pyc` (or relevant module's `__pycache__`).

### OSM Cache Picking Wrong File
`_latest_geojson()` (mtime-based) in old experiment scripts picked the most-recently-modified file regardless of scenario. All experiment scripts now use `OSMExtractor.extract_villages/extract_shelters()` with bbox-keyed cache. Never use mtime-based file selection.

### Hazard Preview Bleeding Across Scenarios
`query_hazard_grid()` returns the full shared cache dict. In `preview_hazard.py`, always apply a bbox filter after loading the cache to clip to the current scenario's region.

### Tuban osmnx Empty Cache
`use_cached_osm=true` can serve a stale empty network if osmnx had a prior failed query. Set `use_cached_osm: false` and re-run to force a fresh OSM download for any scenario where extraction returns 0 nodes.

### parallel_runner `default_pop_density`
`cfg.extraction.default_pop_density` does not exist — the field is `cfg.extraction.village_pop_density`. Any new runner code must use the correct field name.

### Edge Pruning Disconnects Graph (villages with no routes)
`prune_impassable=True` in `graph_builder.build()` removed edges with risk > 0.9, fragmenting the graph into isolated components — villages in disconnected components got zero routes. **Default changed to `prune_impassable=False`**. Risk is penalized through composite edge weights, not by deletion. Never set this True for evacuation scenarios.

### Shelter Geometries Were Rectangular (OSM building shapes)
`_cluster_shelters` was using `unary_union` of raw OSM building polygons, producing rectangular/irregular shapes on the map. **Fixed**: all shelter geometries (single and merged) are now replaced with a circle of equivalent area, computed in UTM and reprojected to WGS84 (64-segment approximation). Delete the scenario's shelter cache file to re-extract with circular geometries.

### Route Risk Threshold as Hard Filter (villages with no routes)
`max_route_risk_threshold` was used as a hard filter — any route whose worst-edge risk exceeded it was discarded entirely. This caused villages near high-risk zones to get zero routes. **Fixed**: the threshold is removed from routing; risk is scored via `weight_risk` in composite score instead. All road-reachable shelters are scored and ranked, never filtered out.

### node_coords from Mtime-Based Network JSON (straight-line routes on map)
`_load_viz_extras` was selecting the most-recently-modified `network_*.json` to build node coordinates for route polylines. In multi-scenario setups this picked the wrong file, causing routes to render as straight lines. **Fixed**: `node_coords` is now read directly from `G.nodes(data=True)` (the live graph) and passed via `_load_viz_extras(config, villages, shelters, G=G)`.

---

## Cache File Locations

```
data/raw/osm_cache/
  network_<bbox_hash>.json              — Road network (nodes + edges)
  villages_<bbox+params hash>.json      — Extracted villages
  villages_<...>.partial.json           — In-progress village extraction
  shelters_<bbox+params hash>.json      — Extracted shelters
  shelters_<...>.partial.json           — In-progress shelter extraction

data/raw/inarisk_cache/
  poi_risk_cache.json                   — {cache_key: {"lat6,lon6": score}}
                                          cache_key examples: "villages_landslide",
                                          "shelters_volcano", "villages_composite"
  road_risk_cache.json                  — {layer_key: {"lat2,lon2": score}}
                                          layer_key examples: "edges_landslide"
                                          Keys are rounded to 2 decimal places (~1.1 km grid)
```

The `road_risk_cache.json` doubles as a grid lookup for POI enrichment. On the first run of a new region it will be empty; API calls are made. On subsequent runs, POI points snap to the existing grid and API calls are skipped.

---

## Output Files (per scenario)

```
output/<scenario_id>/
  optimization_summary.json     — KPIs: evacuation ratio, avg risk, distance, time
  routes.csv                    — All candidate routes (village × shelter × rank)
  routes_summary.json           — Same as CSV in JSON
  graph_stats.json              — Edge count, risk distribution histogram
  evacuation_results.csv        — Per-village: assigned shelter, pop, distance
  evacuation_map.html           — Interactive Folium map (routes + polygons + layers)
  evacuation_summary.png        — Coverage bar chart
  timings_naive.json            — Stage-by-stage runtimes (naive mode)
  timings_parallel.json         — Stage-by-stage runtimes (parallel mode)
```

---

## Pipeline Stage Order

1. **Extract** — road network, villages, shelters (OSM + osmnx)
2. **Risk scoring** — InaRISK POI enrichment (villages + shelters)
3. **Graph build** — NetworkX graph + InaRISK edge enrichment + propagate POI risk
4. **Routing** — Dijkstra per village → all shelters, composite score ranking, min_routes guarantee
5. **Assignment** — Greedy/LP population assignment to shelters
6. **Outputs** — JSON/CSV stats + Folium map

---

## Dependencies

Core: `osmnx`, `networkx`, `shapely`, `folium`, `requests`, `scikit-learn` (DBSCAN), `pyyaml`, `scipy` (LP assignment)

Optional: `geopandas` (export_shp), `psycopg2` (wilayah DB for fill_uncovered_l9)

Install: `pip install -r requirements.txt`

---

## GAMA Simulation Integration

After running the pipeline, generate GAMA ABM input files:
```bash
python -m experiments.prepare_gama_inputs --config configs/<scenario>.yaml
```
GAMA model: `misc/EvacuationModel.gaml` (rewritten for pipeline handoff input format).
Input format: building clusters CSV with centroid, population, assigned shelter, and route geometry.
