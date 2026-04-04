# RespondOR-EvacuationRoute

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research-grade disaster evacuation route optimization system for Indonesia. Integrates OpenStreetMap road networks, InaRISK BNPB hazard data, and multi-objective routing to maximize the number of people evacuated to safe shelters. Designed for four disaster types — flood, landslide, earthquake, and volcano — with validated scenarios drawn from major 2024 Indonesian disaster events.

---

## Overview

Given a disaster event, the system:

1. **Extracts** road network, villages, and shelters from OpenStreetMap using configurable multi-source village extraction (admin boundaries, place nodes, building clusters)
2. **Scores hazard risk** via the InaRISK BNPB API for each village, shelter, and road segment
3. **Builds** a weighted evacuation graph with composite edge weights (distance × road quality × risk penalty)
4. **Computes** optimized routes from each village to candidate shelters using Dijkstra's algorithm
5. **Assigns** village populations to shelters using greedy optimization, maximizing total evacuated population subject to shelter capacity constraints
6. **Exports** results as interactive HTML maps, SHP files (for GIS / GAMA Platform), and CSV for analysis

---

## Repository Structure

```
RespondOR-EvacuationRoute/
├── configs/
│   ├── demak_flood_2024.yaml           # Demak Regency flood, Feb 2024
│   ├── sukabumi_landslide_2024.yaml    # Cisolok landslide, Dec 2024
│   ├── tuban_earthquake_2024.yaml      # Tuban M6.0 earthquake, Mar 2024
│   └── lewotobi_eruption_2024.yaml     # Lewotobi Laki-Laki eruption, Nov 2024
├── data/
│   └── raw/osm_cache/                  # Cached OSM extracts (auto-populated per scenario)
├── experiments/
│   ├── export_shp.py                   # Export villages/shelters/roads as SHP + preview map
│   ├── prepare_gama_inputs.py          # Write GAMA simulation inputs (CSV + SHP + JSON)
│   ├── build_legacy_input.py           # Build preloaded/legacy input files from OSM cache
│   ├── compare_routes.py               # Compare OSM vs legacy route outputs
│   └── scenarios/                      # Per-scenario experiment notebooks/scripts
├── hpc/
│   └── slurm_job.sh                    # SLURM job script (4 nodes × 8 MPI ranks)
├── output/                             # Pipeline outputs (auto-generated, per scenario)
│   ├── demak_flood_2024/
│   ├── sukabumi_landslide_2024/
│   ├── tuban_earthquake_2024/
│   └── lewotobi_eruption_2024/
├── simulation/
│   └── models/EvacuationModel.gaml     # GAMA agent-based evacuation model
├── src/
│   ├── config/config_loader.py         # YAML config loader + dataclasses
│   ├── data/
│   │   ├── inarisk_client.py           # InaRISK BNPB API client (with caching)
│   │   ├── models.py                   # Core dataclasses (Node, Edge, Village, Shelter, Route)
│   │   ├── osm_extractor.py            # OSM extraction (osmnx 2.x) + multi-source village extraction
│   │   └── population_loader.py        # Population and shelter capacity estimation
│   ├── graph/graph_builder.py          # Weighted NetworkX evacuation graph
│   ├── hpc/
│   │   ├── distributed_runner.py       # MPI runner (mpi4py; fallback: ProcessPool)
│   │   ├── naive_runner.py             # Sequential pipeline
│   │   └── parallel_runner.py          # ThreadPool (I/O) + ProcessPool (routing)
│   ├── routing/
│   │   ├── assignment.py               # Greedy population-to-shelter assignment
│   │   └── heuristic_optimizer.py      # Dijkstra-based route computation and scoring
│   ├── visualization/visualizer.py     # Folium interactive map + matplotlib charts
│   └── main.py                         # CLI entry point
└── tests/                              # pytest test suite
```

---

## Scenarios

Four scenarios based on major 2024 Indonesian disaster events:

| Scenario | Config | Disaster | Center | Radius |
|---|---|---|---|---|
| Demak Flood | `demak_flood_2024.yaml` | Flood, Feb 2024 | -6.894, 110.633 | 25 km |
| Sukabumi Landslide | `sukabumi_landslide_2024.yaml` | Landslide, Dec 2024 | -7.123, 106.727 | 18 km |
| Tuban Earthquake | `tuban_earthquake_2024.yaml` | M6.0 Earthquake, Mar 2024 | -6.538, 112.045 | 30 km |
| Lewotobi Eruption | `lewotobi_eruption_2024.yaml` | Volcano, Nov 2024 | -8.503, 122.775 | 15 km |

> Scenarios are under active development. Extraction parameters, shelter tags, and routing weights are calibrated per event. See individual config files for full settings.

---

## Algorithm

### Edge Weight (composite)

```
w(e) = distance(e) × quality_weight(e) × (1 + risk_weight × risk_score(e))
```

- `quality_weight`: road type factor (motorway=0.5 → footway=2.5)
- `risk_score`: InaRISK hazard index normalized to [0, 1]
- Edges with `risk_score > max_route_risk_threshold` are pruned as impassable

### Route Composite Score (lower = better)

```
score = w_dist × (dist/50km) + w_risk × avg_risk + w_quality × (worst_quality/3) + w_time × (time/2h)
```

Default weights: `w_dist=0.30, w_risk=0.40, w_quality=0.20, w_time=0.10` (scenario-tunable)

### Population Assignment

Greedy algorithm: sort all (village, shelter, route) candidates by composite score, assign greedily while tracking remaining shelter capacity and unmet village demand.

---

## Setup

### Requirements

- Python 3.14 (via pyenv)
- GAMA Platform 2025.6.4 (for simulation, optional)
- OpenMPI or Intel MPI (for HPC mode, optional)

### Installation

```bash
pyenv virtualenv 3.14.2 respondor-evroute
pyenv local respondor-evroute
pip install -r requirements.txt

# Optional: MPI support
# macOS:  brew install open-mpi && pip install mpi4py
# Linux:  module load openmpi/4.1 && pip install mpi4py
```

---

## Usage

### Run a scenario pipeline

```bash
# Sequential (default)
python -m src.main --config configs/demak_flood_2024.yaml

# Parallel (multiprocessing, single node)
python -m src.main --config configs/demak_flood_2024.yaml --mode parallel --workers 8

# HPC/MPI (multi-node)
mpirun -n 8 python -m src.main --config configs/demak_flood_2024.yaml --mode hpc
```

### Export shapefiles and preview map

```bash
python -m experiments.export_shp --config configs/demak_flood_2024.yaml
```

Outputs to `output/<scenario_id>/gama_shp/`:
- `villages.shp` — village polygons with population quintile class
- `shelters.shp` — shelter polygons with capacity quintile class and InaRISK risk score
- `roads.shp` — road network with speed, capacity, highway type, and InaRISK risk score
- `preview.html` — interactive map matching the pipeline visualization style

### Prepare GAMA simulation inputs

```bash
python -m experiments.prepare_gama_inputs --config configs/demak_flood_2024.yaml
```

Outputs to `output/<scenario_id>/gama_inputs/`:
- `villages.csv` — village agents with population, risk, and source
- `shelters.csv` — shelter agents with capacity and assigned load
- `routes.csv` — assigned routes with waypoints as WKT LINESTRING
- `scenario.json` — disaster parameters and simulation hints
- `roads.shp` — road network for GAMA road-following movement

### Build preloaded / legacy input files

```bash
python -m experiments.build_legacy_input --config configs/demak_flood_2024.yaml
```

Outputs to `output/<scenario_id>/legacy_input/` — network JSON, PYCGR, village/shelter GeoJSON, POI CSV, and a `scenario_preloaded.yaml` that skips all extraction on re-run.

### Compare OSM vs legacy routes

```bash
python -m experiments.compare_routes --config configs/demak_flood_2024.yaml
```

---

## Pipeline Outputs

All outputs written to `output/<scenario_id>/`:

| File | Description |
|---|---|
| `evacuation_map.html` | Interactive Folium map — village/shelter polygons by quintile, actual road-path routes, risk heatmap toggle |
| `evacuation_results.csv` | Per-village assignment: shelter, population, distance, travel time, risk |
| `optimization_summary.json` | Aggregate metrics: evacuation ratio, avg risk, runtime, shelter utilization |
| `timings_naive.json` | Stage-by-stage runtime breakdown |
| `evacuation_summary.png` | Population and evacuation bar chart |
| `gama_shp/` | SHP export + preview map (run `export_shp`) |
| `gama_inputs/` | GAMA simulation inputs (run `prepare_gama_inputs`) |
| `legacy_input/` | Preloaded input files (run `build_legacy_input`) |

---

## Village Extraction Sources

Villages can be extracted from three OSM sources, composable in any order. Each source adds only settlements not already covered by a polygon from a previous source.

| Source | What it uses | Best for |
|---|---|---|
| `admin_boundary` | `boundary=administrative` closed polygons | Java, Sumatra — well-mapped admin data |
| `place_nodes` | `place=village\|hamlet\|...` point nodes → synthetic circles | Remote islands, highlands with sparse admin data |
| `building_clusters` | DBSCAN-grouped building footprints → convex hull polygons | Dense areas with good building traces but no boundaries |

Configure in YAML under `extraction`:

```yaml
extraction:
  # Source order matters — each fills gaps left by previous sources
  village_sources: [admin_boundary, place_nodes, building_clusters]

  # admin_boundary: OSM admin levels to try in order
  village_admin_levels: [9, 8, 7]   # 9=desa, 8=kecamatan, 7=kabupaten

  # place_nodes: per-tag radius and population density
  village_place_tags: [village, hamlet, town, suburb, quarter]
  village_place_settings:
    hamlet:  {radius_m: 300,  pop_density: 600}
    village: {radius_m: 800,  pop_density: 1200}
    town:    {radius_m: 2000, pop_density: 3000}
    suburb:  {radius_m: 1200, pop_density: 4000}
    quarter: {radius_m: 600,  pop_density: 3500}

  # building_clusters: DBSCAN parameters
  village_cluster_eps_m: 300           # cluster radius (metres)
  village_cluster_min_buildings: 10    # minimum buildings per cluster
  village_persons_per_dwelling: 4.0    # fallback occupancy

  # Per-OSM-building-type occupancy (persons); absent types use persons_per_dwelling
  # Set to 0.0 to exclude non-residential buildings from population counting
  village_building_persons:
    house: 4.5
    apartments: 20.0
    farm: 5.5
    commercial: 0.0
    mosque: 0.0
    # ... (full default table in config_loader.py)
```

---

## Configuration Reference

Full annotated examples: [configs/](configs/)

### Key fields

| Field | Values | Description |
|---|---|---|
| `disaster.type` | `volcano` `earthquake` `flood` `landslide` | Selects InaRISK hazard layer |
| `region.type` | `circle` `bbox` | Region shape |
| `region.radius_km` | float | Radius for circle region |
| `extraction.use_cached_osm` | `true` `false` | Skip OSM HTTP download if cache exists |
| `extraction.village_sources` | list | Ordered village extraction sources |
| `extraction.village_admin_levels` | list | OSM admin levels to try (e.g. `[9, 8, 7]`) |
| `extraction.village_place_settings` | dict | Per-place-tag `radius_m` and `pop_density` |
| `extraction.village_building_persons` | dict | Per-building-type occupancy count |
| `extraction.shelter_tags` | dict | OSM tags to query for shelters |
| `extraction.shelter_min_area_m2` | float | Minimum polygon area for shelter inclusion |
| `routing.weight_risk` | `0.0–1.0` | Weight of hazard penalty in composite score |
| `routing.weight_distance` | `0.0–1.0` | Weight of distance in composite score |
| `routing.weight_road_quality` | `0.0–1.0` | Weight of road quality in composite score |
| `routing.max_route_risk_threshold` | `0.0–1.0` | Discard routes with max risk above this |
| `execution.mode` | `naive` `parallel` `hpc` | Execution mode |
| `execution.n_workers` | integer | Worker count for parallel mode |
| `preloaded_network_json` | file path | Skip extraction — use pre-extracted network JSON |
| `preloaded_network_pycgr` | file path | Skip extraction — use pre-extracted PYCGR file |
| `preloaded_villages_geojson` | file path | Skip extraction — use pre-extracted villages |
| `preloaded_shelters_geojson` | file path | Skip extraction — use pre-extracted shelters |
| `preloaded_poi_csv` | file path | Legacy RespondOR v1 combined POI CSV |

---

## GAMA Simulation

After running the pipeline and exporting SHP files, prepare GAMA simulation inputs:

```bash
python -m experiments.prepare_gama_inputs --config configs/demak_flood_2024.yaml
```

Then in GAMA Platform:
1. Open `simulation/models/EvacuationModel.gaml`
2. Set the `inputs_dir` experiment parameter to `output/<scenario_id>/gama_inputs/`
3. Run `EvacuationExperiment` (GUI) or `BatchHeadless`

The GAMA model reads `villages.csv`, `shelters.csv`, `routes.csv`, `scenario.json`, and optionally `roads.shp` for road-following movement. EvacueeAgents follow pre-computed waypoints from the optimization stage, with BPR congestion modeling and hazard proximity speed penalties.

---

## InaRISK API

Hazard data is sourced from BNPB's [InaRISK](https://inarisk.bnpb.go.id) service via the ImageServer identify endpoint.

- API response: `"value": "0.85"` or `"NoData"` (→ 0.0)
- Normalized score: [0, 1]; values above `max_route_risk_threshold` prune edges as impassable
- Supported layers: `volcano`, `earthquake`, `flood`, `landslide`
- Responses cached in `data/raw/osm_cache/<scenario>/` after first query
- Rate-limited (`inarisk_rate_limit_s`) with up to 2 retries on timeout

> **Note:** InaRISK coverage varies by hazard type and geography. For areas outside the mapped hazard extent (e.g. remote eastern Indonesia for some layers), the API returns `NoData` for all points — this is correct geographic behavior, not a bug. The risk heatmap layer in `evacuation_map.html` is only rendered when at least one village has a risk score > 0.

---

## HPC Execution (MPI + SLURM)

```bash
# Local test — 4 MPI ranks on one machine
mpirun -n 4 python -m src.main --config configs/demak_flood_2024.yaml --mode hpc

# Cluster deployment
sbatch hpc/slurm_job.sh configs/demak_flood_2024.yaml
```

- **Rank 0**: loads OSM data, builds weighted graph, queries InaRISK, broadcasts to workers
- **Ranks 1..N**: receive a village partition, compute Dijkstra routes independently
- **Gather**: routes collected at rank 0 for greedy assignment and output

**Fallback:** If `mpi4py` is not installed, HPC mode automatically uses `ProcessPoolExecutor` on a single node.

---

## Preloaded Input Mode

To skip OSM + InaRISK extraction on repeat runs, generate preloaded files once:

```bash
python -m experiments.build_legacy_input --config configs/demak_flood_2024.yaml
```

Then run directly from cache:

```bash
python -m src.main --config output/demak_flood_2024/legacy_input/scenario_preloaded.yaml
```

The generated `scenario_preloaded.yaml` is identical to the source config but with all `preloaded_*` fields set and `use_cached_osm: true`. POI CSV includes `admin_level` and `source` columns tracking which village extraction method produced each settlement.

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## References

- [InaRISK BNPB](https://inarisk.bnpb.go.id) — Indonesia national hazard risk data
- [osmnx](https://github.com/gboeing/osmnx) — OpenStreetMap network extraction
- [GAMA Platform](https://gama-platform.org) — Agent-based modeling for evacuation simulation
- [OsmToRoadGraph / PYCGR format](https://github.com/AndGem/OsmToRoadGraph) — Legacy network file format
- [BPR function](https://en.wikipedia.org/wiki/Bureau_of_Public_Roads) — Traffic congestion model
