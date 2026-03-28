# RespondOR-EvacuationRoute

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Research-grade disaster evacuation route optimization system for Indonesia. Integrates OpenStreetMap road networks, InaRISK BNPB hazard data, and multi-objective routing to maximize the number of people evacuated to safe shelters.

## Overview

Given a disaster event (earthquake, volcano, flood, or landslide), the system:

1. Extracts the road network, villages, and shelter candidates from OpenStreetMap (or loads a pre-extracted PYCGR network) for the affected region
2. Queries the InaRISK BNPB API for hazard risk indices at each location
3. Builds a weighted evacuation graph with composite edge weights (distance × road quality × risk penalty)
4. Computes optimized routes from each village to candidate shelters using Dijkstra's algorithm
5. Assigns village populations to shelters using greedy or LP optimization, maximizing total evacuated population subject to shelter capacity constraints
6. Optionally runs agent-based evacuation simulation using GAMA-platform
7. Benchmarks and compares performance across sequential, parallel, and HPC execution modes

### Goals

- **Population-maximizing assignment** — not just shortest paths, but routes that get the most people out
- **Hazard-aware routing** — InaRISK risk scores penalize or block routes through high-risk areas
- **Dual input modes** — live OSM extraction (osmnx) or pre-extracted PYCGR network
- **Three execution modes** — naive (sequential), parallel (multiprocessing), and HPC (MPI + SLURM)
- **GAMA integration** — agent-based simulation for realistic congestion and dynamic re-routing analysis

---

## Repository Structure

```
RespondOR-EvacuationRoute/
├── configs/
│   ├── disaster_scenario.yaml          # Main example (Merapi OSM)
│   ├── jakarta_osm_input.yaml          # Jakarta flood — OSM input
│   ├── jakarta_legacy_input.yaml       # Jakarta flood — PYCGR input
│   ├── merapi_benchmark_osm.yaml       # Merapi benchmark — OSM mode
│   └── merapi_benchmark_pycgr.yaml     # Merapi benchmark — PYCGR mode
├── data/
│   ├── examples/jakarta/               # Jakarta input files (PYCGR + POI CSV)
│   │   ├── jakarta.pycgrc              # Pre-extracted road network (160K nodes)
│   │   └── jakarta_locations.csv       # Villages + shelters (346 POIs)
│   ├── processed/                      # Auto-generated exports (e.g. merapi_pycgr/)
│   └── raw/osm_cache/                  # Cached OSM extracts (auto-populated)
├── docs/
│   └── SECTIONS_D_TO_O.md             # Design spec (algorithms, HPC, GAMA)
├── experiments/
│   └── compare_input_modes.py          # OSM vs PYCGR benchmark runner
├── hpc/
│   └── slurm_job.sh                   # SLURM job script (4 nodes × 8 MPI ranks)
├── output/                            # Pipeline outputs (maps, CSVs, benchmarks)
├── plugins/
│   └── gama-respondor-plugin/         # Java/Maven GAMA plugin (OSM + InaRISK operators)
├── simulation/
│   └── models/EvacuationModel.gaml    # GAMA agent-based evacuation model
├── src/
│   ├── config/config_loader.py        # YAML config loader + dataclasses
│   ├── data/
│   │   ├── inarisk_client.py          # InaRISK BNPB API client (with caching)
│   │   ├── models.py                  # Core dataclasses (Node, Edge, Village, Shelter, Route)
│   │   ├── osm_extractor.py           # OSM extraction (osmnx 2.x) + PYCGR import/export
│   │   └── population_loader.py       # Population and shelter capacity estimation
│   ├── graph/graph_builder.py         # Weighted NetworkX evacuation graph
│   ├── hpc/
│   │   ├── distributed_runner.py      # MPI runner (mpi4py; fallback: ProcessPool)
│   │   ├── naive_runner.py            # Sequential pipeline
│   │   └── parallel_runner.py         # ThreadPool (I/O) + ProcessPool (routing)
│   ├── routing/
│   │   ├── assignment.py              # Greedy and LP population-to-shelter assignment
│   │   └── heuristic_optimizer.py     # Dijkstra-based route computation and scoring
│   ├── simulation/gama_orchestrator.py # GAMA headless invocation and output parsing
│   ├── visualization/visualizer.py    # Folium interactive map + matplotlib charts
│   └── main.py                        # CLI entry point
└── tests/                             # pytest test suite
```

---

## Algorithm

### Edge Weight (composite)

```
w(e) = distance(e) × quality_weight(e) × (1 + risk_weight × risk_score(e))
```

- `quality_weight`: road type factor (motorway=0.5 → footway=2.5)
- `risk_score`: InaRISK hazard index normalized to [0, 1]
- Edges with `risk_score > 0.9` are pruned as impassable

### Route Composite Score (lower = better)

```
score = w_dist × (dist / 50km) + w_risk × avg_risk + w_quality × (worst_quality / 3) + w_time × (time / 2h)
```

Default weights: `w_dist=0.25, w_risk=0.45, w_quality=0.20, w_time=0.10`

### Population Assignment

Greedy algorithm: sort all (village, shelter, route) candidates by composite score, assign greedily while tracking remaining shelter capacity and unmet village demand. LP fallback available via `scipy.optimize.linprog`.

---

## Setup

### Requirements

- Python 3.14 (via pyenv)
- Java 11+ (for GAMA plugin, optional)
- GAMA-platform 1.9+ (for simulation, optional)
- OpenMPI or Intel MPI (for HPC mode, optional)

### Installation

```bash
# Create virtual environment
pyenv virtualenv 3.14.2 respondor-evroute
pyenv local respondor-evroute

# Install dependencies
pip install -r requirements.txt

# Optional: MPI support (requires OpenMPI system package)
# macOS:   brew install open-mpi && pip install mpi4py
# Linux:   module load openmpi/4.1 && pip install mpi4py
```

---

## Usage

### Run the pipeline

```bash
# Sequential (baseline)
python -m src.main --config configs/disaster_scenario.yaml --mode naive

# Parallel (multiprocessing, single node)
python -m src.main --config configs/disaster_scenario.yaml --mode parallel --workers 8

# HPC/MPI (multi-node, see HPC section below)
mpirun -n 8 python -m src.main --config configs/disaster_scenario.yaml --mode hpc

# With GAMA simulation
python -m src.main --config configs/disaster_scenario.yaml --mode naive --simulate
```

### Outputs (in `output/<scenario_id>/`)

| File | Description |
|---|---|
| `evacuation_map.html` | Interactive Folium map with routes, villages, shelters, risk heatmap |
| `evacuation_results.csv` | Per-village assignment: shelter, population, distance, travel time |
| `optimization_summary.json` | Aggregate metrics: evacuation ratio, avg risk, shelter utilization |
| `timings_naive.json` | Stage-by-stage runtime breakdown |
| `evacuation_summary.png` | Benchmark comparison chart |

---

## Input Modes

The system supports two input modes. Both produce identical internal data structures — the rest of the pipeline is the same.

### Mode A — OSM (live extraction)

The default mode. Uses `osmnx` to extract road networks, villages, and shelters from OpenStreetMap. Data is cached locally after the first download.

```yaml
# configs/disaster_scenario.yaml
extraction:
  osm_cache_dir: "data/raw/osm_cache/merapi"
  use_cached_osm: true          # skip HTTP download if cache exists
  network_type: "all"
  village_admin_level: 9        # 9=desa, 8=kecamatan, 7=kabupaten
```

Requires an internet connection on first run. Subsequent runs use the local cache with no network access.

### Mode B — PYCGR (pre-extracted)

For pre-extracted networks in the RespondOR v1 / OsmToRoadGraph text format. Provide a PYCGR network file and a POI CSV file — no OSM download needed.

```yaml
# configs/jakarta_legacy_input.yaml
preloaded_network_pycgr: "data/examples/jakarta/jakarta.pycgrc"
preloaded_poi_csv:        "data/examples/jakarta/jakarta_locations.csv"
```

#### PYCGR network format

Plain text. Comment lines start with `#`. Structure:

```
# comments
<n_nodes>
<n_edges>
<node_id> <lat> <lon>          (one per line, n_nodes lines)
...
<src> <tgt> <length_m> <highway_type> <max_speed_kmh> <bidirectional>  (n_edges lines)
...
```

Example (`merapi_network.pycgr`, first few lines):
```
# RespondOR export — Merapi OSM cache
83486
108452
0 -7.541 110.446
1 -7.542 110.447
...
0 1 42.3 residential 30 1
...
```

#### POI CSV format

No header. Columns: `name, type, latitude, longitude[, node_id]`

- `type=village` → treated as evacuation origin
- `type=shelter`, `depot`, `hospital`, `building` etc. → treated as shelter destination

```csv
ANCOL,village,-6.125215,106.8362474,118962
ANCOL BARAT,shelter,-6.130000,106.830000,118963
```

The Jakarta example data (346 POIs: 212 villages + 134 shelters) is included at [data/examples/jakarta/](data/examples/jakarta/).

#### Exporting OSM data to PYCGR

You can export any cached OSM scenario to PYCGR format for archival or portability:

```python
from src.data.osm_extractor import OSMExtractor

extractor = OSMExtractor(cache_dir="data/raw/osm_cache/merapi")
nodes, edges = extractor.extract_road_network(region, use_cache=True)
villages     = extractor.extract_villages(region, use_cache=True)
shelters     = extractor.extract_shelters(region, use_cache=True)

extractor.export_to_pycgr(nodes, edges, "output/merapi_network.pycgr")
extractor.export_pois_to_csv(villages, shelters, "output/merapi_pois.csv")
```

---

## Example Scenarios

### Mount Merapi Eruption (OSM mode)

```yaml
# configs/disaster_scenario.yaml
scenario_id: "merapi_eruption_2024"
disaster:
  type: "volcano"
  lat: -7.5407
  lon: 110.4457
  severity: 0.8
region:
  type: "circle"
  center: [-7.5407, 110.4457]
  radius_km: 20.0
```

```bash
python -m src.main --config configs/disaster_scenario.yaml --mode naive
```

### Jakarta Flood (OSM mode)

```yaml
# configs/jakarta_osm_input.yaml
scenario_id: "jakarta_flood_osm"
disaster:
  type: "flood"
  lat: -6.2384
  lon: 106.7918
  severity: 0.7
region:
  type: "bbox"
  bbox: [-6.4000, 106.6000, -6.0800, 106.9800]
```

### Jakarta Flood (PYCGR mode)

```yaml
# configs/jakarta_legacy_input.yaml
scenario_id: "jakarta_flood_legacy"
disaster:
  type: "flood"
  lat: -6.2384
  lon: 106.7918
  severity: 0.7
region:
  type: "bbox"
  bbox: [-6.4000, 106.6000, -6.0800, 106.9800]
preloaded_network_pycgr: "data/examples/jakarta/jakarta.pycgrc"
preloaded_poi_csv:        "data/examples/jakarta/jakarta_locations.csv"
```

```bash
python -m src.main --config configs/jakarta_legacy_input.yaml --mode naive
```

---

## Configuration Reference

Full annotated example: [configs/disaster_scenario.yaml](configs/disaster_scenario.yaml)

Key fields:

| Field | Values | Description |
|---|---|---|
| `disaster.type` | `volcano` `earthquake` `flood` `landslide` | Selects InaRISK hazard layer |
| `region.type` | `circle` `bbox` | Region shape |
| `extraction.use_cached_osm` | `true` `false` | Skip OSM download if cache exists |
| `extraction.village_admin_level` | `9` (desa) `8` (kecamatan) | OSM admin level for village boundaries |
| `routing.weight_risk` | `0.0–1.0` | Weight of hazard penalty in composite score |
| `routing.max_route_risk_threshold` | `0.0–1.0` | Discard routes with max risk above this |
| `routing.assignment_method` | `greedy` `lp` | Population-to-shelter assignment algorithm |
| `execution.mode` | `naive` `parallel` `hpc` | Execution mode |
| `execution.n_workers` | integer | Worker count for parallel mode |
| `benchmark_village_limit` | integer | Cap on villages routed (benchmark runs only) |
| `preloaded_network_pycgr` | file path | PYCGR network file |
| `preloaded_poi_csv` | file path | POI CSV file (villages + shelters) |

---

## HPC Execution (MPI + SLURM)

The HPC mode uses MPI (`mpi4py`) for distributed route computation across cluster nodes. No separate cluster daemon is needed — SLURM launches MPI processes directly.

### How it works

- **Rank 0 (master)**: loads OSM data, builds the weighted graph, queries InaRISK, then broadcasts the graph and shelters to all workers
- **Ranks 1..N (workers)**: receive a village partition (round-robin split), compute routes independently using Dijkstra
- **Gather**: routes are collected back at rank 0 for greedy assignment and output

### Local testing (mpirun)

```bash
# 4 ranks on one machine
mpirun -n 4 python -m src.main --config configs/disaster_scenario.yaml --mode hpc
```

### Cluster deployment (SLURM)

```bash
# Submit with default config (4 nodes × 8 ranks = 32 MPI processes)
sbatch hpc/slurm_job.sh

# Custom config
sbatch hpc/slurm_job.sh configs/merapi_benchmark_osm.yaml

# With benchmark comparison
sbatch hpc/slurm_job.sh configs/disaster_scenario.yaml --benchmark
```

SLURM defaults (edit [hpc/slurm_job.sh](hpc/slurm_job.sh) to change):

```
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8    # 32 MPI ranks total
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=compute
```

### Install mpi4py

```bash
# macOS (local testing)
brew install open-mpi
pip install mpi4py

# Linux cluster
module load openmpi/4.1        # or intel-mpi/2021
pip install mpi4py
```

**Fallback:** If `mpi4py` is not installed, HPC mode automatically falls back to `ProcessPoolExecutor` on a single node with no code change needed.

---

## Benchmark

### Input mode × execution mode comparison

The `experiments/compare_input_modes.py` script runs all 6 combinations of input mode (OSM vs PYCGR) and execution mode (naive, parallel-4, parallel-8) on the Merapi scenario, then produces a timing report and chart.

```bash
python -m experiments.compare_input_modes
```

**Resource constraint:** Each run is capped at `benchmark_village_limit: 500` villages to keep each combination under ~10 min on a standard laptop. Risk data is cached after the first run so InaRISK API calls are not repeated.

**Outputs** (in `output/benchmark/`):

| File | Description |
|---|---|
| `comparison_results.json` | Raw results for all 6 runs |
| `comparison_report.txt` | ASCII table: wall time, CPU time, memory, stage breakdown, speedup |
| `comparison_chart.png` | Bar chart: stage timings and routing speedup |

### Per-run timing files

Each pipeline run also writes a JSON timing file to `output/<scenario_id>/`:

```json
{
  "mode": "parallel",
  "n_workers": 4,
  "timings": {
    "extraction": 2.04,
    "risk_scoring": 320.7,
    "graph_build": 1889.7,
    "routing": 203.0,
    "assignment": 0.007
  },
  "total": 2415.4
}
```

---

## InaRISK API

Hazard data is sourced from BNPB's [InaRISK](https://inarisk.bnpb.go.id) service. Risk scores are queried per-point in batches and cached locally.

- Raw API values: 1 (low) – 3 (high)
- Normalized: `(raw - 1) / 2.0` → [0, 1]
- Edges with normalized risk > 0.9 are pruned as impassable

Supported hazard layers: `volcano`, `earthquake`, `flood`, `landslide`

The client respects `inarisk_rate_limit_s` between batch calls and retries on timeout (up to 2 retries). Cached responses are stored in `data/raw/osm_cache/<scenario>/inarisk/`.

---

## GAMA Simulation

After route optimization, optionally run an agent-based simulation in [GAMA-platform](https://gama-platform.org) to model realistic congestion and dynamic re-routing:

```bash
python -m src.main --config configs/disaster_scenario.yaml --mode naive --simulate
```

The GAMA model ([simulation/models/EvacuationModel.gaml](simulation/models/EvacuationModel.gaml)) takes the optimized routes as input and simulates individual agents (vehicles/pedestrians) moving along them, accounting for road capacity and congestion. Simulation outputs are written to `output/simulation/<scenario_id>/`.

A Java/Maven GAMA plugin ([plugins/gama-respondor-plugin/](plugins/gama-respondor-plugin/)) provides OSM and InaRISK operators directly inside GAML for advanced use cases.

---

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

Test suite covers: graph construction, composite weights, Dijkstra routing, greedy assignment, InaRISK client, edge pruning, PYCGR import/export, and POI CSV parsing.

---

## References

- [InaRISK BNPB](https://inarisk.bnpb.go.id) — Indonesia national hazard risk data
- [osmnx](https://github.com/gboeing/osmnx) — OpenStreetMap network extraction
- [GAMA-platform](https://gama-platform.org) — Agent-based modeling for evacuation simulation
- [OsmToRoadGraph / PYCGR format](https://github.com/AndGem/OsmToRoadGraph) — Legacy network file format
- [BPR function](https://en.wikipedia.org/wiki/Bureau_of_Public_Roads) — Traffic congestion model
