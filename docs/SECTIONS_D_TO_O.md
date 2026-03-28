# RespondOR-EvacuationRoute: Design Reference (Sections D–O)

---

## SECTION D — DATA MODEL

All data models are in [src/data/models.py](../src/data/models.py).

| Class | Fields | Notes |
|-------|--------|-------|
| `DisasterInput` | location, disaster_type, name, severity | Input parameters |
| `RegionOfInterest` | region_type, bbox, center, radius_km | Converts to bbox |
| `NetworkNode` | node_id, lat, lon, risk_scores | Dict of hazard→score |
| `NetworkEdge` | src, tgt, length_m, highway_type, speed, lanes, risk | BPR capacity computed |
| `Village` | id, name, centroid, population, area, risk_scores | Pop from CSV or area |
| `Shelter` | id, name, centroid, capacity, type, area, risk | Cap from CSV or area |
| `RiskLayer` | lat, lon, disaster_type, risk_score | From InaRISK |
| `EvacuationRoute` | village_id→shelter_id, path, dist, time, risk, score | Ranked by composite |
| `Assignment` | village→shelter→route, pop_count, fraction | LP or greedy |
| `OptimizationResult` | assignments, routes, stats, mode, runtime | Full pipeline output |
| `SimulationOutput` | saved/delayed/failed, ratio, times, bottlenecks | Per GAMA run |
| `BenchmarkResult` | mode, workers, wall_time, cpu_time, mem, speedup | Performance |

---

## SECTION E — ALGORITHMIC DESIGN

### E1. OSM Extraction (osm_extractor.py)

```
ALGORITHM: ExtractOSMData(region, use_cache)
  bbox = region.to_bbox()
  cache_key = hash(bbox + network_type)
  IF cache_key in disk_cache AND use_cache:
    RETURN load_from_cache(cache_key)
  graph = osmnx.graph_from_bbox(bbox, network_type="all")
  nodes, edges = convert_oxgraph(graph)
  FOR each edge:
    highway = edge.tags["highway"]
    speed = edge.tags["maxspeed"] OR speed_lookup[highway]
    lanes = edge.tags["lanes"] OR 1
  save_to_cache(cache_key, nodes, edges)
  RETURN nodes, edges
```

### E2. InaRISK Extraction (inarisk_client.py)

```
ALGORITHM: GetRiskScores(points, disaster_type)
  config = INARISK_CONFIG[disaster_type]
  batches = split(points, batch_size=20)
  FOR each batch:
    FOR each point in batch:
      x, y = to_web_mercator(point.lat, point.lon)
      url = BASE_URL / config.service / MapServer / 0 / query
      params = {geometry: (x,y), spatialRel: intersects, outFields: INDEKS_BAHAYA}
      response = HTTP_GET(url, params)
      raw_val = response.features[0].attributes.INDEKS_BAHAYA
      score = normalize(raw_val)  # (val-1)/2 if val in [1,3]
    sleep(rate_limit_s)
  RETURN scores[0..N]
```

### E3. Graph Construction (graph_builder.py)

```
ALGORITHM: BuildEvacuationGraph(nodes, edges, disaster_type)
  G = nx.Graph()
  FOR each node: G.add_node(id, lat, lon)
  FOR each edge:
    IF edge.risk > impassable_threshold: SKIP
    quality_w = highway_quality_weights[edge.highway_type]
    risk_w    = edge.risk_score
    composite = edge.length_m * quality_w * (1 + risk_weight * risk_w)
    G.add_edge(src, tgt, weight=composite, length=..., travel_time=..., ...)
  FOR each village: village.nearest_node = argmin_dist(G.nodes, village.centroid)
  FOR each shelter: shelter.nearest_node = argmin_dist(G.nodes, shelter.centroid)
  RETURN G
```

### E4. Route Candidate Generation (heuristic_optimizer.py)

```
ALGORITHM: ComputeRoutes(G, villages, shelters)
  shelter_node_map = {s.nearest_node: s for s in shelters}
  FOR each village v:
    IF v.nearest_node not in G: SKIP
    lengths, paths = Dijkstra(G, source=v.nearest_node, weight="weight")
    candidates = []
    FOR each shelter s:
      IF s.nearest_node not in paths: CONTINUE
      path = paths[s.nearest_node]
      dist, time, risks, qualities = aggregate_path_metrics(G, path)
      IF max(risks) > max_risk_threshold: CONTINUE
      score = w_dist*norm(dist) + w_risk*avg(risks) + w_quality*max(qualities) + w_time*norm(time)
      candidates.append(Route(v, s, path, score))
    SORT candidates by score ASC
    YIELD candidates[:max_routes_per_village]
```

### E5. Population-to-Shelter Assignment (assignment.py)

```
ALGORITHM: GreedyAssignment(villages, shelters, routes_by_village)
  // Flatten and sort all (village, shelter, route) by composite_score ASC
  candidates = [(route.score, v, s, route) for v in villages for route in routes_by_village[v]]
  SORT candidates by score ASC
  shelter_remaining = {s.id: s.capacity for s in shelters}
  village_assigned  = {v.id: 0 for v in villages}
  FOR (score, v, s, route) in candidates:
    remaining_pop = v.population - village_assigned[v.id]
    IF remaining_pop <= 0: CONTINUE  // fully assigned
    cap = shelter_remaining[s.id]
    IF cap <= 0: CONTINUE            // shelter full
    assign = min(remaining_pop, cap)
    village_assigned[v.id] += assign
    shelter_remaining[s.id] -= assign
    RECORD Assignment(v.id, s.id, route, assign)
  RETURN OptimizationResult(assignments)

ALGORITHM: LPAssignment(villages, shelters, routes_by_village)
  // Build LP:
  // Minimize -sum(x_ij * pop_i)   [maximize evacuated]
  // Subject to:
  //   sum_j x_ij <= 1   (each village fully assigned at most once)
  //   sum_i x_ij * pop_i <= cap_j  (shelter capacity)
  //   0 <= x_ij <= 1
  c, A_ub, b_ub, bounds = build_lp_matrices(villages, shelters, routes_by_village)
  result = scipy.linprog(c, A_ub, b_ub, bounds, method="highs")
  RETURN build_assignments(result.x)
```

### E6. Performance Benchmarking (benchmark_runner.py)

```
ALGORITHM: Benchmark(config, modes)
  naive_time = None
  FOR mode in modes:
    gc.collect()
    tracemalloc.start()
    t_start = perf_counter()
    result = run_mode(config, mode)
    wall_time = perf_counter() - t_start
    _, peak_mem = tracemalloc.get_traced_memory()
    IF mode == "naive": naive_time = wall_time
    speedup = naive_time / wall_time IF mode != "naive" ELSE 1.0
    efficiency = speedup / n_workers
    RECORD BenchmarkResult(mode, wall_time, peak_mem, speedup, efficiency)
  RETURN benchmark_results
```

---

## SECTION F — PARALLELIZATION STRATEGY

### F1. Naive (Sequential) Mode

All stages execute in a single process:
- OSM extraction → risk scoring → graph build → route computation (loop over villages) → assignment

**Bottlenecks**: Route computation is O(V × S × E log E) where V=villages, S=shelters, E=edges.

### F2. Parallel (Single-Machine) Mode

| Stage | Parallelism | Method |
|-------|-------------|--------|
| OSM extraction | 3 concurrent queries (roads/villages/shelters) | ThreadPoolExecutor (I/O-bound) |
| Risk scoring | 2 concurrent (villages + shelters) | ThreadPoolExecutor |
| Route computation | V parallel tasks (one per village) | ProcessPoolExecutor (CPU-bound) |
| Simulation batch | N parallel GAMA runs | ProcessPoolExecutor |

**Key design**: `_routes_for_village_standalone` is a module-level function, making it picklable for multiprocessing. The NetworkX graph is passed as argument (pickled once per task).

**Memory sharing**: Graph is copied to each worker process (NetworkX graphs ~50-500MB for city scale). For large graphs, use `ray.put()` for zero-copy sharing.

**Overhead tradeoffs**:
- ProcessPoolExecutor: ~0.1-0.5s startup overhead; good for ≥10 villages
- ThreadPoolExecutor: no GIL bypass for CPU; use only for I/O-bound tasks

### F3. Distributed (HPC) Mode

```
HPC Village Partitioning Strategy:
  N villages partitioned across K workers:
  - Worker i gets villages[i*N/K : (i+1)*N/K]
  - Graph broadcast via ray.put() (shared object store, zero-copy)
  - Results aggregated by head node

Ray Object Store:
  G_ref = ray.put(G)      # Stored once in plasma store
  Each worker: ray.get(G_ref) → local reference (no copy if same node)
  Cross-node: automatic serialization (pickle) on first access
```

**Partitioning strategy**: Village-based partitioning. Each Ray task handles one village (fine-grained) or a village shard (coarse-grained). Fine-grained allows better load balancing; coarse-grained reduces scheduling overhead.

**I/O bottlenecks**:
- InaRISK API is rate-limited (1 req/s) → parallel batching reduces but doesn't eliminate latency
- OSM extraction is cached to disk after first run

**Reproducibility**: Fixed `random_seed` per scenario. GAMA simulation seeds controlled via run_id.

---

## SECTION G — HPC DESIGN

### G1. Job Structure

```
Job: respondor_evacuation
  ├── Node 0 (head):   Ray head + optimization orchestration
  ├── Node 1 (worker): Ray worker + route computation
  ├── Node 2 (worker): Ray worker + route computation
  └── Node 3 (worker): Ray worker + GAMA simulation batch
```

### G2. SLURM Integration

See [hpc/slurm_job.sh](../hpc/slurm_job.sh). Key parameters:
- `--nodes=4` — multi-node execution
- `--ntasks-per-node=8` — 8 Ray workers per node = 32 total
- `--mem=16G` — per-node memory (NetworkX graph ~200MB + data ~500MB)
- `--time=02:00:00` — maximum job time

### G3. Framework Justification

**Ray** (preferred):
- Native Python distributed computing
- Object store with zero-copy sharing (critical for large NetworkX graphs)
- Dynamic task scheduling
- Fault tolerance via driver restart
- Easy local → HPC scaling (`ray.init(address="auto")`)

**Dask** (alternative):
- Better for DataFrame operations (if using geopandas pipelines)
- Delayed task graph compilation before execution
- Slightly higher overhead for task-parallel workloads

**MPI** (for extreme scale):
- Required for tight coupling (rare in this application)
- Use if running on >1000 nodes or if graph partitioning needed

### G4. Expected Scaling

| Workers | Speedup (routing) | Efficiency |
|---------|------------------|------------|
| 1 (naive) | 1.0× | 100% |
| 4 | 3.2–3.6× | 80–90% |
| 8 | 5.6–6.4× | 70–80% |
| 16 | 8–12× | 50–75% |
| 32 | 12–20× | 38–63% |

Sub-linear scaling due to: graph serialization overhead, InaRISK rate limiting, assignment is sequential.

### G5. Failure/Restart

- **Ray**: automatic task retry on worker failure (`max_retries=3`)
- **Checkpointing**: OSM cache saves extraction results; benchmark saves intermediate timings
- **SLURM restart**: `#SBATCH --requeue` for preemptable queues

---

## SECTION H — GAMA INTEGRATION

### H1. Simulation Entities

See [simulation/models/EvacuationModel.gaml](../simulation/models/EvacuationModel.gaml).

| Agent | Attributes | Behavior |
|-------|-----------|---------|
| `EvacueeAgent` | village_id, population_size, target_shelter, status | Move toward shelter at speed reduced by congestion/hazard |
| `ShelterAgent` | shelter_id, max_capacity, current_load | Accept/reject arrivals based on capacity |
| `global` | flow maps, statistics | Update congestion, collect stats |

### H2. Movement & Congestion Logic

```gaml
// BPR congestion factor
float congestion_factor = 1 + bpr_alpha * (flow/capacity)^bpr_beta

// Effective speed
float eff_speed = base_speed / (congestion_factor * hazard_factor)

// Steps to shelter
float steps_to_arrive = (distance_km / eff_speed_kmh) * (60 / time_step_min)
```

### H3. Python-GAMA Data Flow

```
Python Optimization Result
  └── src/simulation/gama_orchestrator.py
       ├── writes: villages.csv  (id, name, lat, lon, population)
       ├── writes: shelters.csv  (id, name, lat, lon, capacity)
       ├── writes: routes.csv    (village_id, shelter_id, distance, time, pop)
       ├── writes: sim_config.json
       └── generates: experiment.xml  (GAMA headless XML)
            └── invokes: gama-headless experiment.xml output_dir/
                 └── reads: monitors CSV per run_id
Python reads back:
  └── total_saved, evacuation_ratio, avg_time, worst_time
```

### H4. GAMA Plugin (Java)

Plugin provides GAML operators:
- `osm_get_villages(bbox)` → list of village records
- `osm_get_shelters(bbox)` → list of shelter candidates
- `inarisk_get_risk(lat, lon, hazard_type)` → risk score [0-1]

Build: `mvn clean package` in `plugins/gama-respondor-plugin/`
Install: copy JAR to `GAMA/plugins/`

---

## SECTION I — IMPLEMENTATION PLAN

| Phase | Tasks | Outputs | Dependencies | Risks |
|-------|-------|---------|-------------|-------|
| **1** Minimal prototype | Graph builder + optimizer + test data | Working naive pipeline | Python deps | OSM data quality |
| **2** Full data pipeline | OSM extractor + InaRISK + population loader | Real data ingestion | Internet/API access | InaRISK rate limits |
| **3** Heuristic optimizer | LP assignment + composite scoring | Optimized routes | scipy | Solver convergence |
| **4** GAMA integration | GAML model + orchestrator + headless runner | ABM simulation | GAMA 1.9.3 | GAMA compatibility |
| **5** Multithreading | ProcessPoolExecutor routing + ThreadPoolExecutor I/O | Parallel pipeline | None (stdlib) | Pickle serialization |
| **6** HPC distribution | Ray integration + SLURM scripts | Distributed execution | Ray cluster access | Network latency |
| **7** Evaluation | Benchmark runner + experiment scripts + charts | Research results | All phases | Time, compute budget |

---

## SECTION J — CODE SCAFFOLDING

All key files have been implemented:

| File | Purpose | Status |
|------|---------|--------|
| [src/main.py](../src/main.py) | CLI entrypoint | ✓ Complete |
| [src/config/config_loader.py](../src/config/config_loader.py) | Config loader | ✓ Complete |
| [src/data/models.py](../src/data/models.py) | Data models | ✓ Complete |
| [src/data/osm_extractor.py](../src/data/osm_extractor.py) | OSM extraction | ✓ Complete |
| [src/data/inarisk_client.py](../src/data/inarisk_client.py) | InaRISK API | ✓ Complete |
| [src/data/population_loader.py](../src/data/population_loader.py) | Pop/capacity | ✓ Complete |
| [src/graph/graph_builder.py](../src/graph/graph_builder.py) | Graph construction | ✓ Complete |
| [src/routing/heuristic_optimizer.py](../src/routing/heuristic_optimizer.py) | Route optimizer | ✓ Complete |
| [src/routing/assignment.py](../src/routing/assignment.py) | LP/greedy assignment | ✓ Complete |
| [src/hpc/naive_runner.py](../src/hpc/naive_runner.py) | Sequential runner | ✓ Complete |
| [src/hpc/parallel_runner.py](../src/hpc/parallel_runner.py) | Parallel runner | ✓ Complete |
| [src/hpc/distributed_runner.py](../src/hpc/distributed_runner.py) | HPC/Ray runner | ✓ Complete |
| [src/simulation/gama_orchestrator.py](../src/simulation/gama_orchestrator.py) | GAMA integration | ✓ Complete |
| [src/benchmark/benchmark_runner.py](../src/benchmark/benchmark_runner.py) | Benchmarking | ✓ Complete |
| [src/visualization/visualizer.py](../src/visualization/visualizer.py) | Maps + charts | ✓ Complete |
| [simulation/models/EvacuationModel.gaml](../simulation/models/EvacuationModel.gaml) | GAMA model | ✓ Complete |
| [tests/test_graph_builder.py](../tests/test_graph_builder.py) | Graph tests | ✓ 8 tests pass |
| [tests/test_optimizer.py](../tests/test_optimizer.py) | Optimizer tests | ✓ 7 tests pass |
| [tests/test_inarisk_client.py](../tests/test_inarisk_client.py) | InaRISK tests | ✓ 7 tests pass |

---

## SECTION K — CONFIGURATION DESIGN

See [configs/disaster_scenario.yaml](../configs/disaster_scenario.yaml) for full example.

Key config blocks:
- `disaster`: type (earthquake/volcano/flood/landslide), lat/lon, severity
- `region`: circle (center + radius_km) or bbox
- `extraction`: OSM cache, village admin level, InaRISK batch settings
- `routing`: score weights (distance/risk/quality/time), max_risk threshold
- `simulation`: GAMA executable, n_runs, max_steps
- `execution`: mode (naive/parallel/hpc), n_workers, Ray/Dask settings

---

## SECTION L — EXPERIMENTAL DESIGN

### L1. Benchmark Scenarios

| Scenario | Type | Region | Pop. | Shelters | Focus |
|----------|------|--------|------|----------|-------|
| Small EQ | earthquake | 10km radius | ~5k | 5 | Baseline |
| Medium Flood | flood | 20km radius | ~50k | 15 | Scale |
| Large Volcano | volcano | 40km radius | ~200k | 30 | HPC |
| Dense Urban | flood | 15km dense | ~300k | 20 | Congestion |
| Sparse Rural | landslide | 30km sparse | ~10k | 3 | Unmet demand |

### L2. Comparison Metrics

| Metric | Definition |
|--------|-----------|
| Evacuation ratio | evacuated / total_population |
| Speedup | naive_time / mode_time |
| Efficiency | speedup / n_workers |
| Avg route risk | mean(route.avg_risk) across assignments |
| Unmet demand | total_population - total_evacuated |
| Shelter utilization | assigned / capacity per shelter |

### L3. Statistical Evaluation

- Repeat each scenario 5× with different random seeds
- Report: mean ± stdev for all metrics
- Speedup plotted vs. n_workers for scalability analysis
- Quality metrics (evacuation ratio, risk) must be equal across all modes (same algorithm, just different parallelism)

---

## SECTION M — VISUALIZATION AND REPORTING

### M1. Outputs

| Output | File | Description |
|--------|------|-------------|
| Interactive map | `evacuation_map.html` | Folium with villages, shelters, routes, risk heatmap |
| Summary chart | `evacuation_summary.png` | Pie (evacuated/unmet) + bar (shelter utilization) |
| Benchmark chart | `benchmark_chart.png` | Wall time, speedup, efficiency comparison |
| Results CSV | `evacuation_results.csv` | Village→shelter assignments with metrics |
| Optimization JSON | `optimization_summary.json` | All aggregate statistics |
| Simulation stats | `simulation_stats.json` | GAMA multi-run aggregate |
| Timing JSON | `timings_naive.json` etc | Per-stage timing per mode |

### M2. Interactive Map Layers

- **Villages**: blue circles sized by population, colored by risk
- **Shelters**: green squares, fill color shows utilization (green→red)
- **Routes**: polylines colored by rank (green=rank1, orange=rank2, red=rank3)
- **Disaster**: red warning marker with 5km radius circle
- **Risk heatmap**: toggleable layer showing InaRISK values

---

## SECTION N — VALIDATION AND LIMITATIONS

### N1. Routing Validation

- **Connectivity check**: verify shortest paths exist between all assigned (village, shelter) pairs
- **Distance sanity**: routes >50km flagged as suspicious
- **Risk sanity**: assigned routes with avg_risk >0.6 flagged for review
- **Capacity balance**: sum of assignments ≤ total shelter capacity

### N2. Simulation Validation

- Compare GAMA simulation evacuation ratio vs. optimizer result
- Expected: simulation ratio ≤ optimizer ratio (congestion and hazard reduce actual outcomes)
- Acceptable discrepancy: ±15%
- Sensitivity analysis on: base_speed_kmh, bpr parameters, time_step

### N3. Data Quality Limitations

| Source | Limitation | Mitigation |
|--------|-----------|------------|
| OSM roads | Incomplete in rural areas | Accept gaps; log warnings |
| OSM admin boundaries | Missing level-9 in some regions | Fall back to place=village |
| InaRISK API | Coarse spatial resolution (~100m-1km cells) | Acceptable for regional planning |
| InaRISK values | Static snapshot; not real-time | Add severity scaling |
| Population estimates | Area-based estimation has ±30-50% error | Use official BPS data if available |
| Shelter capacities | Estimated from building area; may be ±50% | Collect local ground truth |

### N4. Assumptions That May Bias Results

1. **Uniform evacuation start**: all villages begin evacuation simultaneously
2. **Static hazard**: risk field does not evolve during evacuation
3. **Vehicle-only**: footway/path included but assume slow speed
4. **Road passability**: binary (passable/impassable); no partial damage model
5. **Perfect information**: all villagers know their assigned shelter

---

## SECTION O — NEXT STEPS

### O1. Research Publication Path

- **Journal target**: IJDRR (International Journal of Disaster Risk Reduction), NHESS, Computers & Geosciences
- **Paper structure**: Problem formulation + algorithm + Indonesia case study + HPC performance evaluation
- **Key contribution**: Multi-factor population-maximizing evacuation assignment with HPC scalability analysis
- **Dataset**: Release OSM+InaRISK extracted data for Merapi/Sinabung/Semeru case studies

### O2. Real-Time Disaster Response

- Replace static InaRISK with BMKG (meteorology) real-time sensor feeds
- Add WebSocket API endpoint for live route updates
- Implement incremental graph updates as hazard spreads
- Mobile app integration for route guidance

### O3. Remote Sensing Integration

- Ingest LAPAN/Copernicus satellite imagery for:
  - Post-disaster road damage detection (blocked roads → remove from graph)
  - Flood extent mapping (flood polygons → edge risk = 1.0)
  - Building damage assessment (shelter capacity reduction)
- Pipeline: raster → vector damage mask → graph edge update

### O4. Capacity Estimation Improvements

- Use BPS (Badan Pusat Statistik) official population data for villages
- Building footprint analysis (OSM buildings → floor area → capacity)
- Field survey data integration (GIS shapefile with confirmed capacities)
- Crowdsourcing: shelter operators report real-time remaining capacity

### O5. Additional Hazards

- **Tsunami**: include inundation zone maps; coastal evacuation uphill routing
- **Industrial accident**: toxic plume dispersion → dynamic risk field
- **Wildfire**: spread model integration (fire perimeter → progressive edge removal)
- **Multi-hazard**: compound events (earthquake + tsunami, earthquake + landslide)
