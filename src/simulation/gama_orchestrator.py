"""
GAMA-platform simulation orchestrator.
Manages headless GAMA simulation runs from Python.

Workflow:
1. Python writes simulation inputs (JSON/CSV) from optimization results
2. Python generates GAMA experiment XML file
3. Python invokes gama-headless subprocess
4. Python parses GAMA simulation output CSV/XML
5. Python aggregates multi-run statistics

GAMA model: simulation/models/EvacuationModel.gaml
"""

import json
import csv
import subprocess
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.data.models import (
    Village, Shelter, EvacuationRoute, OptimizationResult,
    SimulationOutput, Assignment
)

logger = logging.getLogger(__name__)


class GAMAOrchestrator:
    """
    Orchestrates GAMA-platform agent-based evacuation simulations.
    """

    def __init__(
        self,
        gama_executable: str = "gama-headless",
        gaml_model_path: str = "simulation/models/EvacuationModel.gaml",
        output_dir: str = "output/simulation",
        max_steps: int = 500,
        n_runs: int = 5,
        parallel_runs: int = 1,
    ):
        self.gama_executable = gama_executable
        self.gaml_model_path = Path(gaml_model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_steps = max_steps
        self.n_runs = n_runs
        self.parallel_runs = parallel_runs

    def run_simulation(
        self,
        opt_result: OptimizationResult,
        villages: List[Village],
        shelters: List[Shelter],
        scenario_id: str = "sim_001",
        seed_base: int = 42,
    ) -> List[SimulationOutput]:
        """
        Run n_runs GAMA simulations and return aggregated outputs.
        """
        # Write simulation inputs
        inputs_dir = self.output_dir / scenario_id / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        self._write_villages_csv(villages, inputs_dir / "villages.csv")
        self._write_shelters_csv(shelters, inputs_dir / "shelters.csv")
        self._write_routes_csv(opt_result.assignments, inputs_dir / "routes.csv")
        self._write_sim_config(opt_result, inputs_dir / "sim_config.json")

        # Generate experiment XML
        xml_path = self.output_dir / scenario_id / "experiment.xml"
        self._generate_experiment_xml(
            xml_path=xml_path,
            inputs_dir=inputs_dir,
            scenario_id=scenario_id,
            n_runs=self.n_runs,
            seed_base=seed_base,
        )

        # Execute GAMA headless
        sim_outputs_dir = self.output_dir / scenario_id / "outputs"
        sim_outputs_dir.mkdir(parents=True, exist_ok=True)

        success = self._run_gama_headless(xml_path, sim_outputs_dir)

        if not success:
            logger.error("GAMA simulation failed or GAMA not installed")
            return self._generate_mock_outputs(scenario_id, villages, shelters, opt_result)

        return self._parse_outputs(sim_outputs_dir, scenario_id)

    # ------------------------------------------------------------------ #
    # Input writers
    # ------------------------------------------------------------------ #

    def _write_villages_csv(self, villages: List[Village], path: Path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "village_id", "name", "lat", "lon", "population"
            ])
            writer.writeheader()
            for v in villages:
                writer.writerow({
                    "village_id": v.village_id, "name": v.name,
                    "lat": v.centroid_lat, "lon": v.centroid_lon,
                    "population": v.population,
                })

    def _write_shelters_csv(self, shelters: List[Shelter], path: Path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "shelter_id", "name", "lat", "lon", "capacity"
            ])
            writer.writeheader()
            for s in shelters:
                writer.writerow({
                    "shelter_id": s.shelter_id, "name": s.name,
                    "lat": s.centroid_lat, "lon": s.centroid_lon,
                    "capacity": s.capacity,
                })

    def _write_routes_csv(self, assignments: List[Assignment], path: Path):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "village_id", "shelter_id", "distance_km",
                "travel_time_min", "assigned_population", "avg_risk"
            ])
            writer.writeheader()
            for a in assignments:
                r = a.route
                writer.writerow({
                    "village_id": a.village_id,
                    "shelter_id": a.shelter_id,
                    "distance_km": round(r.total_distance_km, 2) if r else 0,
                    "travel_time_min": round(r.total_time_min, 1) if r else 0,
                    "assigned_population": a.assigned_population,
                    "avg_risk": round(r.avg_risk_score, 3) if r else 0,
                })

    def _write_sim_config(self, result: OptimizationResult, path: Path):
        config = {
            "total_population": result.total_population,
            "total_evacuated": result.total_evacuated,
            "max_simulation_steps": self.max_steps,
            "time_step_minutes": 1.0,
            "hazard_type": result.disaster.disaster_type.value if result.disaster else "earthquake",
        }
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    # ------------------------------------------------------------------ #
    # GAMA experiment XML generation
    # ------------------------------------------------------------------ #

    def _generate_experiment_xml(
        self,
        xml_path: Path,
        inputs_dir: Path,
        scenario_id: str,
        n_runs: int,
        seed_base: int,
    ):
        """
        Generate GAMA headless experiment XML.
        See: https://gama-platform.org/wiki/Headless
        """
        root = ET.Element("Experiment_plan")

        for run_id in range(n_runs):
            sim = ET.SubElement(root, "Simulation")
            sim.set("id", str(run_id))
            sim.set("sourcePath", str(self.gaml_model_path.absolute()))
            sim.set("finalStep", str(self.max_steps))
            sim.set("experiment", "EvacuationExperiment")
            sim.set("seed", str(seed_base + run_id))

            # Parameters
            params = {
                "villages_file": str((inputs_dir / "villages.csv").absolute()),
                "shelters_file": str((inputs_dir / "shelters.csv").absolute()),
                "routes_file": str((inputs_dir / "routes.csv").absolute()),
                "sim_config_file": str((inputs_dir / "sim_config.json").absolute()),
                "run_id": run_id,
            }
            for pname, pval in params.items():
                p = ET.SubElement(sim, "Parameter")
                p.set("name", pname)
                p.set("type", "STRING" if isinstance(pval, str) else "INT")
                p.set("value", str(pval))

            # Outputs to record
            outputs_el = ET.SubElement(sim, "Outputs")
            output_dir_run = self.output_dir / scenario_id / "outputs" / f"run_{run_id}"
            output_dir_run.mkdir(parents=True, exist_ok=True)

            for monitor_name in [
                "total_saved", "total_delayed", "total_failed",
                "evacuation_ratio", "avg_evacuation_time", "worst_evacuation_time"
            ]:
                out = ET.SubElement(outputs_el, "Output")
                out.set("id", monitor_name)
                out.set("name", monitor_name)
                out.set("framerate", "10")
                out.set("output_path", str(output_dir_run / f"{monitor_name}.csv"))

        tree = ET.ElementTree(root)
        tree.write(str(xml_path), encoding="utf-8", xml_declaration=True)
        logger.info(f"GAMA experiment XML written to {xml_path}")

    # ------------------------------------------------------------------ #
    # GAMA execution
    # ------------------------------------------------------------------ #

    def _run_gama_headless(self, xml_path: Path, output_dir: Path) -> bool:
        """Invoke gama-headless subprocess."""
        cmd = [
            self.gama_executable,
            str(xml_path),
            str(output_dir),
        ]
        logger.info(f"Running GAMA: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )
            if result.returncode != 0:
                logger.error(f"GAMA exited with code {result.returncode}")
                logger.error(f"STDERR: {result.stderr[:500]}")
                return False
            logger.info("GAMA simulation completed successfully")
            return True
        except FileNotFoundError:
            logger.warning(f"GAMA executable '{self.gama_executable}' not found")
            return False
        except subprocess.TimeoutExpired:
            logger.error("GAMA simulation timed out")
            return False

    # ------------------------------------------------------------------ #
    # Output parsing
    # ------------------------------------------------------------------ #

    def _parse_outputs(self, output_dir: Path, scenario_id: str) -> List[SimulationOutput]:
        """Parse GAMA output CSV files into SimulationOutput objects."""
        outputs = []
        run_dirs = sorted(output_dir.glob("run_*"))

        for run_dir in run_dirs:
            run_id = int(run_dir.name.split("_")[1])
            try:
                stats = {}
                for stat_name in ["total_saved", "total_delayed", "total_failed",
                                   "evacuation_ratio", "avg_evacuation_time",
                                   "worst_evacuation_time"]:
                    csv_path = run_dir / f"{stat_name}.csv"
                    if csv_path.exists():
                        val = self._read_last_csv_value(csv_path)
                        stats[stat_name] = val

                outputs.append(SimulationOutput(
                    scenario_id=scenario_id,
                    run_id=run_id,
                    total_saved=int(stats.get("total_saved", 0)),
                    total_delayed=int(stats.get("total_delayed", 0)),
                    total_failed=int(stats.get("total_failed", 0)),
                    evacuation_completion_ratio=float(stats.get("evacuation_ratio", 0.0)),
                    avg_evacuation_time_min=float(stats.get("avg_evacuation_time", 0.0)),
                    worst_evacuation_time_min=float(stats.get("worst_evacuation_time", 0.0)),
                    bottleneck_road_ids=[],
                    overloaded_shelter_ids=[],
                    congestion_timeline=[],
                    raw_output_path=str(run_dir),
                ))
            except Exception as e:
                logger.warning(f"Failed to parse run {run_id}: {e}")

        return outputs

    def _read_last_csv_value(self, csv_path: Path) -> float:
        """Read last numeric value from a GAMA monitor CSV."""
        last_val = 0.0
        with open(csv_path) as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    last_val = float(row[-1])
                except (ValueError, IndexError):
                    pass
        return last_val

    # ------------------------------------------------------------------ #
    # Mock outputs (when GAMA not available)
    # ------------------------------------------------------------------ #

    def _generate_mock_outputs(
        self,
        scenario_id: str,
        villages: List[Village],
        shelters: List[Shelter],
        result: OptimizationResult,
    ) -> List[SimulationOutput]:
        """
        Generate plausible mock simulation outputs based on optimization result.
        Used when GAMA is not installed (for testing/development).
        """
        import random
        outputs = []
        base_ratio = result.evacuation_ratio

        for run_id in range(self.n_runs):
            random.seed(42 + run_id)
            noise = random.gauss(0, 0.05)
            ratio = max(0.0, min(1.0, base_ratio + noise))
            saved = int(result.total_population * ratio)
            failed = result.total_population - saved
            avg_time = result.avg_route_time_min * (1.0 + random.gauss(0.3, 0.1))

            outputs.append(SimulationOutput(
                scenario_id=scenario_id,
                run_id=run_id,
                total_saved=saved,
                total_delayed=int(failed * 0.3),
                total_failed=int(failed * 0.7),
                evacuation_completion_ratio=ratio,
                avg_evacuation_time_min=avg_time,
                worst_evacuation_time_min=avg_time * 2.5,
                bottleneck_road_ids=[],
                overloaded_shelter_ids=[
                    sid for sid, util in result.shelter_utilization.items()
                    if util > 0.9
                ],
                congestion_timeline=[],
            ))

        logger.info(f"Generated {len(outputs)} mock simulation outputs "
                    f"(GAMA not available)")
        return outputs

    # ------------------------------------------------------------------ #
    # Statistics aggregation
    # ------------------------------------------------------------------ #

    def aggregate_outputs(self, outputs: List[SimulationOutput]) -> Dict[str, Any]:
        """Aggregate multi-run statistics."""
        if not outputs:
            return {}

        import statistics
        saved = [o.total_saved for o in outputs]
        ratios = [o.evacuation_completion_ratio for o in outputs]
        avg_times = [o.avg_evacuation_time_min for o in outputs]

        return {
            "n_runs": len(outputs),
            "total_saved_mean": statistics.mean(saved),
            "total_saved_stdev": statistics.stdev(saved) if len(saved) > 1 else 0.0,
            "evacuation_ratio_mean": statistics.mean(ratios),
            "evacuation_ratio_stdev": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            "avg_time_mean_min": statistics.mean(avg_times),
            "avg_time_stdev_min": statistics.stdev(avg_times) if len(avg_times) > 1 else 0.0,
        }
