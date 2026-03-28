"""
Population and shelter capacity loader.
Supports:
  - CSV files with village_id/shelter_id + population/capacity
  - Estimation from polygon area if no data available
"""

import csv
import logging
from pathlib import Path
from typing import List, Dict, Optional

from src.data.models import Village, Shelter

logger = logging.getLogger(__name__)

# Estimated m² floor area per shelter occupant
DEFAULT_M2_PER_SHELTER_PERSON = 2.0

# Population density fallback (persons / km²)
DEFAULT_POPULATION_DENSITY_PER_KM2 = 500.0


class PopulationLoader:
    """Loads or estimates population counts for villages."""

    def load_from_csv(self, villages: List[Village], csv_path: str) -> int:
        """
        Load population from CSV.
        Expected columns: name (or village_id), population
        Returns number of villages matched.
        """
        lookup = self._load_csv_lookup(csv_path, key_cols=["village_id", "name", "id"],
                                       value_col="population")
        matched = 0
        for v in villages:
            pop = lookup.get(v.village_id) or lookup.get(v.name.lower())
            if pop is not None:
                v.population = int(pop)
                matched += 1

        logger.info(f"Matched {matched}/{len(villages)} villages from CSV")
        return matched

    def estimate_from_area(
        self,
        villages: List[Village],
        density_per_km2: float = DEFAULT_POPULATION_DENSITY_PER_KM2,
        max_village_population: int = 10_000,
    ) -> None:
        """
        Estimate population from polygon area for villages with population=0.
        Assumes uniform density. Caps at max_village_population to prevent
        inflated estimates from large admin-boundary polygons.
        """
        for v in villages:
            if v.population == 0 and v.area_m2 > 0:
                area_km2 = v.area_m2 / 1_000_000.0
                estimated = int(area_km2 * density_per_km2)
                v.population = max(1, min(estimated, max_village_population))

        zeros = sum(1 for v in villages if v.population == 0)
        if zeros:
            logger.warning(
                f"{zeros} villages have population=0 (no area data); setting to 500 default"
            )
            for v in villages:
                if v.population == 0:
                    v.population = 500

    def apply_population(
        self,
        villages: List[Village],
        population_csv: Optional[str] = None,
        density_per_km2: float = DEFAULT_POPULATION_DENSITY_PER_KM2,
    ) -> None:
        """
        Apply population data: from CSV if available, else estimate from area.
        """
        matched = 0
        if population_csv and Path(population_csv).exists():
            matched = self.load_from_csv(villages, population_csv)

        # Estimate for remaining zeros
        unmatched = [v for v in villages if v.population == 0]
        if unmatched:
            logger.info(f"Estimating population for {len(unmatched)} villages from area")
            self.estimate_from_area(unmatched, density_per_km2)

    # ------------------------------------------------------------------ #

    def _load_csv_lookup(self, csv_path: str, key_cols: List[str], value_col: str) -> Dict:
        lookup = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = [c.lower() for c in (reader.fieldnames or [])]

            for row in reader:
                row_lower = {k.lower(): v for k, v in row.items()}
                for key_col in key_cols:
                    key_val = row_lower.get(key_col)
                    if key_val:
                        val = row_lower.get(value_col.lower(), row_lower.get("population", 0))
                        try:
                            lookup[str(key_val).lower()] = int(float(val))
                        except (ValueError, TypeError):
                            pass
        return lookup


class ShelterCapacityLoader:
    """Loads or estimates shelter capacities."""

    def load_from_csv(self, shelters: List[Shelter], csv_path: str) -> int:
        """Load capacity from CSV. Expected columns: shelter_id/name, capacity."""
        lookup = {}
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_lower = {k.lower(): v for k, v in row.items()}
                key = str(row_lower.get("shelter_id", row_lower.get("name", ""))).lower()
                cap_str = row_lower.get("capacity", "0")
                try:
                    cap = int(float(cap_str))
                    lookup[key] = cap
                except (ValueError, TypeError):
                    pass

        matched = 0
        for s in shelters:
            cap = lookup.get(s.shelter_id.lower()) or lookup.get(s.name.lower())
            if cap is not None:
                s.capacity = cap
                matched += 1

        logger.info(f"Matched {matched}/{len(shelters)} shelter capacities from CSV")
        return matched

    def estimate_from_area(
        self,
        shelters: List[Shelter],
        m2_per_person: float = DEFAULT_M2_PER_SHELTER_PERSON,
    ) -> None:
        """Estimate capacity from polygon area for shelters with capacity=0."""
        for s in shelters:
            if s.capacity == 0 and s.area_m2 > 0:
                s.capacity = max(10, int(s.area_m2 / m2_per_person))

        zeros = sum(1 for s in shelters if s.capacity == 0)
        if zeros:
            logger.warning(f"{zeros} shelters have no area data; using default capacity=200")
            for s in shelters:
                if s.capacity == 0:
                    s.capacity = 200

    def apply_capacity(
        self,
        shelters: List[Shelter],
        capacity_csv: Optional[str] = None,
        m2_per_person: float = DEFAULT_M2_PER_SHELTER_PERSON,
    ) -> None:
        """Apply capacity: from CSV if available, else estimate from area."""
        if capacity_csv and Path(capacity_csv).exists():
            self.load_from_csv(shelters, capacity_csv)
        unmatched = [s for s in shelters if s.capacity == 0]
        if unmatched:
            self.estimate_from_area(unmatched, m2_per_person)
