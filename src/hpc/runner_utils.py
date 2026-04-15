"""
Shared helpers for all pipeline runners (naive, parallel, hpc/MPI).
"""

import logging
from src.data.models import DisasterType

logger = logging.getLogger(__name__)


def resolve_hazard_layers(cfg, disaster) -> dict:
    """
    Return {DisasterType: weight} for risk enrichment.
    Uses cfg.routing.hazard_layers when configured; falls back to
    {disaster.disaster_type: 1.0} for single-layer scenarios.
    """
    raw = cfg.routing.hazard_layers  # {str: float}
    if raw:
        resolved = {}
        for name, weight in raw.items():
            try:
                resolved[DisasterType(name)] = float(weight)
            except ValueError:
                logger.warning(f"Unknown hazard layer '{name}' in hazard_layers — skipping")
        if resolved:
            return resolved
    return {disaster.disaster_type: 1.0}
