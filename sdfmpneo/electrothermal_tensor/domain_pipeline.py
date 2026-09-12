"""Training entry points that consume a frozen reachable-state domain report."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .domain import (
    ReachableStateDomainReport,
    load_reachable_state_domain_report,
    validated_state_bounds,
)
from .pipeline import build_fixed_neural_rom, build_geometry_neural_rom
from .signatures import fixed_research_physical_signature, geometry_research_physical_signature


def _report(value) -> ReachableStateDomainReport:
    if isinstance(value, ReachableStateDomainReport):
        return value
    return load_reachable_state_domain_report(Path(value))


def build_fixed_neural_rom_from_domain_report(
    physical_model,
    domain_report,
    *,
    operating_lower,
    operating_upper,
    **pipeline_options,
):
    """Build a fixed-geometry neural ROM using a signature-bound state box."""
    report = _report(domain_report)
    signature = fixed_research_physical_signature(physical_model)
    operating_lower = np.asarray(operating_lower, dtype=float).reshape(-1)
    operating_upper = np.asarray(operating_upper, dtype=float).reshape(-1)
    rank = int(physical_model.core.thermal_model.rank)
    state_lower, state_upper = validated_state_bounds(
        report,
        expected_physical_signature=signature,
        geometry_lower=np.empty(0),
        geometry_upper=np.empty(0),
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        thermal_rank=rank,
    )
    return build_fixed_neural_rom(
        physical_model,
        state_lower=state_lower,
        state_upper=state_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        physical_signature=signature,
        **pipeline_options,
    )


def build_geometry_neural_rom_from_domain_report(
    geometry_model,
    domain_report,
    *,
    operating_lower,
    operating_upper,
    **pipeline_options,
):
    """Build a geometry-family neural ROM using a signature-bound state box."""
    report = _report(domain_report)
    signature = geometry_research_physical_signature(geometry_model)
    operating_lower = np.asarray(operating_lower, dtype=float).reshape(-1)
    operating_upper = np.asarray(operating_upper, dtype=float).reshape(-1)
    n_geometry = len(geometry_model.geometry_names)
    rank = int(geometry_model.reference.core.thermal_model.rank)
    state_lower, state_upper = validated_state_bounds(
        report,
        expected_physical_signature=signature,
        geometry_lower=-np.ones(n_geometry),
        geometry_upper=np.ones(n_geometry),
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        thermal_rank=rank,
    )
    return build_geometry_neural_rom(
        geometry_model,
        state_lower=state_lower,
        state_upper=state_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        physical_signature=signature,
        **pipeline_options,
    )


__all__ = [
    "build_fixed_neural_rom_from_domain_report",
    "build_geometry_neural_rom_from_domain_report",
]
