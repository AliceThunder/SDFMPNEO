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
from .provenance import state_domain_report_hash
from .signatures import fixed_research_physical_signature, geometry_research_physical_signature


def _report_and_metadata(value) -> tuple[ReachableStateDomainReport, dict]:
    if isinstance(value, ReachableStateDomainReport):
        report = value
        source = None
    else:
        source_path = Path(value).expanduser().resolve()
        report = load_reachable_state_domain_report(source_path)
        source = str(source_path)
    metadata = {"state_domain_report_hash": state_domain_report_hash(report)}
    if source is not None:
        metadata["state_domain_report"] = source
    return report, metadata


def _domain_dataset_metadata(pipeline_options: dict, report_metadata: dict) -> dict:
    existing = dict(pipeline_options.pop("dataset_metadata_extra", {}) or {})
    conflict = sorted(set(existing).intersection(report_metadata))
    if conflict:
        raise ValueError(f"dataset metadata already defines domain provenance keys: {conflict}")
    existing.update(report_metadata)
    return existing


def build_fixed_neural_rom_from_domain_report(
    physical_model,
    domain_report,
    *,
    operating_lower,
    operating_upper,
    **pipeline_options,
):
    """Build a fixed-geometry neural ROM using a signature-bound state box."""
    report, report_metadata = _report_and_metadata(domain_report)
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
    options = dict(pipeline_options)
    options["dataset_metadata_extra"] = _domain_dataset_metadata(options, report_metadata)
    return build_fixed_neural_rom(
        physical_model,
        state_lower=state_lower,
        state_upper=state_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        physical_signature=signature,
        **options,
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
    report, report_metadata = _report_and_metadata(domain_report)
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
    options = dict(pipeline_options)
    options["dataset_metadata_extra"] = _domain_dataset_metadata(options, report_metadata)
    return build_geometry_neural_rom(
        geometry_model,
        state_lower=state_lower,
        state_upper=state_upper,
        operating_lower=operating_lower,
        operating_upper=operating_upper,
        physical_signature=signature,
        **options,
    )


__all__ = [
    "build_fixed_neural_rom_from_domain_report",
    "build_geometry_neural_rom_from_domain_report",
]
