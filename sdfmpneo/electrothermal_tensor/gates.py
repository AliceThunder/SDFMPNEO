"""Explicit Gate 1--7 production-readiness accounting."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductionBudgets:
    quadratic_identity_max_relative: float
    heat_max_relative: float
    vector_field_max_relative: float
    trajectory_max_coordinate_error: float
    trajectory_max_temperature_error: float | None = None
    vector_field_seconds: float | None = None
    trajectory_query_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value is not None and float(value) <= 0.0:
                raise ValueError(f"{name} must be positive when specified")


@dataclass(frozen=True)
class GateRecord:
    name: str
    passed: bool | None
    detail: str


@dataclass(frozen=True)
class ProductionReadinessReport:
    gates: tuple[GateRecord, ...]
    ready: bool


def evaluate_production_readiness(
    *,
    budgets: ProductionBudgets,
    quadratic_identity_report,
    surrogate_report,
    vector_field_report,
    trajectory_report,
    pod_assessed: bool,
    active_subspace_assessed: bool,
    stability_assessed: bool,
    long_time_checked: bool,
    reproducible_training: bool,
    persistence_roundtrip: bool,
    vector_field_seconds: float | None = None,
    trajectory_query_seconds: float | None = None,
) -> ProductionReadinessReport:
    """Evaluate the documented gates without silently inventing engineering limits.

    Gates 2--4 are diagnostic gates: POD compressibility may choose a larger rank,
    active-subspace analysis need not justify input compression, and stability may
    legitimately be non-contractive. They pass when the corresponding analysis
    has actually been performed, not when a preferred outcome was obtained.
    """
    records = []
    q1 = quadratic_identity_report.maximum_relative_error <= budgets.quadratic_identity_max_relative
    records.append(GateRecord("Gate 1: quadratic identity", q1, f"max_rel={quadratic_identity_report.maximum_relative_error:.6g}"))
    records.append(GateRecord("Gate 2: tensor compressibility", bool(pod_assessed), "POD/SVD assessment completed" if pod_assessed else "missing POD/SVD assessment"))
    records.append(GateRecord("Gate 3: thermal active subspace", bool(active_subspace_assessed), "sensitivity spectrum assessed" if active_subspace_assessed else "missing sensitivity spectrum"))
    records.append(GateRecord("Gate 4: stability region", bool(stability_assessed), "logarithmic-norm region assessed" if stability_assessed else "missing stability analysis"))
    gate5 = (
        surrogate_report.maximum_relative_heat_error <= budgets.heat_max_relative
        and vector_field_report.maximum_relative_error <= budgets.vector_field_max_relative
    )
    records.append(GateRecord("Gate 5: neural learnability", gate5, f"heat_max={surrogate_report.maximum_relative_heat_error:.6g}, field_max={vector_field_report.maximum_relative_error:.6g}"))
    gate6 = trajectory_report.maximum_coordinate_error <= budgets.trajectory_max_coordinate_error
    if budgets.trajectory_max_temperature_error is not None:
        gate6 = gate6 and trajectory_report.maximum_temperature_error is not None and trajectory_report.maximum_temperature_error <= budgets.trajectory_max_temperature_error
    gate6 = gate6 and bool(long_time_checked)
    records.append(GateRecord("Gate 6: trajectory/output", gate6, f"coord_max={trajectory_report.maximum_coordinate_error:.6g}, long_time_checked={bool(long_time_checked)}"))

    performance_ok = True
    details = []
    if budgets.vector_field_seconds is not None:
        performance_ok = performance_ok and vector_field_seconds is not None and vector_field_seconds <= budgets.vector_field_seconds
        details.append(f"field_s={vector_field_seconds}")
    if budgets.trajectory_query_seconds is not None:
        performance_ok = performance_ok and trajectory_query_seconds is not None and trajectory_query_seconds <= budgets.trajectory_query_seconds
        details.append(f"query_s={trajectory_query_seconds}")
    records.append(GateRecord("Gate 7: performance", performance_ok, ", ".join(details) if details else "no hard timing budget supplied; benchmark recorded separately"))

    operational = bool(reproducible_training and persistence_roundtrip)
    records.append(GateRecord("Operational reproducibility", operational, f"training={bool(reproducible_training)}, persistence={bool(persistence_roundtrip)}"))
    ready = all(record.passed is True for record in records)
    return ProductionReadinessReport(tuple(records), bool(ready))


__all__ = [
    "GateRecord",
    "ProductionBudgets",
    "ProductionReadinessReport",
    "evaluate_production_readiness",
]
