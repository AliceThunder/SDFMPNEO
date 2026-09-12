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

    @property
    def incomplete(self) -> bool:
        return any(record.passed is None for record in self.gates)


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
    active_subspace_detail: str | None = None,
) -> ProductionReadinessReport:
    """Evaluate documented gates without silently inventing engineering limits.

    Gates 2--4 are diagnostic gates: POD compressibility may choose a larger rank,
    active-subspace analysis need not justify input compression, and stability may
    legitimately be non-contractive. They pass when the corresponding analysis
    has actually been performed, not when a preferred outcome was obtained.

    Gate 7 is deliberately fail-closed. A model cannot be declared production
    ready until at least one real wall-clock performance budget is supplied and
    the corresponding measurement has been recorded. Missing performance data
    yields ``passed=None`` rather than an accidental pass.
    """
    records: list[GateRecord] = []

    q1 = quadratic_identity_report.maximum_relative_error <= budgets.quadratic_identity_max_relative
    records.append(
        GateRecord(
            "Gate 1: quadratic identity",
            q1,
            f"max_rel={quadratic_identity_report.maximum_relative_error:.6g}",
        )
    )
    records.append(
        GateRecord(
            "Gate 2: tensor compressibility",
            bool(pod_assessed),
            "POD/SVD assessment completed" if pod_assessed else "missing POD/SVD assessment",
        )
    )
    gate3_detail = active_subspace_detail
    if gate3_detail is None:
        gate3_detail = (
            "sensitivity spectrum assessed"
            if active_subspace_assessed
            else "missing full sensitivity spectrum"
        )
    records.append(
        GateRecord(
            "Gate 3: thermal active subspace",
            bool(active_subspace_assessed),
            str(gate3_detail),
        )
    )
    records.append(
        GateRecord(
            "Gate 4: stability region",
            bool(stability_assessed),
            "logarithmic-norm region assessed" if stability_assessed else "missing stability analysis",
        )
    )

    gate5 = (
        surrogate_report.maximum_relative_heat_error <= budgets.heat_max_relative
        and vector_field_report.maximum_relative_error <= budgets.vector_field_max_relative
    )
    records.append(
        GateRecord(
            "Gate 5: neural learnability",
            gate5,
            "heat_max="
            f"{surrogate_report.maximum_relative_heat_error:.6g}, "
            f"field_max={vector_field_report.maximum_relative_error:.6g}",
        )
    )

    gate6 = trajectory_report.maximum_coordinate_error <= budgets.trajectory_max_coordinate_error
    if budgets.trajectory_max_temperature_error is not None:
        gate6 = (
            gate6
            and trajectory_report.maximum_temperature_error is not None
            and trajectory_report.maximum_temperature_error <= budgets.trajectory_max_temperature_error
        )
    gate6 = gate6 and bool(long_time_checked)
    records.append(
        GateRecord(
            "Gate 6: trajectory/output",
            gate6,
            f"coord_max={trajectory_report.maximum_coordinate_error:.6g}, "
            f"long_time_checked={bool(long_time_checked)}",
        )
    )

    timing_budget_supplied = (
        budgets.vector_field_seconds is not None
        or budgets.trajectory_query_seconds is not None
    )
    details: list[str] = []
    performance_ok = True
    benchmark_complete = True
    if budgets.vector_field_seconds is not None:
        benchmark_complete = benchmark_complete and vector_field_seconds is not None
        performance_ok = (
            performance_ok
            and vector_field_seconds is not None
            and vector_field_seconds <= budgets.vector_field_seconds
        )
        details.append(
            f"field_s={vector_field_seconds}, budget={budgets.vector_field_seconds}"
        )
    if budgets.trajectory_query_seconds is not None:
        benchmark_complete = benchmark_complete and trajectory_query_seconds is not None
        performance_ok = (
            performance_ok
            and trajectory_query_seconds is not None
            and trajectory_query_seconds <= budgets.trajectory_query_seconds
        )
        details.append(
            f"query_s={trajectory_query_seconds}, budget={budgets.trajectory_query_seconds}"
        )

    if not timing_budget_supplied:
        gate7: bool | None = None
        detail = "no production timing budget supplied"
    elif not benchmark_complete:
        gate7 = None
        detail = ", ".join(details) + "; required wall-clock measurement missing"
    else:
        gate7 = bool(performance_ok)
        detail = ", ".join(details)
    records.append(GateRecord("Gate 7: performance", gate7, detail))

    operational = bool(reproducible_training and persistence_roundtrip)
    records.append(
        GateRecord(
            "Operational reproducibility",
            operational,
            f"training={bool(reproducible_training)}, persistence={bool(persistence_roundtrip)}",
        )
    )
    ready = all(record.passed is True for record in records)
    return ProductionReadinessReport(tuple(records), bool(ready))


__all__ = [
    "GateRecord",
    "ProductionBudgets",
    "ProductionReadinessReport",
    "evaluate_production_readiness",
]
