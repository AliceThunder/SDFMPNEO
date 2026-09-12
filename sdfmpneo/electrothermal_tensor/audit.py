"""End-to-end execution of the documented Gate 1--7 validation protocol."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .benchmark import benchmark_trajectory_queries, benchmark_vector_field
from .gates import ProductionBudgets, ProductionReadinessReport, evaluate_production_readiness
from .pod import PODRankDiagnostic, pod_rank_sweep
from .stability import analyze_stability_callbacks
from .validation import (
    ActiveSubspaceReport,
    QuadraticIdentityReport,
    StabilityReport,
    SurrogateValidationReport,
    TrajectoryValidationReport,
    VectorFieldValidationReport,
    active_subspace_spectrum,
    finite_difference_tensor_jacobian,
    validate_quadratic_identity,
    validate_surrogate_on_dataset,
    validate_trajectory,
    validate_vector_field,
)


@dataclass(frozen=True)
class TrajectoryAuditCase:
    name: str
    times: np.ndarray
    initial_state: np.ndarray
    geometry: np.ndarray
    operating: np.ndarray
    neural_max_step: float
    long_time_check: bool = False

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float).reshape(-1)
        state = np.asarray(self.initial_state, dtype=float).reshape(-1)
        geometry = np.asarray(self.geometry, dtype=float).reshape(-1)
        operating = np.asarray(self.operating, dtype=float).reshape(-1)
        if not self.name:
            raise ValueError("trajectory audit case requires a name")
        if times.size < 1 or np.any(~np.isfinite(times)) or np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
            raise ValueError("trajectory audit times must be sorted, finite and non-negative")
        if not np.isfinite(float(self.neural_max_step)) or float(self.neural_max_step) <= 0.0:
            raise ValueError("trajectory neural_max_step must be finite and positive")
        if np.any(~np.isfinite(state)) or np.any(~np.isfinite(geometry)) or np.any(~np.isfinite(operating)):
            raise ValueError("trajectory audit inputs must be finite")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "operating", operating)


@dataclass(frozen=True)
class GateSuiteConfig:
    pod_ranks: tuple[int, ...] = (8, 16, 32, 64, 128)
    operating_samples_per_state: int = 4
    active_subspace_sample_count: int = 4
    active_subspace_relative_step: float = 1e-5
    benchmark_repeats: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.operating_samples_per_state) < 1:
            raise ValueError("operating_samples_per_state must be positive")
        if int(self.active_subspace_sample_count) < 1:
            raise ValueError("active_subspace_sample_count must be positive")
        if float(self.active_subspace_relative_step) <= 0.0:
            raise ValueError("active_subspace_relative_step must be positive")
        if int(self.benchmark_repeats) < 1:
            raise ValueError("benchmark_repeats must be positive")


@dataclass(frozen=True)
class GateSuiteReport:
    quadratic_identity: QuadraticIdentityReport
    pod_diagnostics: tuple[PODRankDiagnostic, ...]
    active_subspace: ActiveSubspaceReport
    physical_stability: StabilityReport
    surrogate: SurrogateValidationReport
    vector_field: VectorFieldValidationReport
    trajectories: tuple[tuple[str, TrajectoryValidationReport], ...]
    aggregate_trajectory: TrajectoryValidationReport
    vector_field_seconds: float | None
    trajectory_query_seconds: float | None
    readiness: ProductionReadinessReport


def _aggregate_trajectories(reports: list[TrajectoryValidationReport]) -> TrajectoryValidationReport:
    if not reports:
        raise ValueError("at least one trajectory audit case is required")
    temperature = [v.maximum_temperature_error for v in reports if v.maximum_temperature_error is not None]
    temperature_scalar = [
        v.maximum_temperature_scalar_error
        for v in reports
        if v.maximum_temperature_scalar_error is not None
    ]
    return TrajectoryValidationReport(
        sample_count=sum(v.sample_count for v in reports),
        maximum_coordinate_error=max(v.maximum_coordinate_error for v in reports),
        final_coordinate_error=max(v.final_coordinate_error for v in reports),
        maximum_relative_coordinate_error=max(v.maximum_relative_coordinate_error for v in reports),
        maximum_temperature_error=max(temperature) if temperature else None,
        maximum_temperature_scalar_error=max(temperature_scalar) if temperature_scalar else None,
    )


def _available_pod_ranks(dataset, requested: tuple[int, ...]) -> tuple[int, ...]:
    train_count = len(dataset.indices("train"))
    maximum = min(train_count, dataset.outputs.shape[1])
    ranks = tuple(sorted({int(v) for v in requested if 1 <= int(v) <= maximum}))
    if not ranks:
        ranks = (maximum,)
    return ranks


def run_gate_suite(
    *,
    model,
    dataset,
    pod,
    tensor_factory,
    direct_heat_factory,
    physical_vector_field,
    physical_jacobian_factory,
    thermal_operators,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
    trajectory_cases: tuple[TrajectoryAuditCase, ...],
    budgets: ProductionBudgets,
    temperature_reconstructor=None,
    config: GateSuiteConfig | None = None,
    reproducible_training: bool,
    persistence_roundtrip: bool,
) -> GateSuiteReport:
    """Run all documented gates on frozen data and explicit trajectory cases.

    The routine never feeds validation/test failures back into training.  It is
    therefore safe to archive the returned report as frozen evidence.  Expensive
    Gate 3 finite differences and Gate 4 physical Jacobians use the real physics
    callbacks supplied by the application, not derivatives of the neural model.
    """
    cfg = GateSuiteConfig() if config is None else config
    test_ids = dataset.indices("test")
    if len(test_ids) < 1:
        raise ValueError("frozen test split is empty")
    lo = np.asarray(operating_lower, dtype=float).reshape(-1)
    hi = np.asarray(operating_upper, dtype=float).reshape(-1)
    if lo.shape != (dataset.current_dimension,) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("operating bounds mismatch")

    states = dataset.states[test_ids]
    geometries = dataset.geometries[test_ids]
    rng = np.random.default_rng(int(cfg.seed))
    operating = rng.uniform(lo, hi, size=(len(test_ids), dataset.current_dimension))

    quadratic = validate_quadratic_identity(
        tensor_factory,
        direct_heat_factory,
        states,
        geometries,
        operating,
    )

    pod_diagnostics = pod_rank_sweep(
        dataset,
        _available_pod_ranks(dataset, cfg.pod_ranks),
        operating_lower=lo,
        operating_upper=hi,
        operating_samples_per_state=cfg.operating_samples_per_state,
        seed=cfg.seed,
    )

    active_ids = test_ids[: min(len(test_ids), int(cfg.active_subspace_sample_count))]
    jacobians = np.asarray(
        [
            finite_difference_tensor_jacobian(
                tensor_factory,
                dataset.states[index],
                dataset.geometries[index],
                relative_step=cfg.active_subspace_relative_step,
            )
            for index in active_ids
        ]
    )
    active = active_subspace_spectrum(jacobians)

    physical_stability = analyze_stability_callbacks(
        physical_jacobian_factory,
        thermal_operators,
        states,
        geometries,
        operating,
    )

    surrogate_report = validate_surrogate_on_dataset(
        model.surrogate,
        dataset,
        operating_lower=lo,
        operating_upper=hi,
        split="test",
        operating_samples_per_state=cfg.operating_samples_per_state,
        seed=cfg.seed,
    )
    vector_report = validate_vector_field(
        model.field,
        states,
        geometries,
        operating,
        physical_vector_field,
    )

    trajectory_pairs: list[tuple[str, TrajectoryValidationReport]] = []
    for case in trajectory_cases:
        report = validate_trajectory(
            model,
            physical_vector_field,
            case.times,
            initial_state=case.initial_state,
            geometry=case.geometry,
            operating=case.operating,
            neural_max_step=case.neural_max_step,
            temperature_reconstructor=temperature_reconstructor,
        )
        trajectory_pairs.append((case.name, report))
    aggregate = _aggregate_trajectories([value for _, value in trajectory_pairs])

    vector_seconds = None
    query_seconds = None
    if budgets.vector_field_seconds is not None:
        timing = benchmark_vector_field(
            model.field,
            states[0],
            geometries[0],
            operating[0],
            repeats=cfg.benchmark_repeats,
        )
        vector_seconds = timing.median_seconds
    if budgets.trajectory_query_seconds is not None:
        case = trajectory_cases[-1]
        timing = benchmark_trajectory_queries(
            model,
            [float(case.times[-1])],
            initial_state=case.initial_state,
            geometry=case.geometry,
            operating=case.operating,
            max_step=case.neural_max_step,
        )
        query_seconds = timing[float(case.times[-1])].median_seconds

    long_time_checked = any(case.long_time_check for case in trajectory_cases)
    readiness = evaluate_production_readiness(
        budgets=budgets,
        quadratic_identity_report=quadratic,
        surrogate_report=surrogate_report,
        vector_field_report=vector_report,
        trajectory_report=aggregate,
        pod_assessed=bool(pod_diagnostics),
        active_subspace_assessed=True,
        stability_assessed=True,
        long_time_checked=long_time_checked,
        reproducible_training=reproducible_training,
        persistence_roundtrip=persistence_roundtrip,
        vector_field_seconds=vector_seconds,
        trajectory_query_seconds=query_seconds,
    )
    return GateSuiteReport(
        quadratic_identity=quadratic,
        pod_diagnostics=pod_diagnostics,
        active_subspace=active,
        physical_stability=physical_stability,
        surrogate=surrogate_report,
        vector_field=vector_report,
        trajectories=tuple(trajectory_pairs),
        aggregate_trajectory=aggregate,
        vector_field_seconds=vector_seconds,
        trajectory_query_seconds=query_seconds,
        readiness=readiness,
    )


__all__ = [
    "GateSuiteConfig",
    "GateSuiteReport",
    "TrajectoryAuditCase",
    "run_gate_suite",
]
