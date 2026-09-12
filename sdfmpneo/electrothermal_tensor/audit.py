"""End-to-end execution of the documented Gate 1--7 validation protocol."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .benchmark import benchmark_trajectory_queries, benchmark_vector_field
from .gates import ProductionBudgets, ProductionReadinessReport, evaluate_production_readiness
from .pod import PODRankDiagnostic, pod_rank_sweep
from .stability import analyze_stability_callbacks
from .symmetric import tensor_svec
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
    neural_method: str = "etd2"
    neural_rtol: float = 1e-5
    neural_atol: float = 1e-8
    neural_initial_step: float | None = None

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
        if self.neural_method not in {"etd2", "etd2_adaptive", "adaptive_etd2", "imex"}:
            raise ValueError("unsupported trajectory audit neural_method")
        if float(self.neural_rtol) <= 0.0 or float(self.neural_atol) <= 0.0:
            raise ValueError("trajectory neural tolerances must be positive")
        if self.neural_initial_step is not None and float(self.neural_initial_step) <= 0.0:
            raise ValueError("neural_initial_step must be positive when supplied")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "initial_state", state)
        object.__setattr__(self, "geometry", geometry)
        object.__setattr__(self, "operating", operating)

    def predict_options(self) -> dict:
        if self.neural_method in {"etd2_adaptive", "adaptive_etd2"}:
            return {
                "rtol": float(self.neural_rtol),
                "atol": float(self.neural_atol),
                "initial_step": self.neural_initial_step,
            }
        return {}


@dataclass(frozen=True)
class GateSuiteConfig:
    pod_ranks: tuple[int, ...] = (8, 16, 32, 64, 128)
    operating_samples_per_state: int = 4
    active_subspace_sample_count: int = 4
    active_subspace_relative_step: float = 1e-5
    active_subspace_mode: str = "full"
    randomized_direction_count: int = 16
    benchmark_repeats: int = 100
    trajectory_benchmark_repeats: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if int(self.operating_samples_per_state) < 1:
            raise ValueError("operating_samples_per_state must be positive")
        if int(self.active_subspace_sample_count) < 1:
            raise ValueError("active_subspace_sample_count must be positive")
        if float(self.active_subspace_relative_step) <= 0.0:
            raise ValueError("active_subspace_relative_step must be positive")
        if self.active_subspace_mode not in {"full", "randomized_screening"}:
            raise ValueError("active_subspace_mode must be 'full' or 'randomized_screening'")
        if int(self.randomized_direction_count) < 1:
            raise ValueError("randomized_direction_count must be positive")
        if int(self.benchmark_repeats) < 1 or int(self.trajectory_benchmark_repeats) < 1:
            raise ValueError("benchmark repeat counts must be positive")


@dataclass(frozen=True)
class GateSuiteReport:
    quadratic_identity: QuadraticIdentityReport
    pod_diagnostics: tuple[PODRankDiagnostic, ...]
    active_subspace: ActiveSubspaceReport
    active_subspace_method: str
    active_subspace_direction_count: int
    physical_stability: StabilityReport
    surrogate: SurrogateValidationReport
    vector_field: VectorFieldValidationReport
    trajectories: tuple[tuple[str, TrajectoryValidationReport], ...]
    aggregate_trajectory: TrajectoryValidationReport
    vector_field_seconds: float | None
    trajectory_query_seconds: float | None
    readiness: ProductionReadinessReport


class _TrajectoryMethodProxy:
    def __init__(self, model, case: TrajectoryAuditCase):
        self.model = model
        self.case = case

    def predict(self, time, *, initial_state, geometry, operating, max_step, method="etd2"):
        del method
        return self.model.predict(
            time,
            initial_state=initial_state,
            geometry=geometry,
            operating=operating,
            max_step=max_step,
            method=self.case.neural_method,
            **self.case.predict_options(),
        )


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


def _randomized_projected_tensor_jacobian(
    tensor_factory,
    state,
    geometry,
    directions: np.ndarray,
    *,
    relative_step: float,
) -> np.ndarray:
    """Approximate ``J`` by ``J QQ^T`` using central directional differences.

    This is a screening diagnostic only.  Its spectrum is the sensitivity seen
    in the chosen random input subspace and cannot certify that directions
    orthogonal to that subspace are unimportant.
    """
    a = np.asarray(state, dtype=float).reshape(-1)
    g = np.asarray(geometry, dtype=float).reshape(-1)
    Q = np.asarray(directions, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != a.size:
        raise ValueError("randomized sensitivity directions have wrong dimension")
    output_dimension = tensor_svec(np.asarray(tensor_factory(a, g), dtype=float)).size
    directional = np.empty((output_dimension, Q.shape[1]), dtype=float)
    state_scale = max(1.0, float(np.linalg.norm(a)))
    for k in range(Q.shape[1]):
        direction = Q[:, k]
        h = float(relative_step) * state_scale
        yp = tensor_svec(
            np.asarray(tensor_factory(a + h * direction, g), dtype=float)
        ).reshape(-1)
        ym = tensor_svec(
            np.asarray(tensor_factory(a - h * direction, g), dtype=float)
        ).reshape(-1)
        directional[:, k] = (yp - ym) / (2.0 * h)
    return directional @ Q.T


def _gate3_jacobians(tensor_factory, dataset, active_ids, cfg: GateSuiteConfig):
    if cfg.active_subspace_mode == "full":
        return np.asarray(
            [
                finite_difference_tensor_jacobian(
                    tensor_factory,
                    dataset.states[index],
                    dataset.geometries[index],
                    relative_step=cfg.active_subspace_relative_step,
                )
                for index in active_ids
            ]
        ), dataset.thermal_rank
    count = min(int(cfg.randomized_direction_count), dataset.thermal_rank)
    rng = np.random.default_rng(int(cfg.seed) + 101)
    raw = rng.normal(size=(dataset.thermal_rank, count))
    directions, _ = np.linalg.qr(raw, mode="reduced")
    jacobians = np.asarray(
        [
            _randomized_projected_tensor_jacobian(
                tensor_factory,
                dataset.states[index],
                dataset.geometries[index],
                directions,
                relative_step=cfg.active_subspace_relative_step,
            )
            for index in active_ids
        ]
    )
    return jacobians, count


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
    """Run all documented gates on frozen data and explicit trajectory cases."""
    cfg = GateSuiteConfig() if config is None else config
    if not trajectory_cases:
        raise ValueError("at least one trajectory audit case is required")
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
    jacobians, direction_count = _gate3_jacobians(
        tensor_factory, dataset, active_ids, cfg
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
            _TrajectoryMethodProxy(model, case),
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
            method=case.neural_method,
            predict_options=case.predict_options(),
            repeats=cfg.trajectory_benchmark_repeats,
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
        active_subspace_method=cfg.active_subspace_mode,
        active_subspace_direction_count=direction_count,
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
