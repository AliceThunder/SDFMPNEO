from types import SimpleNamespace

import numpy as np

from sdfmpneo.electrothermal_tensor.audit import (
    GateSuiteConfig,
    TrajectoryAuditCase,
    run_gate_suite,
)
from sdfmpneo.electrothermal_tensor.dataset import QuadraticJouleDataset
from sdfmpneo.electrothermal_tensor.gates import (
    ProductionBudgets,
    evaluate_production_readiness,
)
from sdfmpneo.electrothermal_tensor.pod import fit_dataset_pod
from sdfmpneo.electrothermal_tensor.vector_field import FixedThermalOperatorFamily


class _ZeroSurrogate:
    state_dimension = 1
    geometry_dimension = 0

    class _Pod:
        current_dimension = 1

    pod = _Pod()

    def predict_packed_numpy(self, state, geometry):
        return np.zeros(3)


class _ExactDecayField:
    def __init__(self):
        self.thermal_operators = FixedThermalOperatorFamily(
            np.array([[1.0]]), np.array([[1.0]]), geometry_dimension=0
        )

    def vector_field(self, state, geometry, operating):
        return -np.asarray(state, dtype=float)


class _ExactDecayModel:
    def __init__(self):
        self.surrogate = _ZeroSurrogate()
        self.field = _ExactDecayField()

    def predict(self, time, *, initial_state, geometry, operating, max_step, method="etd2"):
        del geometry, operating, max_step, method
        state = np.exp(-float(time)) * np.asarray(initial_state, dtype=float)
        return SimpleNamespace(state=state)


def _zero_tensor(state, geometry):
    del state, geometry
    return np.zeros((1, 2, 2))


def _zero_heat(state, geometry, operating):
    del state, geometry, operating
    return np.zeros(1)


def _physical_field(state, geometry, operating):
    del geometry, operating
    return -np.asarray(state, dtype=float)


def _physical_jacobian(state, geometry, operating):
    del state, geometry, operating
    return np.array([[-1.0]])


def test_missing_performance_evidence_cannot_be_production_ready():
    quadratic = SimpleNamespace(maximum_relative_error=0.0)
    surrogate = SimpleNamespace(maximum_relative_heat_error=0.0)
    field = SimpleNamespace(maximum_relative_error=0.0)
    trajectory = SimpleNamespace(maximum_coordinate_error=0.0, maximum_temperature_error=0.0)
    report = evaluate_production_readiness(
        budgets=ProductionBudgets(
            quadratic_identity_max_relative=1e-10,
            heat_max_relative=1e-3,
            vector_field_max_relative=1e-3,
            trajectory_max_coordinate_error=1e-3,
        ),
        quadratic_identity_report=quadratic,
        surrogate_report=surrogate,
        vector_field_report=field,
        trajectory_report=trajectory,
        pod_assessed=True,
        active_subspace_assessed=True,
        stability_assessed=True,
        long_time_checked=True,
        reproducible_training=True,
        persistence_roundtrip=True,
    )
    assert not report.ready
    assert report.incomplete
    assert next(record for record in report.gates if record.name.startswith("Gate 7")).passed is None


def test_gate_suite_runs_all_gates_on_an_exact_synthetic_system():
    rng = np.random.default_rng(7)
    states = rng.uniform(-0.5, 0.5, size=(30, 1))
    geometries = np.empty((30, 0))
    tensors = np.zeros((30, 1, 2, 2))
    dataset = QuadraticJouleDataset.from_tensors(
        states, geometries, tensors, split_seed=3
    )
    pod = fit_dataset_pod(dataset, rank=1)
    model = _ExactDecayModel()
    thermal = model.field.thermal_operators
    case = TrajectoryAuditCase(
        name="analytic-decay",
        times=np.array([0.0, 0.1, 0.5, 1.0]),
        initial_state=np.array([0.3]),
        geometry=np.empty(0),
        operating=np.array([0.2]),
        neural_max_step=0.05,
        long_time_check=True,
    )
    report = run_gate_suite(
        model=model,
        dataset=dataset,
        pod=pod,
        tensor_factory=_zero_tensor,
        direct_heat_factory=_zero_heat,
        physical_vector_field=_physical_field,
        physical_jacobian_factory=_physical_jacobian,
        thermal_operators=thermal,
        operating_lower=np.array([-1.0]),
        operating_upper=np.array([1.0]),
        trajectory_cases=(case,),
        budgets=ProductionBudgets(
            quadratic_identity_max_relative=1e-12,
            heat_max_relative=1e-12,
            vector_field_max_relative=1e-12,
            trajectory_max_coordinate_error=1e-7,
            vector_field_seconds=1.0,
            trajectory_query_seconds=1.0,
        ),
        config=GateSuiteConfig(
            pod_ranks=(1,),
            operating_samples_per_state=1,
            active_subspace_sample_count=1,
            benchmark_repeats=2,
            seed=4,
        ),
        reproducible_training=True,
        persistence_roundtrip=True,
    )
    assert report.quadratic_identity.maximum_relative_error == 0.0
    assert report.physical_stability.uniformly_contractive_on_samples
    assert report.aggregate_trajectory.maximum_coordinate_error < 1e-7
    assert report.readiness.ready
