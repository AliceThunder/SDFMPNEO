import numpy as np
import pytest
import scipy.linalg

from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM
from sdfmpneo.electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.electrothermal_tensor.surrogate import NeuralTensorSurrogate
from sdfmpneo.electrothermal_tensor.vector_field import FixedThermalOperatorFamily


def _model():
    torch = pytest.importorskip("torch")
    # One current parameter, but the Joule tensor is exactly zero. Dynamics are
    # therefore driven only by the deterministic hard thermal RHS forcing.
    pod = TensorPOD(
        mean=np.zeros(6),
        basis=np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        singular_values=np.array([1.0]),
        thermal_rank=2,
        current_dimension=1,
    )
    network = build_residual_mlp(
        ResidualMLPConfig(input_dimension=2, output_dimension=1, width=8, blocks=1),
        FeatureNormalizer(np.zeros(2), np.ones(2)),
    )
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    surrogate = NeuralTensorSurrogate(
        network,
        pod,
        state_dimension=2,
        geometry_dimension=0,
        coefficient_mean=np.array([0.0]),
        coefficient_scale=np.array([1.0]),
    )
    M = np.array([[1.7, 0.2], [0.2, 1.1]])
    K = np.array([[2.2, 0.15], [0.15, 1.4]])
    forcing = np.array([0.6, -0.25])
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        FixedThermalOperatorFamily(M, K),
        thermal_rhs_forcing=forcing,
        physical_signature="forcing-unit",
        training_domain={
            "state_lower": np.array([-2.0, -2.0]),
            "state_upper": np.array([2.0, 2.0]),
            "geometry_lower": np.empty(0),
            "geometry_upper": np.empty(0),
            "operating_lower": np.array([-1.0]),
            "operating_upper": np.array([1.0]),
        },
    )
    return model, M, K, forcing


def _exact_linear_state(M, K, forcing, a0, time):
    L = np.linalg.solve(M, K)
    steady = np.linalg.solve(K, forcing)
    return steady + scipy.linalg.expm(-float(time) * L) @ (a0 - steady)


def test_hard_thermal_forcing_drives_dynamics_without_polluting_joule_output():
    model, M, K, forcing = _model()
    a0 = np.array([0.2, -0.1])
    u = np.array([0.4])
    result = model.predict(
        0.8,
        initial_state=a0,
        geometry=np.empty(0),
        operating=u,
        max_step=0.8,
        method="etd2",
    )
    expected = _exact_linear_state(M, K, forcing, a0, 0.8)
    np.testing.assert_allclose(result.state, expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(result.heat_source, np.zeros(2), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        result.derivative,
        np.linalg.solve(M, -K @ result.state + forcing),
        rtol=3e-13,
        atol=3e-13,
    )


def test_hard_thermal_forcing_is_used_by_steady_batch_and_persistence(tmp_path):
    model, M, K, forcing = _model()
    steady = model.steady_state(
        initial_guess=np.zeros(2),
        geometry=np.empty(0),
        operating=np.array([0.0]),
        tolerance=1e-12,
    )
    assert steady.converged
    np.testing.assert_allclose(steady.state, np.linalg.solve(K, forcing), rtol=2e-12, atol=2e-12)

    initial = np.array([[0.0, 0.0], [0.2, -0.1]])
    operating = np.array([[0.0], [0.6]])
    batch = model.predict_batch_fixed_etd2(
        0.5,
        initial_states=initial,
        geometry=np.empty(0),
        operating=operating,
        max_step=0.5,
    )
    expected = np.vstack([_exact_linear_state(M, K, forcing, row, 0.5) for row in initial])
    np.testing.assert_allclose(batch.states, expected, rtol=3e-12, atol=3e-12)
    np.testing.assert_allclose(batch.heat_sources, np.zeros_like(batch.heat_sources), rtol=0.0, atol=0.0)

    path = model.save(tmp_path / "forcing.npz")
    restored = StructurePreservingNeuralElectroThermalROM.load(
        path,
        expected_physical_signature="forcing-unit",
    )
    np.testing.assert_allclose(restored.field.thermal_rhs_forcing, forcing, rtol=0.0, atol=0.0)
    restored_result = restored.predict(
        0.5,
        initial_state=initial[1],
        geometry=np.empty(0),
        operating=operating[1],
        max_step=0.5,
    )
    np.testing.assert_allclose(restored_result.state, batch.states[1], rtol=3e-12, atol=3e-12)
