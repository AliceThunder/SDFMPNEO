import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM
from sdfmpneo.electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from sdfmpneo.electrothermal_tensor.physical_layer import decode_heat_source_numpy
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.electrothermal_tensor.surrogate import NeuralTensorSurrogate
from sdfmpneo.electrothermal_tensor.vector_field import FixedThermalOperatorFamily


def _model_with_truncated_energy():
    torch = pytest.importorskip("torch")
    # One thermal mode, one operating variable => svec width 3.
    pod = TensorPOD(
        mean=np.array([0.7, 0.2, 0.4]),
        basis=np.array([[1.0], [0.0], [0.0]]),
        singular_values=np.array([2.0]),
        thermal_rank=1,
        current_dimension=1,
        total_centered_energy=10.0,
    )
    config = ResidualMLPConfig(input_dimension=1, output_dimension=1, width=8, blocks=1)
    network = build_residual_mlp(config, FeatureNormalizer(np.zeros(1), np.ones(1)))
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    surrogate = NeuralTensorSurrogate(
        network,
        pod,
        state_dimension=1,
        geometry_dimension=0,
        coefficient_mean=np.array([0.3]),
        coefficient_scale=np.array([0.5]),
    )
    return StructurePreservingNeuralElectroThermalROM(
        surrogate,
        FixedThermalOperatorFamily(np.array([[1.4]]), np.array([[2.1]])),
        physical_signature="pod-energy-test",
        training_domain={
            "state_lower": np.array([-1.0]),
            "state_upper": np.array([1.0]),
            "geometry_lower": np.empty(0),
            "geometry_upper": np.empty(0),
            "operating_lower": np.array([-2.0]),
            "operating_upper": np.array([2.0]),
        },
    )


def test_truncated_pod_total_energy_survives_model_roundtrip(tmp_path):
    model = _model_with_truncated_energy()
    assert model.surrogate.pod.energy_fraction() == pytest.approx(0.4)
    path = model.save(tmp_path / "model.npz")
    loaded = StructurePreservingNeuralElectroThermalROM.load(
        path,
        expected_physical_signature="pod-energy-test",
        device="cpu",
    )
    assert loaded.surrogate.pod.total_centered_energy == pytest.approx(10.0)
    assert loaded.surrogate.pod.energy_fraction() == pytest.approx(0.4)


def test_operating_specific_contraction_is_exact_and_cached():
    model = _model_with_truncated_energy()
    surrogate = model.surrogate
    state = np.array([0.25])
    geometry = np.empty(0)
    operating = np.array([0.8])
    beta = surrogate.predict_coefficients_numpy(state, geometry)

    prepared = surrogate.prepare_operating_numpy(operating)
    repeated = surrogate.prepare_operating_numpy(operating.copy())
    assert repeated is prepared

    expected = decode_heat_source_numpy(beta, surrogate.pod, operating)
    np.testing.assert_allclose(prepared.evaluate(beta), expected, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        surrogate.heat_source_numpy(state, geometry, operating),
        expected,
        rtol=0.0,
        atol=2e-15,
    )
