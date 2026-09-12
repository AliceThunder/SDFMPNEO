import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM
from sdfmpneo.electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.electrothermal_tensor.surrogate import NeuralTensorSurrogate
from sdfmpneo.electrothermal_tensor.vector_field import FixedThermalOperatorFamily


def _model():
    torch = pytest.importorskip("torch")
    # Two thermal modes, one current. The zero MLP leaves beta=0, so the mean
    # tensor defines a nontrivial current-dependent but state-independent source.
    pod = TensorPOD(
        mean=np.array([
            0.8, 0.1, 0.5,
            0.2, -0.05, 0.3,
        ]),
        basis=np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]),
        singular_values=np.array([1.0]),
        thermal_rank=2,
        current_dimension=1,
    )
    config = ResidualMLPConfig(input_dimension=2, output_dimension=1, width=8, blocks=1)
    network = build_residual_mlp(config, FeatureNormalizer(np.zeros(2), np.ones(2)))
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
    M = np.array([[1.8, 0.15], [0.15, 1.2]])
    K = np.array([[2.1, 0.12], [0.12, 1.4]])
    return StructurePreservingNeuralElectroThermalROM(
        surrogate,
        FixedThermalOperatorFamily(M, K),
        training_domain={
            "state_lower": np.array([-2.0, -2.0]),
            "state_upper": np.array([2.0, 2.0]),
            "geometry_lower": np.empty(0),
            "geometry_upper": np.empty(0),
            "operating_lower": np.array([-2.0]),
            "operating_upper": np.array([2.0]),
        },
    )


def test_batched_fixed_etd2_matches_independent_predictions():
    model = _model()
    initial = np.array([[0.1, -0.2], [0.4, 0.05], [-0.3, 0.2]])
    operating = np.array([[0.1], [0.8], [-0.6]])
    batch = model.predict_batch_fixed_etd2(
        0.7,
        initial_states=initial,
        geometry=np.empty(0),
        operating=operating,
        max_step=0.1,
    )
    independent = [
        model.predict(
            0.7,
            initial_state=a0,
            geometry=np.empty(0),
            operating=u,
            max_step=0.1,
            method="etd2",
        )
        for a0, u in zip(initial, operating)
    ]
    np.testing.assert_allclose(
        batch.states,
        np.stack([item.state for item in independent]),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        batch.heat_sources,
        np.stack([item.heat_source for item in independent]),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        batch.derivatives,
        np.stack([item.derivative for item in independent]),
        rtol=2e-12,
        atol=2e-12,
    )
    assert batch.steps == independent[0].steps


def test_single_and_batch_etd_share_same_cached_generalized_spectrum():
    model = _model()
    geometry = np.empty(0)
    first = model._spectrum_for_geometry(geometry)
    second = model._spectrum_for_geometry(geometry)
    assert first is second
    assert len(model._spectrum_cache) == 1

    model.predict(
        0.2,
        initial_state=np.array([0.1, -0.1]),
        geometry=geometry,
        operating=np.array([0.2]),
        max_step=0.1,
        method="etd2_adaptive",
    )
    model.predict_batch_fixed_etd2(
        0.2,
        initial_states=np.array([[0.1, -0.1], [0.0, 0.1]]),
        geometry=geometry,
        operating=np.array([[0.2], [0.4]]),
        max_step=0.1,
    )
    assert model._spectrum_for_geometry(geometry) is first
    assert len(model._spectrum_cache) == 1


def test_batched_inference_rejects_domain_violation():
    model = _model()
    with pytest.raises(ValueError, match="outside the trained domain"):
        model.predict_batch_fixed_etd2(
            0.1,
            initial_states=np.array([[0.0, 0.0], [0.0, 0.0]]),
            geometry=np.empty(0),
            operating=np.array([[0.0], [3.0]]),
            max_step=0.05,
        )
