import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM
from sdfmpneo.electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.electrothermal_tensor.surrogate import NeuralTensorSurrogate
from sdfmpneo.electrothermal_tensor.vector_field import FixedThermalOperatorFamily


def test_neural_rom_npz_roundtrip_without_pickle(tmp_path):
    torch = pytest.importorskip("torch")
    pod = TensorPOD(
        mean=np.array([1.0, 0.25, 2.0]),
        basis=np.array([[1.0], [0.0], [0.0]]),
        singular_values=np.array([1.0]),
        thermal_rank=1,
        current_dimension=1,
    )
    config = ResidualMLPConfig(input_dimension=1, output_dimension=1, width=8, blocks=1)
    network = build_residual_mlp(config, FeatureNormalizer(np.array([0.0]), np.array([1.0])))
    for parameter in network.parameters():
        torch.nn.init.zeros_(parameter)
    surrogate = NeuralTensorSurrogate(
        network,
        pod,
        state_dimension=1,
        geometry_dimension=0,
        coefficient_mean=np.array([0.0]),
        coefficient_scale=np.array([1.0]),
    )
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        FixedThermalOperatorFamily(np.array([[1.0]]), np.array([[2.0]])),
        physical_signature="unit-physics-v1",
        training_domain={
            "state_lower": np.array([-2.0]),
            "state_upper": np.array([2.0]),
            "geometry_lower": np.empty(0),
            "geometry_upper": np.empty(0),
            "operating_lower": np.array([-2.0]),
            "operating_upper": np.array([2.0]),
        },
    )
    path = model.save(tmp_path / "neural_rom.npz")
    loaded = StructurePreservingNeuralElectroThermalROM.load(
        path,
        expected_physical_signature="unit-physics-v1",
    )
    a = np.array([0.4])
    u = np.array([0.7])
    np.testing.assert_allclose(
        loaded.field.heat_source(a, np.empty(0), u),
        model.field.heat_source(a, np.empty(0), u),
        rtol=0.0,
        atol=0.0,
    )
    original = model.predict(
        0.5,
        initial_state=a,
        geometry=np.empty(0),
        operating=u,
        max_step=0.5,
    )
    restored = loaded.predict(
        0.5,
        initial_state=a,
        geometry=np.empty(0),
        operating=u,
        max_step=0.5,
    )
    np.testing.assert_allclose(restored.state, original.state, rtol=2e-13, atol=2e-13)
