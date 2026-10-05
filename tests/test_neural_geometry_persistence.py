import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.geometry_thermal import AffineGeometryThermalOperatorFamily
from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM
from sdfmpneo.electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.electrothermal_tensor.surrogate import NeuralTensorSurrogate


def _geometry_family():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ]
    )
    tetrahedra = np.array(
        [
            [4, 1, 2, 3],
            [0, 4, 2, 3],
            [0, 1, 4, 3],
            [0, 1, 2, 4],
        ],
        dtype=int,
    )
    directions = np.zeros((1, len(vertices), 3))
    directions[0, 4, 0] = 0.02
    return AffineGeometryThermalOperatorFamily(
        reference_vertices=vertices,
        tetrahedra=tetrahedra,
        vertex_directions=directions,
        parameter_names=("interior_shift",),
        geometry_reference=np.array([1.0]),
        geometry_lower=np.array([0.9]),
        geometry_upper=np.array([1.1]),
        volumetric_heat_capacity=np.full(4, 2.0),
        thermal_conductivity=np.full(4, 3.0),
        thermal_basis=np.ones((1, 1)),
        free_nodes=np.array([4]),
        cache_size=4,
    )


def test_geometry_neural_rom_roundtrip_embeds_exact_thermal_chart(tmp_path):
    torch = pytest.importorskip("torch")
    pod = TensorPOD(
        mean=np.array([0.5, 0.0, 1.0]),
        basis=np.array([[1.0], [0.0], [0.0]]),
        singular_values=np.array([1.0]),
        thermal_rank=1,
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
        state_dimension=1,
        geometry_dimension=1,
        coefficient_mean=np.array([0.0]),
        coefficient_scale=np.array([1.0]),
    )
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        _geometry_family(),
        physical_signature="geometry-unit-v1",
        training_domain={
            "state_lower": np.array([-2.0]),
            "state_upper": np.array([2.0]),
            "geometry_lower": np.array([-1.0]),
            "geometry_upper": np.array([1.0]),
            "operating_lower": np.array([-2.0]),
            "operating_upper": np.array([2.0]),
        },
    )
    z = np.array([0.4])
    original_operator = model.thermal_operators.operator(z)
    path = model.save(tmp_path / "geometry_neural.npz")
    loaded = StructurePreservingNeuralElectroThermalROM.load(
        path,
        expected_physical_signature="geometry-unit-v1",
    )
    restored_operator = loaded.thermal_operators.operator(z)
    np.testing.assert_allclose(restored_operator.mass, original_operator.mass, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(restored_operator.stiffness, original_operator.stiffness, rtol=0.0, atol=0.0)
    kwargs = dict(
        time=0.2,
        initial_state=np.array([0.1]),
        geometry=z,
        operating=np.array([0.3]),
        max_step=0.05,
    )
    expected = model.predict(**kwargs)
    actual = loaded.predict(**kwargs)
    np.testing.assert_allclose(actual.state, expected.state, rtol=2e-13, atol=2e-13)


def test_prediction_fails_when_intermediate_state_leaves_training_box():
    torch = pytest.importorskip("torch")
    pod = TensorPOD(
        mean=np.array([10.0, 0.0, 0.0]),
        basis=np.array([[1.0], [0.0], [0.0]]),
        singular_values=np.array([1.0]),
        thermal_rank=1,
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
        state_dimension=1,
        geometry_dimension=1,
        coefficient_mean=np.array([0.0]),
        coefficient_scale=np.array([1.0]),
    )
    model = StructurePreservingNeuralElectroThermalROM(
        surrogate,
        _geometry_family(),
        training_domain={
            "state_lower": np.array([-0.2]),
            "state_upper": np.array([0.2]),
            "geometry_lower": np.array([-1.0]),
            "geometry_upper": np.array([1.0]),
            "operating_lower": np.array([-1.0]),
            "operating_upper": np.array([1.0]),
        },
    )
    with pytest.raises(ValueError, match="left the trained thermal-state domain"):
        model.predict(
            1.0,
            initial_state=np.array([0.0]),
            geometry=np.array([0.0]),
            operating=np.array([0.0]),
            max_step=0.2,
        )
