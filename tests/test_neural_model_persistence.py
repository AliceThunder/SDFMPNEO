import json

import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.certification import verify_model_persistence_roundtrip
from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM
from sdfmpneo.electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.electrothermal_tensor.surrogate import NeuralTensorSurrogate
from sdfmpneo.electrothermal_tensor.vector_field import FixedThermalOperatorFamily


def _model():
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
    return StructurePreservingNeuralElectroThermalROM(
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


def test_neural_rom_npz_roundtrip_without_pickle(tmp_path):
    model = _model()
    path = model.save(
        tmp_path / "neural_rom.npz",
        metadata={
            "dataset_hash": "dataset-v1",
            "training_report": {"best_epoch": 12},
        },
    )
    loaded = StructurePreservingNeuralElectroThermalROM.load(
        path,
        expected_physical_signature="unit-physics-v1",
    )
    assert loaded.artifact_metadata["dataset_hash"] == "dataset-v1"
    assert loaded.artifact_metadata["training_report"]["best_epoch"] == 12

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


def test_formal_roundtrip_verifier_checks_metadata_domain_signature_and_numerics():
    model = _model()
    model.artifact_metadata = {
        "dataset_hash": "dataset-v1",
        "training_report": {"best_epoch": 12},
    }
    report = verify_model_persistence_roundtrip(model, device="cpu")
    assert report.passed is True
    assert report.metadata_preserved is True
    assert report.training_domain_preserved is True
    assert report.physical_signature_preserved is True
    assert report.maximum_heat_source_error == pytest.approx(0.0, abs=1e-15)
    assert report.maximum_vector_field_error == pytest.approx(0.0, abs=1e-15)


def test_loaded_artifact_metadata_is_recursively_extended_not_replaced(tmp_path):
    model = _model()
    first = model.save(
        tmp_path / "first.npz",
        metadata={
            "dataset_hash": "dataset-v1",
            "training": {"seed": 7, "device": "cpu"},
        },
    )
    loaded = StructurePreservingNeuralElectroThermalROM.load(first)
    second = loaded.save(
        tmp_path / "second.npz",
        metadata={
            "training": {"audited": True},
            "certification": {"production_ready": False, "gate_report_hash": "abc"},
        },
    )
    restored = StructurePreservingNeuralElectroThermalROM.load(second)
    assert restored.artifact_metadata["dataset_hash"] == "dataset-v1"
    assert restored.artifact_metadata["training"] == {
        "seed": 7,
        "device": "cpu",
        "audited": True,
    }
    assert restored.artifact_metadata["certification"]["gate_report_hash"] == "abc"

    with np.load(second, allow_pickle=False) as data:
        payload = json.loads(str(data["metadata_json"]))
    assert payload["format_version"] == 3
    assert "source_revision" in payload
    assert "software_environment" in payload
