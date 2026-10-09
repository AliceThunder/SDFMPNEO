from types import SimpleNamespace

import numpy as np
import torch

from sdfmpneo_vnext.balanced_objectives import (
    PORT_TRAINING_CONTRACT,
    SPATIAL_TRAINING_CONTRACT,
    balanced_port_batch_loss,
    balanced_spatial_relative_loss,
)
from sdfmpneo_vnext.spatial_boundary_features import (
    boundary_aware_conductor_coordinates,
)


def test_balanced_spatial_loss_is_scale_invariant_and_finite_near_zero_truth():
    weights = np.ones(4, dtype=float)
    target = torch.full((4, 2, 2), 1.0e-12, dtype=torch.complex64)
    predicted = torch.full((4, 2, 2), 1.0e-3, dtype=torch.complex64)

    value = balanced_spatial_relative_loss(predicted, target, weights)
    scaled = balanced_spatial_relative_loss(
        predicted * 1.0e-6,
        target * 1.0e-6,
        weights,
    )

    assert torch.isfinite(value)
    assert 0.0 <= float(value) <= 1.0001
    assert torch.allclose(value, scaled, rtol=2e-5, atol=1e-7)


def _boundary_scene():
    geometry = SimpleNamespace(
        conductor_width=2.0,
        conductor_thickness=4.0,
        cross_section_exponent=4.0,
    )
    return SimpleNamespace(coils=[SimpleNamespace(geometry=geometry)])


def test_boundary_aware_conductor_coordinates_expose_superellipse_radius():
    coordinates = boundary_aware_conductor_coordinates(
        _boundary_scene(),
        np.asarray([0, 0, 0]),
        np.asarray([0.0, 0.25, 0.5]),
        np.asarray([[1.0, 0.0], [0.0, 2.0], [0.5, 0.0]]),
    )

    assert coordinates.shape == (3, 6)
    assert np.allclose(coordinates[:2, 2], 1.0)
    assert np.isclose(coordinates[2, 2], 0.5)
    assert np.allclose(coordinates[:, 3], coordinates[:, 2] ** 2)


def test_tensor_boundary_schema_does_not_replace_legacy_coordinates():
    import sdfmpneo_vnext.hybrid_spatial_neural as legacy
    import sdfmpneo_vnext.tensor_spatial_neural as tensor_spatial

    coordinates = legacy._coordinate_features(
        _boundary_scene(),
        np.asarray([0]),
        np.asarray([0.25]),
        np.asarray([[0.5, 0.0]]),
    )

    assert coordinates.shape == (1, 4)
    assert tensor_spatial.TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA == 2


def test_controlled_training_installs_balanced_contracts():
    import sdfmpneo_vnext._controlled_training_core as port_core
    import sdfmpneo_vnext.controlled_spatial_consistent as spatial_core
    import sdfmpneo_vnext.spatial_consistent_training as spatial_objective
    import sdfmpneo_vnext.controlled_training as controlled

    assert PORT_TRAINING_CONTRACT == 2
    assert SPATIAL_TRAINING_CONTRACT == 5
    assert port_core._batch_loss is balanced_port_batch_loss
    assert spatial_objective._coordinate_features is boundary_aware_conductor_coordinates
    assert spatial_objective.SPATIAL_TRAINING_CONTRACT == SPATIAL_TRAINING_CONTRACT
    assert spatial_core.SPATIAL_TRAINING_CONTRACT == SPATIAL_TRAINING_CONTRACT

    legacy = controlled._legacy_port_signature({}, "dataset")
    current = port_core._port_signature({}, "dataset")
    assert current != legacy


def test_controlled_port_domain_uses_full_dataset_after_resume(monkeypatch):
    import sdfmpneo_vnext.controlled_training as controlled

    samples = (object(), object())
    artifact = SimpleNamespace(material_domain={"package_epsilon": (2.0, 9.0)})
    expected = {"package_epsilon": (1.5, 10.0)}

    def fake_train(received, **kwargs):
        assert received == samples
        return artifact, [{"epoch": 1}]

    monkeypatch.setattr(controlled._port_core, "train_port_controlled", fake_train)
    monkeypatch.setattr(
        controlled,
        "_sample_tensor_ranges",
        lambda received: expected if tuple(received) == samples else None,
    )

    result, history = controlled.train_port_controlled(
        samples,
        config={},
        cache_key="dataset",
        checkpoint_path="unused.pt",
        control=None,
        resume=True,
    )

    assert result is artifact
    assert result.material_domain == expected
    assert history == [{"epoch": 1}]
