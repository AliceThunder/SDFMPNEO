from types import SimpleNamespace

import numpy as np
import torch

from sdfmpneo_vnext.balanced_objectives import (
    PORT_TRAINING_CONTRACT,
    SPATIAL_TRAINING_CONTRACT,
    balanced_port_batch_loss,
    balanced_spatial_relative_loss,
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


def test_controlled_training_installs_balanced_contracts():
    import sdfmpneo_vnext._controlled_training_core as port_core
    import sdfmpneo_vnext.controlled_spatial_consistent as spatial_core
    import sdfmpneo_vnext.spatial_consistent_training as spatial_objective
    import sdfmpneo_vnext.controlled_training as controlled

    assert PORT_TRAINING_CONTRACT == 2
    assert SPATIAL_TRAINING_CONTRACT == 3
    assert port_core._batch_loss is balanced_port_batch_loss
    assert spatial_objective._weighted_relative_loss is balanced_spatial_relative_loss
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
