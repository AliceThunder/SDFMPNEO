from types import SimpleNamespace

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from sdfmpneo_vnext.hybrid_neural import (
    HybridNormalizer,
    HybridPhysicsFactoredResidualNet,
)


def _empty_package_sample():
    encoded = SimpleNamespace(
        coil=SimpleNamespace(
            node_features=np.zeros((2, 3), dtype=float),
            pair_features=np.zeros((2, 2, 2), dtype=float),
        ),
        package_features=np.empty((0, 4), dtype=float),
        coil_package_features=np.empty((2, 0, 5), dtype=float),
        package_pair_features=np.empty((0, 0, 6), dtype=float),
    )
    return SimpleNamespace(
        encoded=encoded,
        target_impedance=np.eye(2, dtype=complex),
        baseline_resistance=np.zeros((2, 2), dtype=float),
        baseline_reactance=np.zeros((2, 2), dtype=float),
    )


def test_hybrid_normalizer_uses_neutral_stats_for_absent_packages():
    normalizer = HybridNormalizer.fit((_empty_package_sample(),))

    assert np.all(np.isfinite(normalizer.package_mean))
    assert np.all(np.isfinite(normalizer.package_scale))
    assert np.allclose(normalizer.package_mean, 0.0)
    assert np.allclose(normalizer.package_scale, 1.0)
    assert np.allclose(normalizer.coil_package_mean, 0.0)
    assert np.allclose(normalizer.coil_package_scale, 1.0)
    assert np.allclose(normalizer.package_pair_mean, 0.0)
    assert np.allclose(normalizer.package_pair_scale, 1.0)


def test_hybrid_latent_accepts_empty_package_graph_without_virtual_node():
    model = HybridPhysicsFactoredResidualNet(
        coil_dim=3,
        coil_pair_dim=2,
        package_dim=4,
        cross_dim=5,
        package_pair_dim=6,
        hidden_dim=8,
        factor_rank=2,
        depth=1,
    )
    coil, package = model._latent(
        torch.zeros((2, 3)),
        torch.zeros((2, 2, 2)),
        torch.empty((0, 4)),
        torch.empty((2, 0, 5)),
        torch.empty((0, 0, 6)),
    )

    assert coil.shape == (2, 8)
    assert package.shape == (0, 8)
    assert torch.all(torch.isfinite(coil))
