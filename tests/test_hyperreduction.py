import torch

from sdfmpneo.hyperreduction import (
    build_positive_basis_operator_cubature,
    independent_certifier_indices,
)


def test_positive_operator_cubature_reconstructs_moments_without_snapshots() -> None:
    moments = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    rule = build_positive_basis_operator_cubature(
        moments,
        relative_tolerance=1.0e-10,
        max_points=3,
    )
    target = moments.sum(dim=0)
    reconstructed = rule.apply(moments)

    assert torch.all(rule.weights >= 0)
    assert rule.relative_error < 1.0e-10
    assert torch.allclose(reconstructed, target, atol=1.0e-9, rtol=1.0e-9)


def test_certifier_is_disjoint_from_online_cubature() -> None:
    moments = torch.eye(6, dtype=torch.float64)
    rule = build_positive_basis_operator_cubature(
        moments,
        relative_tolerance=0.7,
        max_points=2,
    )
    certifier = independent_certifier_indices(
        element_count=6,
        cubature_indices=rule.indices,
        count=2,
        seed=17,
    )
    overlap = set(rule.indices.tolist()).intersection(certifier.tolist())
    assert not overlap
