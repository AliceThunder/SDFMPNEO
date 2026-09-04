import pytest
import torch

from sdfmpneo.config import RankLevel
from sdfmpneo.contracts import PhysicsSeedBundle
from sdfmpneo.reduction import seeded_metric_orthonormalize
from sdfmpneo.seeds import validate_seed_prefix_ranks


def test_physics_seed_is_preserved_as_metric_prefix() -> None:
    torch.manual_seed(23)
    batch, n, seed_rank, total_rank = 1, 12, 2, 5
    seed = torch.randn(batch, n, seed_rank, dtype=torch.float64)
    neural = torch.randn(batch, n, total_rank, dtype=torch.float64)
    a = torch.randn(n, n, dtype=torch.float64)
    metric = a.T @ a + torch.eye(n, dtype=torch.float64)

    combined = seeded_metric_orthonormalize(
        seed,
        neural,
        metric,
        total_rank=total_rank,
        jitter=1.0e-12,
        rank_rtol=1.0e-12,
    )
    gram = torch.einsum("bnr,nm,bms->brs", combined, metric, combined)
    assert torch.allclose(
        gram,
        torch.eye(total_rank, dtype=torch.float64)[None, ...],
        atol=1.0e-9,
        rtol=1.0e-9,
    )

    # The prefix need not equal the raw seed columns numerically after whitening,
    # but it must span exactly the same seed subspace.
    seed_projector = seed @ torch.linalg.pinv(seed)
    prefix = combined[..., :seed_rank]
    residual = prefix - seed_projector @ prefix
    assert torch.linalg.vector_norm(residual).item() < 1.0e-9


def test_seed_rank_cannot_exceed_first_online_rank() -> None:
    def z(n: int, r: int) -> torch.Tensor:
        return torch.zeros(1, n, r)

    seeds = PhysicsSeedBundle(
        current=z(10, 9),
        thermal=z(10, 2),
        velocity=z(10, 2),
        log_tke=z(10, 1),
        log_omega=z(10, 1),
    )
    first = RankLevel(em=8, thermal=8, velocity=12, tke=4, omega=4)
    with pytest.raises(ValueError, match="physical backbone must never be truncated"):
        validate_seed_prefix_ranks(seeds, first)
