import pytest
import torch

from sdfmpneo.networks.reference import CoordinateModeDecoder
from sdfmpneo.reduction import BasisRankError, metric_orthonormalize


def test_coordinate_decoder_rejects_hidden_rank_bottleneck() -> None:
    with pytest.raises(ValueError, match="structural rank bottleneck"):
        CoordinateModeDecoder(
            query_dim=3,
            context_dim=8,
            hidden_dim=8,
            rank=12,
        )


def test_metric_whitening_rejects_exact_basis_collapse() -> None:
    column = torch.randn(1, 10, 1, dtype=torch.float64)
    collapsed = torch.cat((column, column, torch.randn(1, 10, 1, dtype=torch.float64)), dim=-1)
    metric = torch.eye(10, dtype=torch.float64)
    with pytest.raises(BasisRankError, match="rank deficient|collapsed"):
        metric_orthonormalize(
            collapsed,
            metric,
            jitter=1.0e-6,
            rank_rtol=1.0e-12,
        )


def test_metric_whitening_does_not_create_rank_when_dofs_are_insufficient() -> None:
    impossible = torch.randn(1, 3, 4, dtype=torch.float64)
    metric = torch.eye(3, dtype=torch.float64)
    with pytest.raises(BasisRankError, match="requests rank"):
        metric_orthonormalize(impossible, metric)
