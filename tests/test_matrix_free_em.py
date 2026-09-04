import math

import torch

from sdfmpneo.contracts import EMOperators
from sdfmpneo.physics.em import (
    DenseEMActionAdapter,
    project_matrix_free_em_system,
    sea_loss_consistency,
    solve_em_matrix_free,
    solve_em_schur,
)
from sdfmpneo.reduction import metric_orthonormalize


def _spd(n: int, shift: float = 1.0) -> torch.Tensor:
    a = torch.randn(n, n, dtype=torch.float64)
    return a.T @ a + shift * torch.eye(n, dtype=torch.float64)


class RecordingDenseActions(DenseEMActionAdapter):
    def __init__(self, operators: EMOperators) -> None:
        super().__init__(operators)
        self.system_shapes: list[tuple[int, ...]] = []
        self.dissipation_shapes: list[tuple[int, ...]] = []
        self.feedback_shapes: list[tuple[int, ...]] = []

    def apply_system(self, vectors: torch.Tensor) -> torch.Tensor:
        self.system_shapes.append(tuple(vectors.shape))
        return super().apply_system(vectors)

    def apply_dissipation(self, vectors: torch.Tensor) -> torch.Tensor:
        self.dissipation_shapes.append(tuple(vectors.shape))
        return super().apply_dissipation(vectors)

    def port_feedback(self, vectors: torch.Tensor) -> torch.Tensor:
        self.feedback_shapes.append(tuple(vectors.shape))
        return super().port_feedback(vectors)


def test_matrix_free_projection_matches_dense_schur_without_full_trial_matrix() -> None:
    torch.manual_seed(31)
    batch, n, rank, ports = 1, 11, 4, 2
    basis = torch.randn(batch, n, rank, dtype=torch.float64)
    basis = metric_orthonormalize(
        basis,
        torch.eye(n, dtype=torch.float64),
        jitter=1.0e-12,
    )

    resistance = _spd(n, shift=0.3)[None, ...]
    inductance = _spd(n, shift=0.2)[None, ...] * 1.0e-6
    coupling = torch.randn(batch, n, ports, dtype=torch.float64) * 8.0e-4
    omega = torch.tensor([2.0 * math.pi * 100.0e3], dtype=torch.float64)
    z0 = torch.tensor(
        [[[0.12 + 12.0j, 0.01 + 1.8j], [0.01 + 1.8j, 0.16 + 13.0j]]],
        dtype=torch.complex128,
    )
    operators = EMOperators(
        resistance=resistance,
        inductance=inductance,
        coupling=coupling,
        z_background=z0,
        omega=omega,
    )

    dense = solve_em_schur(basis, operators)
    actions = RecordingDenseActions(operators)
    projected = project_matrix_free_em_system(basis, actions)
    matrix_free = solve_em_matrix_free(basis, actions)

    assert projected.system_matrix.shape == (batch, rank, rank)
    assert projected.rhs_matrix.shape == (batch, rank, ports)
    assert projected.feedback_matrix.shape == (batch, ports, rank)
    assert projected.dissipation_matrix.shape == (batch, rank, rank)

    assert actions.system_shapes == [(batch, n, rank), (batch, n, rank)]
    assert actions.dissipation_shapes == [(batch, n, rank), (batch, n, rank)]
    assert actions.feedback_shapes == [(batch, n, rank), (batch, n, rank)]
    assert all(shape[-1] == rank for shape in actions.system_shapes)
    assert all(shape[-1] != n for shape in actions.system_shapes)

    assert torch.allclose(
        matrix_free.reduced_coefficients,
        dense.reduced_coefficients,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert torch.allclose(
        matrix_free.impedance_total,
        dense.impedance_total,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert torch.allclose(
        matrix_free.seawater_loss_matrix,
        dense.seawater_loss_matrix,
        atol=1.0e-11,
        rtol=1.0e-11,
    )
    assert sea_loss_consistency(matrix_free).max().item() < 1.0e-10
