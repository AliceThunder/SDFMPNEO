import math

import torch

from sdfmpneo.config import SolverConfig
from sdfmpneo.contracts import EMOperators
from sdfmpneo.physics.em import (
    minimum_dissipation_eigenvalue,
    reciprocity_defect,
    sea_loss_consistency,
    solve_em_schur,
)
from sdfmpneo.reduction import assemble_solenoidal_basis, metric_orthonormalize
from sdfmpneo.solvers.equilibrium import solve_pseudo_transient_newton


def _spd(n: int, shift: float = 1.0) -> torch.Tensor:
    a = torch.randn(n, n, dtype=torch.float64)
    return a.T @ a + shift * torch.eye(n, dtype=torch.float64)


def test_de_rham_generation_is_exactly_divergence_free() -> None:
    curl = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], dtype=torch.float64
    )
    divergence = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    harmonic = torch.tensor([[1.0], [-1.0], [0.0]], dtype=torch.float64)
    assert torch.allclose(divergence @ curl, torch.zeros(1, 2, dtype=torch.float64))
    assert torch.allclose(
        divergence @ harmonic, torch.zeros(1, 1, dtype=torch.float64)
    )

    raw = torch.randn(1, 3, 2, dtype=torch.float64)
    basis = assemble_solenoidal_basis(curl, harmonic, raw)
    defect = torch.einsum("dn,bnr->bdr", divergence, basis)
    assert torch.max(torch.abs(defect)).item() < 1.0e-12


def test_metric_orthonormalisation_is_hard_constraint() -> None:
    torch.manual_seed(2)
    n, rank = 9, 4
    basis = torch.randn(2, n, rank, dtype=torch.float64)
    metric = _spd(n, shift=2.0)
    orth = metric_orthonormalize(basis, metric, jitter=1.0e-12)
    gram = torch.einsum("bnr,nm,bms->brs", orth, metric, orth)
    eye = torch.eye(rank, dtype=torch.float64).expand_as(gram)
    assert torch.allclose(gram, eye, atol=1.0e-9, rtol=1.0e-9)


def test_em_schur_preserves_reciprocity_passivity_and_joule_loss() -> None:
    torch.manual_seed(7)
    batch, n, rank, ports = 1, 7, 4, 2
    basis = torch.randn(batch, n, rank, dtype=torch.float64)
    basis = metric_orthonormalize(
        basis, torch.eye(n, dtype=torch.float64), jitter=1.0e-12
    )

    resistance = _spd(n, shift=0.5)[None, ...]
    inductance = _spd(n, shift=0.2)[None, ...] * 1.0e-6
    coupling = torch.randn(batch, n, ports, dtype=torch.float64) * 1.0e-3

    r_cu = torch.diag(torch.tensor([0.15, 0.20], dtype=torch.float64))
    l_air = torch.tensor([[25.0, 3.0], [3.0, 28.0]], dtype=torch.float64) * 1.0e-6
    omega = torch.tensor([2.0 * math.pi * 100.0e3], dtype=torch.float64)
    z0 = (r_cu + 1j * omega[0] * l_air)[None, ...]

    solution = solve_em_schur(
        basis,
        EMOperators(
            resistance=resistance,
            inductance=inductance,
            coupling=coupling,
            z_background=z0,
            omega=omega,
        ),
    )

    assert reciprocity_defect(solution.impedance_sea).max().item() < 1.0e-11
    assert reciprocity_defect(solution.impedance_total).max().item() < 1.0e-11
    assert minimum_dissipation_eigenvalue(solution.impedance_sea).min().item() > -1.0e-11
    assert sea_loss_consistency(solution).max().item() < 1.0e-10


def test_pseudo_transient_newton_solves_nonlinear_equilibrium() -> None:
    def residual(z: torch.Tensor) -> torch.Tensor:
        return torch.stack((z[0] ** 2 - 2.0, z[1] - 3.0))

    result = solve_pseudo_transient_newton(
        residual,
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        SolverConfig(max_newton_steps=40, residual_tolerance=1.0e-10),
    )
    target = torch.tensor([math.sqrt(2.0), 3.0], dtype=torch.float64)
    assert result.converged
    assert torch.allclose(result.state, target, atol=1.0e-8, rtol=1.0e-8)
