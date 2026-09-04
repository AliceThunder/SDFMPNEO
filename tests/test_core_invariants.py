import math

import torch

from sdfmpneo.config import RankLevel, SolverConfig
from sdfmpneo.contracts import (
    BasisQueryFeatures,
    EMOperators,
    GeometryEncoding,
    TopologyOperators,
)
from sdfmpneo.networks.reference import ReferenceCoordinateBasisGenerator
from sdfmpneo.physics.em import (
    minimum_dissipation_eigenvalue,
    reciprocity_defect,
    sea_loss_consistency,
    solve_em_schur,
)
from sdfmpneo.physics.flow import magnetic_reynolds_number, positive_sst_state
from sdfmpneo.physics.thermal import conservative_mortar_matrix
from sdfmpneo.reduction import assemble_solenoidal_basis, metric_orthonormalize
from sdfmpneo.solvers.equilibrium import solve_pseudo_transient_newton
from sdfmpneo.state import ReducedStateLayout
from sdfmpneo.training import implicit_parameter_gradients


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


def test_slow_operating_features_cannot_leak_into_em_basis() -> None:
    torch.manual_seed(11)
    topology = TopologyOperators(
        curl_current=torch.randn(3, 2),
        harmonic_current=torch.randn(3, 1),
        curl_velocity=torch.randn(4, 2),
        harmonic_velocity=torch.randn(4, 1),
    )
    queries = BasisQueryFeatures(
        current_potential=torch.randn(1, 3, 2),
        thermal=torch.randn(1, 5, 3),
        velocity_potential=torch.randn(1, 3, 2),
        log_tke=torch.randn(1, 4, 1),
        log_omega=torch.randn(1, 4, 1),
    )
    common = dict(
        coil_tokens=torch.randn(1, 6, 4),
        package_tokens=torch.randn(1, 7, 3),
        global_features=torch.randn(1, 5),
        basis_queries=queries,
    )
    geometry_a = GeometryEncoding(slow_features=torch.zeros(1, 2), **common)
    geometry_b = GeometryEncoding(slow_features=torch.ones(1, 2), **common)

    generator = ReferenceCoordinateBasisGenerator(
        coil_token_dim=4,
        package_token_dim=3,
        global_dim=5,
        slow_dim=2,
        query_dims=(2, 3, 2, 1, 1),
        hidden_dim=64,
    )
    raw_a = generator(geometry_a, topology)
    raw_b = generator(geometry_b, topology)

    assert raw_a.current_potential.shape[-1] == 32
    assert torch.allclose(raw_a.current_potential, raw_b.current_potential)
    assert not torch.allclose(raw_a.thermal, raw_b.thermal)


def test_mixed_dimensional_mortar_is_conservative_and_dissipative() -> None:
    interpolation = torch.eye(2, dtype=torch.float64)
    conductance = torch.tensor([2.0, 3.0], dtype=torch.float64)
    coupling = conservative_mortar_matrix(interpolation, conductance)

    assert torch.allclose(coupling, coupling.T)
    assert torch.linalg.eigvalsh(coupling).min().item() > -1.0e-12
    uniform_temperature = torch.ones(4, dtype=torch.float64)
    assert torch.allclose(
        coupling @ uniform_temperature, torch.zeros(4, dtype=torch.float64)
    )


def test_sst_log_state_is_positive_and_rm_is_dimensionless_indicator() -> None:
    log_state = torch.tensor([-10.0, 0.0, 2.0], dtype=torch.float64)
    state = positive_sst_state(log_state, reference_value=1.0e-3)
    assert torch.all(state > 0)

    rm = magnetic_reynolds_number(
        permeability=4.0e-7 * math.pi,
        conductivity=5.0,
        velocity_scale=2.0,
        length_scale=0.2,
    )
    assert 0.0 < rm.item() < 1.0e-4


def test_reduced_state_layout_matches_rank_schedule() -> None:
    rank = RankLevel(em=8, thermal=5, velocity=7, tke=3, omega=2)
    layout = ReducedStateLayout.from_rank(rank)
    z = torch.arange(layout.size, dtype=torch.float64)
    state = layout.split(z)
    assert state.thermal.numel() == 5
    assert state.velocity.numel() == 7
    assert state.log_tke.numel() == 3
    assert state.log_omega.numel() == 2
    assert torch.equal(layout.pack(state), z)


def test_implicit_equilibrium_gradient_matches_analytic_derivative() -> None:
    # F(z, theta)=z-theta^2=0 -> z*(theta)=theta^2.
    # L=0.5*(z+theta)^2 -> dL/dtheta=(theta^2+theta)*(2*theta+1).
    theta = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    z = (theta.detach() ** 2).reshape(1).requires_grad_(True)

    def residual(state: torch.Tensor) -> torch.Tensor:
        return state - theta.square().reshape(1)

    loss = 0.5 * (z[0] + theta).square()
    gradients, adjoint = implicit_parameter_gradients(loss, residual, z, [theta])
    expected = (theta.detach().square() + theta.detach()) * (2.0 * theta.detach() + 1.0)

    assert torch.allclose(gradients[0], expected, atol=1.0e-10, rtol=1.0e-10)
    assert adjoint.numel() == 1
