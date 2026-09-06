import numpy as np
import scipy.sparse as sp

from sdfmpneo.em import (
    CompatibleAphiDiscretization,
    ImpressedCurrentPortSet,
    ResidualGreedyEMReducer,
)


class _TinyPortGrid:
    """Minimal incidence object for closed-loop port validation."""

    def __init__(self, grad):
        self.grad = sp.csr_matrix(grad)
        self.n_edges = self.grad.shape[0]
        self.n_nodes = self.grad.shape[1]


def make_two_port_problem():
    # C G = 0 and G^T R = 0. The two columns of R are independent cotree-like
    # magnetic coordinates. This is deliberately small but uses the same
    # reciprocal A-psi blocks as the spatial solver.
    C = np.array([
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
    ])
    G = np.ones((3, 1))
    R = np.array([
        [1.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
    ])
    Nu = np.diag([2.0, 1.5])
    S0 = np.diag([1.0, 0.8, 0.6])
    S_state = np.zeros((1, 3, 3))
    omega = 3.0

    # First source is only the nominal single-RHS source of ParametricEMProblem;
    # the multiport reducer below is built from both port RHS columns.
    J1 = np.array([1.0, -1.0, 0.0])
    J2 = np.array([0.0, 1.0, -1.0])

    W0 = np.array([S0.copy()])
    Wstate = np.zeros((1, 1, 3, 3))
    disc = CompatibleAphiDiscretization(
        curl=C,
        grad_c=G,
        a_basis=R,
        reluctivity_hodge=Nu,
        conductivity0=S0,
        conductivity_state=S_state,
        source_current=J1,
        omega=omega,
        riesz_metric=np.eye(3),
        thermal_loss_hodge0=W0,
        thermal_loss_hodge_state=Wstate,
    )
    problem = disc.to_parametric_problem()

    grid = _TinyPortGrid(G)
    ports = ImpressedCurrentPortSet.build(
        grid,
        a_basis=R,
        n_scalar=G.shape[1],
        omega=omega,
        edge_currents=np.column_stack([J1, J2]),
        names=("tx", "rx"),
    )
    return disc, problem, ports


def _roundoff_bound(scale, dimension):
    return np.finfo(float).eps * max(1, dimension) * max(1.0, float(scale))


def test_multiport_reciprocity_and_passivity_follow_from_field_operator():
    _, problem, ports = make_two_port_problem()
    a = np.zeros(problem.n_thermal)
    result = ports.evaluate(problem, a)

    scale = np.linalg.norm(result.impedance)
    tol = 100.0 * _roundoff_bound(scale, result.impedance.size)
    assert np.linalg.norm(result.impedance - result.impedance.T) <= tol
    assert result.reciprocity_defect <= 100.0 * np.finfo(float).eps * result.impedance.size

    # R = Re(Z) must be positive semidefinite for passive positive
    # conductivity. Only a floating-point backward-error allowance is used.
    rscale = np.linalg.norm(result.resistance, ord=2)
    assert result.minimum_resistance_eigenvalue >= -100.0 * _roundoff_bound(
        rscale, result.resistance.shape[0]
    )


def test_multiport_input_power_equals_joule_loss():
    disc, problem, ports = make_two_port_problem()
    a = np.zeros(problem.n_thermal)
    result = ports.evaluate(problem, a)

    currents = np.array([1.0 + 0.2j, -0.4 + 0.1j])
    unit_states = ports.solve_coordinate_states(problem, a)
    x = unit_states @ currents

    L_E = disc.electric_extraction()
    joule = 0.5 * np.real(np.vdot(x, L_E.conj().T @ disc.conductivity0 @ L_E @ x))
    input_power = result.average_input_power(currents)

    scale = max(abs(joule), abs(input_power), 1.0)
    assert abs(input_power - joule) <= 200.0 * _roundoff_bound(scale, problem.n_em)


def test_joint_multi_rhs_reduction_preserves_full_impedance_matrix():
    _, problem, ports = make_two_port_problem()
    states = [np.array([0.0])]
    reducer = ResidualGreedyEMReducer(problem)
    reduced = reducer.build_multi_rhs(states, ports.coordinate_rhs, tolerance=1e-12)

    a = states[0]
    full = ports.evaluate(problem, a)
    rom = ports.evaluate(problem, a, reduced_basis=reduced.V)

    zscale = max(np.linalg.norm(full.impedance), 1.0)
    assert np.linalg.norm(rom.impedance - full.impedance) <= 500.0 * _roundoff_bound(
        zscale, problem.n_em * ports.n_ports
    )
    assert rom.maximum_residual_dual_norm < 1e-11


def test_mutual_inductance_is_the_off_diagonal_inductance_output():
    _, problem, ports = make_two_port_problem()
    result = ports.evaluate(problem, np.zeros(problem.n_thermal))
    M = result.mutual_inductance
    assert np.allclose(np.diag(M), 0.0)
    assert np.allclose(M[0, 1], result.inductance[0, 1])
    assert np.allclose(M[1, 0], result.inductance[1, 0])
