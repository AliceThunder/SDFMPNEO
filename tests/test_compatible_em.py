import numpy as np

from sdfmpneo.em import CompatibleAphiDiscretization, ResidualGreedyEMReducer


def make_discretization():
    C = np.array([[1.0, -1.0]])
    G = np.array([[1.0], [1.0]])
    R = np.array([[1.0], [-1.0]]) / np.sqrt(2.0)
    Nu = np.array([[2.0]])
    S0 = np.diag([1.0, 0.8])
    S_state = np.array([
        np.diag([0.10, 0.03]),
        np.diag([0.02, 0.07]),
    ])
    source = np.array([1.0, 0.2])
    omega = 3.0
    metric = np.eye(2)
    W0 = np.array([
        np.diag([0.6, 0.1]),
        np.diag([0.2, 0.5]),
    ])
    Wstate = np.zeros((2, 2, 2, 2))
    Wstate[0, 0] = np.diag([0.02, 0.01])
    Wstate[1, 1] = np.diag([0.01, 0.03])
    return CompatibleAphiDiscretization(C, G, R, Nu, S0, S_state, source, omega, metric, W0, Wstate)


def test_compatible_aphi_assembly_blocks_and_topology():
    disc = make_discretization()
    problem = disc.to_parametric_problem()
    assert np.allclose(disc.curl @ disc.grad_c, 0.0)
    assert problem.A0.shape == (2, 2)
    assert problem.A_state.shape == (2, 2, 2)
    assert problem.H_loss.shape == (2, 2, 2)
    assert problem.H_loss_state.shape == (2, 2, 2, 2)

    C, R, S = disc.curl, disc.a_basis, disc.conductivity0
    K_A = R.T @ (C.T @ disc.reluctivity_hodge @ C) @ R
    expected_tl = K_A + 1j * disc.omega * (R.T @ S @ R)
    assert np.allclose(problem.A0[:1, :1], expected_tl)
    assert np.allclose(problem.A0[:1, 1:], R.T @ S @ disc.grad_c)


def test_compatible_problem_reduces_and_loss_jacobian_matches_difference():
    problem = make_discretization().to_parametric_problem()
    states = [np.array([x, y]) for x in (-0.1, 0.0, 0.1) for y in (-0.1, 0.0, 0.1)]
    model = ResidualGreedyEMReducer(problem).build(states, tolerance=1e-12)
    a = np.array([0.04, -0.03])
    _, J = model.heat_source_and_jacobian(a)
    assert model.residual_dual_norm(a) < 1e-10

    eps = 1e-6
    Jfd = np.zeros_like(J)
    for k in range(2):
        d = np.zeros(2)
        d[k] = eps
        Jfd[:, k] = (model.heat_source(a + d) - model.heat_source(a - d)) / (2 * eps)
    assert np.allclose(J, Jfd, rtol=2e-6, atol=2e-8)
