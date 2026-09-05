"""Run the first SDF-MPNEO executable kernel."""

import numpy as np

from sdfmpneo.analytic import AnalyticEvolutionGraph
from sdfmpneo.certification import contraction_certificate, state_error_certificate
from sdfmpneo.em import ParametricEMProblem, ResidualGreedyEMReducer
from sdfmpneo.thermal import ThermalSpectralModel
from sdfmpneo.training import ElectroThermalResidual


def make_problem():
    M = np.diag([2.0, 1.5])
    K = np.array([[5.0, -1.0], [-1.0, 3.0]])
    thermal = ThermalSpectralModel.build(M, K)

    A0 = np.array([[4.0+2.0j, -0.4j, 0.2], [0.3j, 3.2+1.5j, -0.2j], [0.1, 0.2j, 2.8+1.1j]], dtype=complex)
    A_state = np.array([
        np.diag([0.08+0.02j, 0.03+0.01j, 0.02+0.01j]),
        np.diag([0.02+0.01j, 0.06+0.02j, 0.03+0.01j]),
    ])
    b = np.array([1.0, 0.5, 0.2], dtype=complex)
    H_metric = np.eye(3)
    H_loss = np.array([
        np.diag([0.05, 0.02, 0.01]),
        np.diag([0.01, 0.03, 0.04]),
    ], dtype=complex)
    problem = ParametricEMProblem(A0, A_state, b, H_metric, H_loss)

    grid = np.linspace(-0.25, 0.25, 5)
    states = [np.array([x, y]) for x in grid for y in grid]
    em = ResidualGreedyEMReducer(problem).build(states, tolerance=1e-10)
    return thermal, em


def main():
    thermal, em = make_problem()
    graph = AnalyticEvolutionGraph(thermal.lambdas, np.array([0.2, -0.1]))
    graph.add_product_response("coupling_01_to_0", 0, ("base_0", "base_1"), 0.15)

    t = 0.8
    a, da = graph.evaluate(t)
    residual = ElectroThermalResidual(thermal.lambdas, em).evaluate(a, da)
    q, J = em.heat_source_and_jacobian(a)
    contraction = contraction_certificate(thermal.lambdas, J)

    print("thermal lambdas:", thermal.lambdas)
    print("EM reduced rank:", em.V.shape[1])
    print("a(t):", a)
    print("da/dt:", da)
    print("g_em(a):", q)
    print("physical residual norm:", residual.norm)
    print("kappa:", contraction.kappa)
    print("long-time certified:", contraction.long_time_certified)

    if contraction.long_time_certified:
        cert = state_error_certificate(residual.norm, 0.0, em.residual_dual_norm(a), contraction.kappa)
        print("demonstration state-error bound:", cert.state_error_bound)


if __name__ == "__main__":
    main()
