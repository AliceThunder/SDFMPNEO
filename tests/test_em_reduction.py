import numpy as np

from sdfmpneo.em import ParametricEMProblem, ResidualGreedyEMReducer
from sdfmpneo.em.reduced import ReducedEMModel


def test_snapshot_free_residual_greedy_reduces_residual():
    A0 = np.array([[3+1j, 0.2], [0.1, 2+0.5j]], dtype=complex)
    A_state = np.array([
        np.diag([0.1+0.02j, 0.03+0.01j]),
        np.diag([0.02+0.01j, 0.08+0.02j]),
    ])
    b = np.array([1.0, 0.5], dtype=complex)
    Hm = np.eye(2)
    Hloss = np.array([np.eye(2)*0.03, np.eye(2)*0.02], dtype=complex)
    problem = ParametricEMProblem(A0, A_state, b, Hm, Hloss)
    states = [np.array([x, y]) for x in (-0.2, 0.0, 0.2) for y in (-0.2, 0.0, 0.2)]
    reducer = ResidualGreedyEMReducer(problem)
    initial_model = ReducedEMModel(problem, reducer.initial_basis())
    initial_worst = max(initial_model.residual_dual_norm(s) for s in states)
    model = reducer.build(states, tolerance=1e-12)
    final_worst = max(model.residual_dual_norm(s) for s in states)
    assert final_worst <= initial_worst + 1e-14
    assert final_worst < 1e-10
