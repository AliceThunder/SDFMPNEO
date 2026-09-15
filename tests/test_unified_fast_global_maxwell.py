import numpy as np
import scipy.sparse as sp

import sdfmpneo.unified_certified_local_solve as local_solver
from sdfmpneo.unified_fast_global_maxwell import solve_multi_rhs


class _Background:
    background_config = {
        "linear_solver": {
            "relative_residual_tolerance": 1e-10,
            "direct_max_dofs": 1,
            "iterative_maxiter": 16,
            "iterative_inner_m": 24,
            "ilu_drop_tolerance": 1e-3,
            "ilu_fill_factor": 4.0,
            "ilu_strong_drop_tolerance": 1e-4,
            "ilu_strong_fill_factor": 8.0,
            "ilu_shift_factor": 3e-2,
            "ilu_strong_shift_factor": 1e-1,
        }
    }


def test_global_multi_rhs_iterative_solver_shares_policy_and_certifies_true_residual():
    n = 320
    A = sp.diags(
        (
            -np.ones(n - 1),
            (4.0 + 0.15j) * np.ones(n),
            -np.ones(n - 1),
        ),
        (-1, 0, 1),
        format="csr",
        dtype=complex,
    )
    x0 = np.linspace(0.1, 1.0, n) + 1j * np.linspace(-0.2, 0.3, n)
    x1 = np.sin(np.linspace(0.0, np.pi, n)) + 0.2j * np.cos(np.linspace(0.0, np.pi, n))
    truth = np.column_stack((x0, x1))
    B = A @ truth

    X, residual, history = solve_multi_rhs(_Background(), A, B, local_solver)

    assert X.shape == truth.shape
    assert history
    assert residual <= 1e-10
    scale = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
    actual = np.linalg.norm(B - A @ X, axis=0) / scale
    assert float(np.max(actual)) <= 1e-10
