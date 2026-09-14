import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_certified_local_solve import _iterative_solve, _relative_residual


def test_ilu_lgmres_reaches_certified_true_residual():
    n = 240
    main = (4.0 + 0.2j) * np.ones(n, dtype=complex)
    off = -np.ones(n - 1, dtype=complex)
    A = sp.diags((off, main, off), (-1, 0, 1), format="csr")
    x_true = np.linspace(0.2, 1.1, n) + 1j * np.linspace(-0.3, 0.4, n)
    rhs = A @ x_true
    cfg = {
        "linear_iterative_maxiter": 12,
        "linear_iterative_inner_m": 20,
        "linear_ilu_drop_tolerance": 1e-3,
        "linear_ilu_fill_factor": 4.0,
        "linear_ilu_strong_drop_tolerance": 1e-4,
        "linear_ilu_strong_fill_factor": 8.0,
        "linear_ilu_shift_factor": 1e-9,
    }
    field, residual, history = _iterative_solve(
        A, np.asarray(rhs), np.zeros(n, dtype=complex), cfg, 1e-10
    )
    assert field is not None
    assert history
    assert residual <= 1e-10
    assert _relative_residual(A, field, rhs) <= 1e-10


def test_iterative_solver_can_improve_a_warm_start():
    n = 160
    A = sp.diags(
        (
            -0.5 * np.ones(n - 1),
            (3.0 + 0.1j) * np.ones(n),
            -0.5 * np.ones(n - 1),
        ),
        (-1, 0, 1),
        format="csr",
        dtype=complex,
    )
    x_true = np.sin(np.linspace(0.0, np.pi, n)).astype(complex)
    rhs = A @ x_true
    warm = x_true + 1e-3 * np.cos(np.linspace(0.0, 2.0 * np.pi, n))
    before = _relative_residual(A, warm, rhs)
    field, residual, _ = _iterative_solve(
        A,
        np.asarray(rhs),
        warm,
        {"linear_iterative_maxiter": 8, "linear_iterative_inner_m": 16},
        1e-10,
    )
    assert field is not None
    assert residual < before
    assert residual <= 1e-10
