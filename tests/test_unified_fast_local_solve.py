import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sdfmpneo.unified_certified_local_solve as local_solver
from sdfmpneo.unified_certified_local_solve import _iterative_solve, _relative_residual
from sdfmpneo.unified_fast_local_krylov import _defect_refine, _pilot_krylov
from sdfmpneo.unified_localized_self_solve import _solver_limits


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


def test_defect_refinement_reduces_true_residual_of_existing_field():
    n = 360
    A = sp.diags(
        (
            -np.ones(n - 1),
            (2.2 + 0.03j) * np.ones(n),
            -np.ones(n - 1),
        ),
        (-1, 0, 1),
        format="csr",
        dtype=complex,
    )
    x_true = np.sin(np.linspace(0.0, 4.0 * np.pi, n)).astype(complex)
    rhs = A @ x_true
    field0 = x_true + 2e-5 * np.cos(np.linspace(0.0, 3.0 * np.pi, n))
    before = _relative_residual(A, field0, rhs)
    ilu = spla.spilu(A.tocsc(), drop_tol=1e-2, fill_factor=2.0)
    M = spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)

    field, residual, history = _defect_refine(
        A,
        rhs,
        field0,
        M,
        local_solver._lgmres,
        _relative_residual,
        1e-10,
        steps=3,
        maxiter=12,
        inner_m=16,
        label="test-defect",
    )

    assert history
    assert residual < before
    assert np.isclose(_relative_residual(A, field, rhs), residual, rtol=0.0, atol=1e-15)
    assert residual <= 1e-10


def test_pilot_krylov_accepts_a_useful_preconditioner():
    n = 180
    A = sp.diags(
        (
            -np.ones(n - 1),
            (3.5 + 0.2j) * np.ones(n),
            -np.ones(n - 1),
        ),
        (-1, 0, 1),
        format="csr",
        dtype=complex,
    )
    truth = np.sin(np.linspace(0.0, np.pi, n)).astype(complex)
    rhs = A @ truth
    ilu = spla.spilu(A.tocsc(), drop_tol=1e-3, fill_factor=4.0)
    M = spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)

    candidate, residual, _info, _seconds, before, accepted = _pilot_krylov(
        A,
        rhs,
        np.zeros(n, dtype=complex),
        M,
        local_solver._lgmres,
        _relative_residual,
        maxiter=2,
        inner_m=8,
        accept_ratio=0.95,
    )

    assert accepted
    assert residual < 0.95 * before
    assert _relative_residual(A, candidate, rhs) <= residual * (1.0 + 1e-12)


def test_pilot_krylov_rejects_a_destructive_preconditioner():
    n = 120
    A = sp.eye(n, format="csr", dtype=complex)
    rhs = np.ones(n, dtype=complex)

    def bad_inverse(vector):
        return -1e6 * np.asarray(vector, complex)

    M = spla.LinearOperator(A.shape, matvec=bad_inverse, dtype=complex)
    _candidate, residual, _info, _seconds, before, accepted = _pilot_krylov(
        A,
        rhs,
        np.zeros(n, dtype=complex),
        M,
        local_solver._lgmres,
        _relative_residual,
        maxiter=1,
        inner_m=4,
        accept_ratio=0.95,
    )

    assert not accepted or residual < before



def test_localized_transverse_solver_uses_medium_direct_band():
    direct, fallback = _solver_limits(
        {
            "linear_direct_max_dofs": 60000,
            "linear_direct_fallback_max_dofs": 60000,
            "linear_transverse_direct_max_dofs": 100000,
            "linear_transverse_direct_fallback_max_dofs": 100000,
        }
    )
    assert direct == 100000
    assert fallback == 100000
    assert 64107 <= direct
    assert fallback < 118000
