import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sdfmpneo.unified_certified_local_solve as local_solver
import sdfmpneo.unified_two_level_local_krylov as two_level_local
from sdfmpneo.unified_accurate_residual import accurate_residual_vector
from sdfmpneo.unified_compensated_field import field_is_finite


def test_two_level_polish_recomputes_accurate_and_refines_fine_residual_first():
    assert bool(getattr(two_level_local, "_fine_residual_replacement_installed", False))

    n = 180
    diagonal = (2.0 + 0.15j) + np.linspace(0.0, 0.8, n)
    A = sp.diags(diagonal, format="csr", dtype=complex)
    exact = np.sin(np.linspace(0.0, np.pi, n)) + 1j * np.cos(
        np.linspace(0.0, 2.0 * np.pi, n)
    )
    rhs = np.asarray(A @ exact, complex).reshape(-1)
    perturbation = 4e-5 * (
        np.cos(np.linspace(0.0, 3.0 * np.pi, n))
        + 1j * np.sin(np.linspace(0.0, 4.0 * np.pi, n))
    )
    start = exact + perturbation
    true_before = local_solver._relative_residual(A, start, rhs)

    M = spla.LinearOperator(
        A.shape,
        matvec=lambda value: np.asarray(value, complex).reshape(-1) / diagonal,
        dtype=complex,
    )

    # Deliberately pass a stale reported residual.  The patch must independently
    # accumulate the original physical fine residual before refinement.
    polished, residual, history = two_level_local._galerkin_defect_polish(
        local_solver,
        object(),
        A,
        rhs,
        start,
        0.25 * true_before,
        M,
        1e-11,
        {
            "linear_two_level_residual_replacement_steps": 2,
            "linear_two_level_residual_replacement_maxiter": 8,
            "linear_two_level_residual_replacement_inner_m": 12,
            "linear_two_level_residual_replacement_backtracks": 2,
        },
    )

    assert history
    assert history[0]["solver"] == "two-level-accurate-residual-entry"
    assert np.isclose(history[0]["relative_residual"], true_before, rtol=1e-12, atol=0.0)
    assert history[0]["reported_accurate_discrepancy"] > 0.1
    assert "roundoff_bound_relative" in history[0]
    assert field_is_finite(polished)
    assert residual < true_before
    assert residual <= 1e-11
    final_defect, _stats = accurate_residual_vector(A, polished, rhs, target_relative=1e-11)
    final_relative = np.linalg.norm(final_defect) / np.linalg.norm(rhs)
    assert final_relative <= 1e-11
    replacement_entries = [
        item for item in history
        if item.get("solver", "").startswith("two-level-equilibrated-residual-replacement-")
    ]
    if replacement_entries:
        assert replacement_entries[-1]["linear_closure_relative_error"] <= 1e-11
