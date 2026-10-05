import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_accurate_residual import (
    accurate_residual_vector,
    residual_roundoff_diagnostics,
)


def test_accurate_residual_recovers_sparse_cancellation_lost_by_standard_matvec():
    # The first CSR row represents 1e16 + 1 - 1e16 = 1 exactly at the level of
    # the stored binary64 coefficients.  Ordinary sequential binary64 summation
    # can lose the unit term; the compensated/extended residual must not.
    A = sp.csr_matrix(
        np.array(
            [
                [1.0e16, 1.0, -1.0e16],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=complex,
        )
    )
    x = np.ones(3, dtype=complex)
    rhs = np.ones(3, dtype=complex)

    standard = np.asarray(rhs - A @ x, complex).reshape(-1)
    accurate, stats = accurate_residual_vector(A, x, rhs, target_relative=1e-14)

    assert np.linalg.norm(standard) > 0.1
    assert np.linalg.norm(accurate) <= 1e-14
    assert stats["roundoff_bound_relative"] > 1e-3
    assert stats["accumulation_mode"] in {
        "extended-longdouble-csr",
        "compensated-double-double-csr",
    }


def test_roundoff_diagnostic_is_small_for_well_scaled_diagonal_problem():
    diagonal = np.linspace(1.0, 2.0, 64) + 0.1j
    A = sp.diags(diagonal, format="csr", dtype=complex)
    x = np.linspace(0.2, 1.2, 64) + 1j * np.linspace(-0.3, 0.4, 64)
    rhs = np.asarray(A @ x, complex).reshape(-1)
    stats = residual_roundoff_diagnostics(A, x, rhs)
    assert stats["standard_relative_residual"] <= 1e-15
    assert stats["roundoff_bound_relative"] <= 1e-13
