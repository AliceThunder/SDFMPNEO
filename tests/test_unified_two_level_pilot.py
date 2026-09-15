import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sdfmpneo.unified_certified_local_solve as local_solver
from sdfmpneo.unified_two_level_local_krylov import _relative_pilot


def test_two_level_pilot_target_is_relative_to_warm_start_residual():
    n = 24
    A = sp.eye(n, format="csr", dtype=complex)
    rhs = np.ones(n, dtype=complex)
    x0 = 0.9 * np.ones(n, dtype=complex)
    M = spla.LinearOperator(A.shape, matvec=lambda value: np.asarray(value), dtype=complex)

    candidate, after, info, _seconds, before, accepted, pilot_rtol = _relative_pilot(
        A,
        rhs,
        x0,
        M,
        local_solver._lgmres,
        local_solver._relative_residual,
        maxiter=2,
        inner_m=4,
        accept_ratio=0.8,
    )

    assert np.isclose(before, 0.1, rtol=0.0, atol=1e-14)
    assert np.isclose(pilot_rtol, 0.05, rtol=0.0, atol=1e-14)
    assert info == 0
    assert after < before * 0.8
    assert accepted
    assert np.all(np.isfinite(candidate))
