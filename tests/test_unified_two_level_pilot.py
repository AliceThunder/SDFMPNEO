import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sdfmpneo.unified_certified_local_solve as local_solver
from sdfmpneo.unified_two_level_local_krylov import _relative_pilot, _two_level_minimum_dofs


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


def test_two_level_default_starts_immediately_after_transverse_direct_band():
    cfg = {
        "linear_direct_max_dofs": 60000,
        "linear_transverse_direct_max_dofs": 100000,
    }
    assert _two_level_minimum_dofs(cfg) == 100001
    assert 108327 >= _two_level_minimum_dofs(cfg)
    assert 118000 >= _two_level_minimum_dofs(cfg)


def test_two_level_explicit_threshold_still_overrides_default_policy():
    cfg = {
        "linear_transverse_direct_max_dofs": 100000,
        "linear_two_level_min_dofs": 120000,
    }
    assert _two_level_minimum_dofs(cfg) == 120000
