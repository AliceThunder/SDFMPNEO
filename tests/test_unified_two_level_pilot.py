import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sdfmpneo.unified_certified_local_solve as local_solver
from sdfmpneo.unified_two_level_local_krylov import (
    _relative_pilot,
    _should_galerkin_recover,
    _two_level_minimum_dofs,
)


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
        "linear_direct_fallback_max_dofs": 60000,
        "linear_transverse_direct_max_dofs": 100000,
        "linear_transverse_direct_fallback_max_dofs": 100000,
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


def test_two_level_default_respects_transverse_direct_fallback_band():
    cfg = {
        "linear_transverse_direct_max_dofs": 80000,
        "linear_transverse_direct_fallback_max_dofs": 110000,
    }
    assert _two_level_minimum_dofs(cfg) == 110001



def test_large_coarse_consistency_mismatch_triggers_early_galerkin_recovery():
    cfg = {}
    assert _should_galerkin_recover(
        3.146e-1,
        2.317e-1,
        cfg,
    )
    assert not _should_galerkin_recover(
        9.5e-1,
        2.317e-1,
        cfg,
    )
    assert not _should_galerkin_recover(
        3.146e-1,
        1e-3,
        cfg,
    )


def test_galerkin_recovery_thresholds_are_configurable():
    cfg = {
        "linear_two_level_galerkin_recovery_start_residual": 0.4,
        "linear_two_level_galerkin_recovery_consistency": 0.2,
    }
    assert _should_galerkin_recover(
        0.31,
        0.23,
        cfg,
    )
    assert not _should_galerkin_recover(
        0.41,
        0.23,
        cfg,
    )
