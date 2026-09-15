import numpy as np
import scipy.sparse as sp

import sdfmpneo.unified_transverse_ilu as transverse_ilu


class _FakeILU:
    def solve(self, rhs):
        return np.asarray(rhs)


def test_spilu_zero_pivot_recovery_keeps_mmd_then_adds_tiny_pivot_floor(monkeypatch):
    n = 32
    matrix = sp.diags(
        (
            -np.ones(n - 1),
            (2.0 + 0.1j) * np.ones(n),
            -np.ones(n - 1),
        ),
        (-1, 0, 1),
        format="csr",
        dtype=complex,
    )
    calls = []

    def fake_spilu(candidate, **kwargs):
        calls.append((candidate.copy(), dict(kwargs)))
        if len(calls) < 3:
            raise RuntimeError("Factor is exactly singular")
        return _FakeILU()

    monkeypatch.setattr(transverse_ilu.spla, "spilu", fake_spilu)
    ilu, stats = transverse_ilu._factor_with_pivot_recovery(
        matrix,
        drop_tol=5e-3,
        fill_factor=4.0,
    )

    assert isinstance(ilu, _FakeILU)
    assert len(calls) == 3
    assert all(call[1]["permc_spec"] == "MMD_AT_PLUS_A" for call in calls)
    assert stats["factor_recovery_attempt"] == 3
    assert stats["factor_ordering"] == "MMD_AT_PLUS_A"
    assert np.isclose(stats["factor_pivot_regularization"], 1e-8)
    original_diag = np.asarray(matrix.diagonal())
    recovered_diag = np.asarray(calls[2][0].diagonal())
    assert np.all(np.real(recovered_diag - original_diag) > 0.0)


def test_large_factorization_starts_with_mmd_and_regularizes_before_colamd(monkeypatch):
    n = 200001
    matrix = sp.eye(n, format="csr", dtype=complex)
    calls = []

    def fake_spilu(candidate, **kwargs):
        calls.append((candidate.copy(), dict(kwargs)))
        if len(calls) < 2:
            raise RuntimeError("Factor is exactly singular")
        return _FakeILU()

    monkeypatch.setattr(transverse_ilu.spla, "spilu", fake_spilu)
    _ilu, stats = transverse_ilu._factor_with_pivot_recovery(
        matrix,
        drop_tol=5e-3,
        fill_factor=4.0,
    )

    assert len(calls) == 2
    assert calls[0][1]["permc_spec"] == "MMD_AT_PLUS_A"
    assert calls[1][1]["permc_spec"] == "MMD_AT_PLUS_A"
    assert stats["factor_ordering"] == "MMD_AT_PLUS_A"
    assert np.isclose(stats["factor_pivot_regularization"], 1e-10)


def test_colamd_is_only_used_after_mmd_recovery_attempts_fail(monkeypatch):
    n = 200001
    matrix = sp.eye(n, format="csr", dtype=complex)
    seen = []

    def fake_spilu(candidate, **kwargs):
        seen.append(dict(kwargs))
        if len(seen) <= 5:
            raise RuntimeError("Factor is exactly singular")
        return _FakeILU()

    monkeypatch.setattr(transverse_ilu.spla, "spilu", fake_spilu)
    _ilu, stats = transverse_ilu._factor_with_pivot_recovery(
        matrix,
        drop_tol=5e-3,
        fill_factor=4.0,
    )

    assert len(seen) == 6
    assert all(row["permc_spec"] == "MMD_AT_PLUS_A" for row in seen[:5])
    assert seen[5]["permc_spec"] == "COLAMD"
    assert stats["factor_ordering"] == "COLAMD"
