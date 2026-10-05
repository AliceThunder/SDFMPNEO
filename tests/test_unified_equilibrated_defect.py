import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sdfmpneo.unified_equilibrated_defect import build_equilibrated_defect_system


def test_equilibrated_operator_and_solution_mapping_are_exactly_equivalent():
    n = 96
    diagonal = np.logspace(-8, 8, n) * (1.0 + 0.05j)
    off = 1e-3 * np.sqrt(np.abs(diagonal[:-1] * diagonal[1:]))
    A = sp.diags((off, diagonal, off), (-1, 0, 1), format="csr", dtype=complex)
    rng = np.random.default_rng(23)
    defect = rng.standard_normal(n) + 1j * rng.standard_normal(n)

    lu = spla.splu(A.tocsc())
    M = spla.LinearOperator(A.shape, matvec=lu.solve, dtype=complex)
    system = build_equilibrated_defect_system(A, defect, M, iterations=5)

    R = sp.diags(system.row_scale, format="csr")
    C = sp.diags(system.column_scale, format="csr")
    explicit = (R @ A @ C).tocsr()
    y = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    assert np.allclose(system.operator @ y, explicit @ y, rtol=2e-13, atol=1e-13)

    y_exact = np.asarray(spla.spsolve(explicit.tocsc(), system.rhs), complex).reshape(-1)
    delta = system.physical_correction(y_exact)
    rel = np.linalg.norm(defect - A @ delta) / np.linalg.norm(defect)
    assert rel <= 1e-11

    probe = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    expected = np.asarray(spla.spsolve(explicit.tocsc(), probe), complex).reshape(-1)
    actual = np.asarray(system.preconditioner @ probe, complex).reshape(-1)
    assert np.allclose(actual, expected, rtol=2e-11, atol=1e-11)


def test_ruiz_equilibration_reduces_large_row_and_column_spans():
    n = 80
    diagonal = np.logspace(-10, 10, n) * (1.0 + 0.1j)
    A = sp.diags(diagonal, format="csr", dtype=complex)
    defect = np.ones(n, complex)
    system = build_equilibrated_defect_system(A, defect, None, iterations=4)

    assert system.row_span_before > 1e15
    assert system.column_span_before > 1e15
    assert system.row_span_after < 1e4
    assert system.column_span_after < 1e4
