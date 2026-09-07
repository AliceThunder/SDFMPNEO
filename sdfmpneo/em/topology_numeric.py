from __future__ import annotations

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def certified_integer_topology_matrix(matrix, *, name: str = "topology matrix") -> sp.csr_matrix:
    """Convert an exactly represented real-integer sparse topology matrix.

    Electromagnetic incidence/gauge matrices are discrete topological objects.
    Some production objects store them in complex dtype because they participate
    in complex field algebra, but their entries must still be exactly real
    integers.  Converting a complex sparse matrix directly to ``dtype=int`` lets
    NumPy silently discard an imaginary component.  This helper instead verifies
    the topological invariant first and only then constructs a fresh integer CSR
    matrix from validated real data.

    No numerical closeness threshold is used: a nonzero imaginary component or a
    non-integer real entry means the object is no longer the declared exact
    topology and is rejected fail-closed.
    """

    raw = sp.csr_matrix(matrix, dtype=complex).copy()
    raw.sum_duplicates()
    raw.eliminate_zeros()
    data = np.asarray(raw.data, dtype=complex)
    if data.size and np.any(data.imag != 0.0):
        raise ValueError(f"{name} has a nonzero imaginary component")

    real = data.real
    rounded = np.rint(real)
    if real.size and np.any(real != rounded):
        raise ValueError(f"{name} has a non-integer entry")

    out = sp.csr_matrix(
        (
            rounded.astype(np.int64, copy=False),
            raw.indices.copy(),
            raw.indptr.copy(),
        ),
        shape=raw.shape,
    )
    out.sum_duplicates()
    out.eliminate_zeros()
    return out


def fail_closed_lu_factor(matrix: np.ndarray, *, singular_message: str):
    """Factor a local dense block while turning LAPACK singularity into failure.

    ``scipy.linalg.lu_factor`` reports an exactly zero U pivot as
    ``LinAlgWarning`` instead of raising.  A certified path must not continue with
    such a factor and must not rely on warning filters.  This wrapper promotes
    that LAPACK diagnosis to the declared deterministic ``ValueError`` and also
    checks the returned factor defensively.
    """

    A = np.asarray(matrix, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] == 0:
        raise ValueError("local LU block must be non-empty and square")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", scipy.linalg.LinAlgWarning)
            factor, pivots = scipy.linalg.lu_factor(A, check_finite=False)
    except scipy.linalg.LinAlgWarning as exc:
        raise ValueError(singular_message) from exc
    if np.any(np.diag(factor) == 0.0):
        raise ValueError(singular_message)
    return factor, pivots
