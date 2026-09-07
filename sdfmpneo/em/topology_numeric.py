from __future__ import annotations

import numpy as np
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
