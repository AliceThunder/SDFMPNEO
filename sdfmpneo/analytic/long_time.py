"""Stable semigroup action, including the stationary limit of a response DAG."""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve


def realization_action(A, B, time):
    """Compute exp(A*t) B without overflowing A*t at very large finite t.

    DAG realizations are triangular, with stationary constant rows and strictly
    decaying remaining states. Separate those constants before evaluating a
    long-time transient. Repeated decay rates are retained, not diagonalized.
    Infinity denotes the exact stationary limit, never a training-window clamp.
    """
    t = float(time)
    if np.isnan(t) or t < 0:
        raise ValueError('time must be non-negative or positive infinity')
    A, B = np.asarray(A, complex), np.asarray(B, complex)
    norm = float(np.linalg.norm(A, np.inf))
    if t == 0 or norm == 0:
        return B.copy()
    if np.isfinite(t) and np.log2(t) + np.log2(norm) < 16:
        return expm(A*t) @ B
    static = np.flatnonzero(np.diag(A) == 0)
    transient = np.flatnonzero(np.diag(A) != 0)
    if np.any(A[static] != 0) or np.any(np.real(np.diag(A)[transient]) >= 0):
        if np.isinf(t):
            raise ValueError('realization has no supported stationary limit')
        return expm(A*t) @ B
    out = B.copy()
    if not len(transient):
        return out
    stable = A[np.ix_(transient, transient)]
    coupling = A[np.ix_(transient, static)] @ B[static]
    equilibrium = solve(stable, -coupling, assume_a='gen')
    if np.isinf(t):
        out[transient] = equilibrium
        return out
    # Scaling is computed in log space; even t near float64's maximum is safe.
    squarings = max(0, int(np.ceil(np.log2(t)+np.log2(norm))))
    E = expm(stable*np.ldexp(t, -squarings))
    for _ in range(squarings):
        E = E @ E
        if not np.any(E):
            break
    out[transient] = equilibrium + E @ (B[transient]-equilibrium)
    return out
