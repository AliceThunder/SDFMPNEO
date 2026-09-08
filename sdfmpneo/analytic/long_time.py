"""Stable semigroup action, including the stationary limit of a response DAG."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
import os
from threading import RLock

import numpy as np
from scipy.linalg import expm, solve


_PROPAGATOR_CACHE_LIMIT = max(
    0, int(os.environ.get("SDFMPNEO_PROPAGATOR_CACHE_MB", "96"))
) * 1024 * 1024
_PROPAGATOR_CACHE: OrderedDict[tuple, tuple[np.ndarray, np.ndarray, int]] = OrderedDict()
_PROPAGATOR_CACHE_BYTES = 0
_PROPAGATOR_CACHE_LOCK = RLock()


def clear_realization_propagator_cache() -> None:
    """Clear cached exp(A t) actions; useful for tests and memory-sensitive callers."""
    global _PROPAGATOR_CACHE_BYTES
    with _PROPAGATOR_CACHE_LOCK:
        _PROPAGATOR_CACHE.clear()
        _PROPAGATOR_CACHE_BYTES = 0


def _uncached_realization_action(A: np.ndarray, B: np.ndarray, t: float) -> np.ndarray:
    """Original exact stable action, without propagator reuse."""
    norm = float(np.linalg.norm(A, np.inf))
    if t == 0 or norm == 0:
        return B.copy()
    if np.isfinite(t) and np.log2(t) + np.log2(norm) < 16:
        return expm(A * t) @ B
    static = np.flatnonzero(np.diag(A) == 0)
    transient = np.flatnonzero(np.diag(A) != 0)
    if np.any(A[static] != 0) or np.any(np.real(np.diag(A)[transient]) >= 0):
        if np.isinf(t):
            raise ValueError("realization has no supported stationary limit")
        return expm(A * t) @ B
    out = B.copy()
    if not len(transient):
        return out
    stable = A[np.ix_(transient, transient)]
    coupling = A[np.ix_(transient, static)] @ B[static]
    equilibrium = solve(stable, -coupling, assume_a="gen")
    if np.isinf(t):
        out[transient] = equilibrium
        return out
    squarings = max(0, int(np.ceil(np.log2(t) + np.log2(norm))))
    E = expm(stable * np.ldexp(t, -squarings))
    for _ in range(squarings):
        E = E @ E
        if not np.any(E):
            break
    out[transient] = equilibrium + E @ (B[transient] - equilibrium)
    return out


def _propagator_key(A: np.ndarray, t: float) -> tuple:
    contiguous = np.ascontiguousarray(A)
    digest = blake2b(contiguous.view(np.uint8), digest_size=16).digest()
    return contiguous.shape, contiguous.dtype.str, float(t), digest


def _cached_propagator(A: np.ndarray, t: float) -> np.ndarray:
    """Return the exact linear map B -> exp(A t)B with bounded LRU reuse."""
    global _PROPAGATOR_CACHE_BYTES
    if _PROPAGATOR_CACHE_LIMIT <= 0:
        return _uncached_realization_action(A, np.eye(A.shape[0], dtype=complex), t)

    key = _propagator_key(A, t)
    with _PROPAGATOR_CACHE_LOCK:
        entry = _PROPAGATOR_CACHE.get(key)
        if entry is not None and np.array_equal(entry[0], A):
            _PROPAGATOR_CACHE.move_to_end(key)
            return entry[1]

    identity = np.eye(A.shape[0], dtype=complex)
    propagator = _uncached_realization_action(A, identity, t)
    stored_A = np.array(A, copy=True)
    size = int(stored_A.nbytes + propagator.nbytes)
    if size > _PROPAGATOR_CACHE_LIMIT:
        return propagator

    with _PROPAGATOR_CACHE_LOCK:
        existing = _PROPAGATOR_CACHE.get(key)
        if existing is not None and np.array_equal(existing[0], A):
            _PROPAGATOR_CACHE.move_to_end(key)
            return existing[1]
        if existing is not None:
            _PROPAGATOR_CACHE_BYTES -= existing[2]
        _PROPAGATOR_CACHE[key] = (stored_A, propagator, size)
        _PROPAGATOR_CACHE_BYTES += size
        _PROPAGATOR_CACHE.move_to_end(key)
        while _PROPAGATOR_CACHE_BYTES > _PROPAGATOR_CACHE_LIMIT:
            _, (_, _, removed) = _PROPAGATOR_CACHE.popitem(last=False)
            _PROPAGATOR_CACHE_BYTES -= removed
    return propagator


def realization_action(A, B, time):
    """Compute exp(A*t) B stably, reusing exact propagators across training passes.

    DAG realizations are triangular, with stationary constant rows and strictly
    decaying remaining states. Repeated decay rates are retained, not
    diagonalized. Infinity denotes the exact stationary limit, never a
    training-window clamp. Caching changes only reuse: the returned action is
    the same linear semigroup/stationary map as the uncached implementation.
    """
    t = float(time)
    if np.isnan(t) or t < 0:
        raise ValueError("time must be non-negative or positive infinity")
    A, B = np.asarray(A, complex), np.asarray(B, complex)
    norm = float(np.linalg.norm(A, np.inf))
    if t == 0 or norm == 0:
        return B.copy()
    return _cached_propagator(A, t) @ B
