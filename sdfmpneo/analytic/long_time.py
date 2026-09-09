"""Stable semigroup action, including the stationary limit of a response DAG."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
import os
from threading import RLock

import numpy as np
from scipy.linalg import expm, solve


_PROPAGATOR_CACHE_LIMITS = {
    "main": max(0, int(os.environ.get("SDFMPNEO_PROPAGATOR_CACHE_MB", "96"))) * 1024 * 1024,
    "candidate": max(0, int(os.environ.get("SDFMPNEO_CANDIDATE_PROPAGATOR_CACHE_MB", "128"))) * 1024 * 1024,
}
_PROPAGATOR_CACHES: dict[str, OrderedDict] = {
    name: OrderedDict() for name in _PROPAGATOR_CACHE_LIMITS
}
_PROPAGATOR_CACHE_BYTES = {name: 0 for name in _PROPAGATOR_CACHE_LIMITS}
_PROPAGATOR_CACHE_LOCK = RLock()


def clear_realization_propagator_cache(namespace: str | None = None) -> None:
    """Clear exact propagator LRUs; ``None`` clears main and candidate caches."""
    names = tuple(_PROPAGATOR_CACHES) if namespace is None else (str(namespace),)
    with _PROPAGATOR_CACHE_LOCK:
        for name in names:
            if name not in _PROPAGATOR_CACHES:
                raise ValueError(f"unknown realization cache namespace: {name}")
            _PROPAGATOR_CACHES[name].clear()
            _PROPAGATOR_CACHE_BYTES[name] = 0


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


def _cached_propagator(A: np.ndarray, t: float, namespace: str = "main") -> np.ndarray:
    """Return the exact B -> exp(A t)B map from a bounded namespace-local LRU."""
    name = str(namespace)
    if name not in _PROPAGATOR_CACHES:
        raise ValueError(f"unknown realization cache namespace: {name}")
    limit = _PROPAGATOR_CACHE_LIMITS[name]
    cache = _PROPAGATOR_CACHES[name]
    if limit <= 0:
        return _uncached_realization_action(A, np.eye(A.shape[0], dtype=complex), t)

    key = _propagator_key(A, t)
    with _PROPAGATOR_CACHE_LOCK:
        entry = cache.get(key)
        if entry is not None and np.array_equal(entry[0], A):
            cache.move_to_end(key)
            return entry[1]

    identity = np.eye(A.shape[0], dtype=complex)
    propagator = _uncached_realization_action(A, identity, t)
    stored_A = np.array(A, copy=True)
    size = int(stored_A.nbytes + propagator.nbytes)
    if size > limit:
        return propagator

    with _PROPAGATOR_CACHE_LOCK:
        existing = cache.get(key)
        if existing is not None and np.array_equal(existing[0], A):
            cache.move_to_end(key)
            return existing[1]
        if existing is not None:
            _PROPAGATOR_CACHE_BYTES[name] -= existing[2]
        cache[key] = (stored_A, propagator, size)
        _PROPAGATOR_CACHE_BYTES[name] += size
        cache.move_to_end(key)
        while _PROPAGATOR_CACHE_BYTES[name] > limit:
            _, (_, _, removed) = cache.popitem(last=False)
            _PROPAGATOR_CACHE_BYTES[name] -= removed
    return propagator


def realization_action(A, B, time, *, cache_namespace="main"):
    """Compute exp(A*t) B stably, reusing exact propagators across passes.

    ``cache_namespace`` changes only LRU ownership. The normal DAG and
    Gauss--Newton path uses ``main``; large temporary candidate scans may use
    ``candidate`` so they cannot evict the formal-network working set. Both
    namespaces store the same exact semigroup/stationary linear maps.
    """
    t = float(time)
    if np.isnan(t) or t < 0:
        raise ValueError("time must be non-negative or positive infinity")
    A, B = np.asarray(A, complex), np.asarray(B, complex)
    norm = float(np.linalg.norm(A, np.inf))
    if t == 0 or norm == 0:
        return B.copy()
    return _cached_propagator(A, t, namespace=cache_namespace) @ B
