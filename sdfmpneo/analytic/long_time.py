"""Stable semigroup action, including the stationary limit of a response DAG."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b
import os
from threading import Event, RLock

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

# Training normally observes a realization only through c^T exp(A t) B and its
# time derivative.  Keeping complete d-by-d propagators for those calls wastes
# both memory and the subsequent matrix-matrix multiply.  The observation cache
# stores A,c exactly once per structure and only two length-d rows per time.
_OBSERVATION_CACHE_LIMIT = max(
    0, int(os.environ.get("SDFMPNEO_OBSERVATION_CACHE_MB", "192"))
) * 1024 * 1024
_OBSERVATION_CACHE: OrderedDict = OrderedDict()
_OBSERVATION_CACHE_BYTES = 0
_OBSERVATION_STRUCTURES: dict[tuple, list[tuple[np.ndarray, np.ndarray, int]]] = {}
_OBSERVATION_NEXT_TOKEN = 0
_OBSERVATION_INFLIGHT: dict[tuple[int, float], Event] = {}
_OBSERVATION_CACHE_LOCK = RLock()


def clear_realization_propagator_cache(namespace: str | None = None) -> None:
    """Clear exact propagator LRUs; ``None`` clears main and candidate caches."""
    names = tuple(_PROPAGATOR_CACHES) if namespace is None else (str(namespace),)
    with _PROPAGATOR_CACHE_LOCK:
        for name in names:
            if name not in _PROPAGATOR_CACHES:
                raise ValueError(f"unknown realization cache namespace: {name}")
            _PROPAGATOR_CACHES[name].clear()
            _PROPAGATOR_CACHE_BYTES[name] = 0


def clear_realization_observation_cache() -> None:
    """Clear compact c^T exp(A t) observation rows and their exact structures."""
    global _OBSERVATION_CACHE_BYTES, _OBSERVATION_NEXT_TOKEN
    with _OBSERVATION_CACHE_LOCK:
        _OBSERVATION_CACHE.clear()
        _OBSERVATION_CACHE_BYTES = 0
        _OBSERVATION_STRUCTURES.clear()
        _OBSERVATION_NEXT_TOKEN = 0
        for event in _OBSERVATION_INFLIGHT.values():
            event.set()
        _OBSERVATION_INFLIGHT.clear()


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


def _observation_structure_token(A: np.ndarray, c: np.ndarray) -> int:
    """Intern an A,c pair with exact collision checking and return a stable token."""
    global _OBSERVATION_NEXT_TOKEN
    matrix = np.ascontiguousarray(A, dtype=complex)
    output = np.ascontiguousarray(c, dtype=complex)
    digest = blake2b(digest_size=16)
    digest.update(matrix.view(np.uint8))
    digest.update(output.view(np.uint8))
    key = (matrix.shape, matrix.dtype.str, output.shape, output.dtype.str, digest.digest())
    with _OBSERVATION_CACHE_LOCK:
        bucket = _OBSERVATION_STRUCTURES.setdefault(key, [])
        for stored_A, stored_c, token in bucket:
            if np.array_equal(stored_A, matrix) and np.array_equal(stored_c, output):
                return token
        token = _OBSERVATION_NEXT_TOKEN
        _OBSERVATION_NEXT_TOKEN += 1
        bucket.append((matrix.copy(), output.copy(), token))
        return token


def _uncached_observation_rows(A: np.ndarray, c: np.ndarray, t: float):
    """Return rows mapping B directly to value and time derivative."""
    if t == 0:
        return c.copy(), np.asarray(c @ A, dtype=complex)
    norm = float(np.linalg.norm(A, np.inf))
    if norm == 0:
        return c.copy(), np.zeros_like(c)
    propagator = _uncached_realization_action(
        A, np.eye(A.shape[0], dtype=complex), t
    )
    value_row = np.asarray(c @ propagator, dtype=complex)
    if np.isposinf(t):
        slope_row = np.zeros_like(value_row)
    else:
        slope_row = np.asarray((c @ A) @ propagator, dtype=complex)
    return value_row, slope_row


def _cached_observation_rows(A: np.ndarray, c: np.ndarray, t: float):
    """Compact exact observation cache with one full structure copy per A,c pair."""
    global _OBSERVATION_CACHE_BYTES
    if _OBSERVATION_CACHE_LIMIT <= 0:
        return _uncached_observation_rows(A, c, t)

    token = _observation_structure_token(A, c)
    key = (token, float(t))
    while True:
        with _OBSERVATION_CACHE_LOCK:
            entry = _OBSERVATION_CACHE.get(key)
            if entry is not None:
                _OBSERVATION_CACHE.move_to_end(key)
                return entry[0], entry[1]
            event = _OBSERVATION_INFLIGHT.get(key)
            if event is None:
                event = Event()
                _OBSERVATION_INFLIGHT[key] = event
                builder = True
            else:
                builder = False
        if builder:
            break
        event.wait()

    try:
        value_row, slope_row = _uncached_observation_rows(A, c, t)
        value_row = np.array(value_row, copy=True)
        slope_row = np.array(slope_row, copy=True)
        size = int(value_row.nbytes + slope_row.nbytes)
        with _OBSERVATION_CACHE_LOCK:
            if size <= _OBSERVATION_CACHE_LIMIT:
                _OBSERVATION_CACHE[key] = (value_row, slope_row, size)
                _OBSERVATION_CACHE_BYTES += size
                _OBSERVATION_CACHE.move_to_end(key)
                while _OBSERVATION_CACHE_BYTES > _OBSERVATION_CACHE_LIMIT:
                    _, (_, _, removed) = _OBSERVATION_CACHE.popitem(last=False)
                    _OBSERVATION_CACHE_BYTES -= removed
            event = _OBSERVATION_INFLIGHT.pop(key, None)
            if event is not None:
                event.set()
        return value_row, slope_row
    except Exception:
        with _OBSERVATION_CACHE_LOCK:
            event = _OBSERVATION_INFLIGHT.pop(key, None)
            if event is not None:
                event.set()
        raise


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


def realization_observation_action(A, B, c, time):
    """Evaluate c^T exp(A t)B and its derivative through compact cached rows.

    This is mathematically identical to ``realization_action`` followed by the
    two output contractions, but the cache is O(d) per time instead of O(d^2).
    It is intended for the fixed DAG/Gauss--Newton path; candidate scans that
    need internal states continue to use ``realization_action``.
    """
    t = float(time)
    if np.isnan(t) or t < 0:
        raise ValueError("time must be non-negative or positive infinity")
    A = np.asarray(A, dtype=complex)
    B = np.asarray(B, dtype=complex)
    c = np.asarray(c, dtype=complex)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if B.shape[0] != A.shape[0] or c.shape != (A.shape[0],):
        raise ValueError("observation dimensions do not match")
    value_row, slope_row = _cached_observation_rows(A, c, t)
    values = value_row @ B
    slopes = slope_row @ B
    return values, slopes
