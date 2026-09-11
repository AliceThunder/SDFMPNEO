from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from threading import Lock

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover
    threadpool_limits = None

_EXECUTORS: dict[int, ThreadPoolExecutor] = {}
_EXECUTOR_LOCK = Lock()


def training_point_workers() -> int:
    raw = os.environ.get("SDFMPNEO_POINT_WORKERS")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("SDFMPNEO_POINT_WORKERS must be a positive integer") from exc
        if value < 1:
            raise ValueError("SDFMPNEO_POINT_WORKERS must be a positive integer")
        return value
    return max(1, min(8, os.cpu_count() or 1))


def training_blas_threads() -> int:
    raw = os.environ.get("SDFMPNEO_BLAS_THREADS", "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("SDFMPNEO_BLAS_THREADS must be a positive integer") from exc
    if value < 1:
        raise ValueError("SDFMPNEO_BLAS_THREADS must be a positive integer")
    return value


def training_parallelism() -> dict[str, int]:
    return {
        "cpu_count": int(os.cpu_count() or 1),
        "point_workers": int(training_point_workers()),
        "blas_threads_per_point": int(training_blas_threads()),
    }


def _blas_limit_context():
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=training_blas_threads())


def _executor(workers: int) -> ThreadPoolExecutor:
    with _EXECUTOR_LOCK:
        pool = _EXECUTORS.get(workers)
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="sdfmpneo-point",
            )
            _EXECUTORS[workers] = pool
        return pool


def _ordered_map(function, items, *, monitor=None, progress=None):
    """Deterministic point-parallel map with BLAS oversubscription control."""
    items = list(items)
    total = len(items)
    if progress is not None:
        progress(0, total)
    workers = min(training_point_workers(), total) if total else 1
    if workers <= 1:
        out = []
        for index, item in enumerate(items, 1):
            if monitor is not None:
                monitor.checkpoint()
            out.append(function(item))
            if progress is not None:
                progress(index, total)
        return out

    out = []
    batch_size = max(workers, min(2 * workers, 16))
    pool = _executor(workers)
    with _blas_limit_context():
        for start in range(0, total, batch_size):
            if monitor is not None:
                monitor.checkpoint()
            batch = items[start:start + batch_size]
            out.extend(pool.map(function, batch))
            if progress is not None:
                progress(min(start + len(batch), total), total)
            if monitor is not None:
                monitor.checkpoint()
    return out


__all__ = [
    "training_point_workers", "training_blas_threads", "training_parallelism",
    "_ordered_map",
]
