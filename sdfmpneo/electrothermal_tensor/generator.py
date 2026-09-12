"""Resumable, embarrassingly-parallel physics snapshot generation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .dataset import QuadraticJouleDataset


def _atomic_partial(path: Path, *, states, geometries, tensors, completed) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            states=np.asarray(states, dtype=float),
            geometries=np.asarray(geometries, dtype=float),
            tensors=np.asarray(tensors, dtype=float),
            completed=np.asarray(completed, dtype=bool),
        )
    temporary.replace(path)


def generate_snapshots_resumable(
    states: np.ndarray,
    geometries: np.ndarray,
    tensor_factory: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    checkpoint_path: str | Path,
    checkpoint_every: int = 16,
    max_workers: int = 1,
    split_seed: int = 0,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    metadata: Mapping[str, object] | None = None,
    final_path: str | Path | None = None,
) -> QuadraticJouleDataset:
    """Generate ``G(a,g)`` snapshots and resume from a safe disk boundary.

    The input sample set is immutable across resume.  A partial checkpoint is
    rejected if states or geometries differ, preventing accidental relabeling of
    a previously generated dataset.  Threads operate only across independent
    snapshots; callers should use ``max_workers=1`` for application models whose
    geometry/context cache is not thread-safe.
    """
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or len(a) != len(g) or len(a) < 1:
        raise ValueError("states/geometries must be aligned nonempty matrices")
    checkpoint_every = int(checkpoint_every)
    max_workers = int(max_workers)
    if checkpoint_every < 1 or max_workers < 1:
        raise ValueError("checkpoint_every and max_workers must be positive")
    partial = Path(checkpoint_path)

    tensors = None
    completed = np.zeros(len(a), dtype=bool)
    if partial.exists():
        with np.load(partial, allow_pickle=False) as data:
            saved_a = np.asarray(data["states"], dtype=float)
            saved_g = np.asarray(data["geometries"], dtype=float)
            if not np.array_equal(saved_a, a) or not np.array_equal(saved_g, g):
                raise ValueError("partial snapshot checkpoint belongs to a different sample set")
            tensors = np.asarray(data["tensors"], dtype=float)
            completed = np.asarray(data["completed"], dtype=bool)
        if completed.shape != (len(a),) or tensors.shape[0] != len(a):
            raise ValueError("partial snapshot checkpoint is malformed")

    pending = np.flatnonzero(~completed)
    if tensors is None:
        first = int(pending[0])
        first_tensor = np.asarray(tensor_factory(a[first], g[first]), dtype=float)
        if first_tensor.ndim != 3 or first_tensor.shape[0] != a.shape[1] or first_tensor.shape[-1] != first_tensor.shape[-2]:
            raise ValueError("tensor_factory returned an incompatible tensor")
        tensors = np.empty((len(a),) + first_tensor.shape, dtype=float)
        tensors[first] = first_tensor
        completed[first] = True
        _atomic_partial(partial, states=a, geometries=g, tensors=tensors, completed=completed)
        pending = np.flatnonzero(~completed)

    expected_shape = tensors.shape[1:]

    def evaluate(index: int):
        value = np.asarray(tensor_factory(a[index], g[index]), dtype=float)
        if value.shape != expected_shape or np.any(~np.isfinite(value)):
            raise ValueError("tensor_factory changed shape or returned non-finite values")
        return index, value

    since_checkpoint = 0
    if max_workers == 1:
        iterator = map(evaluate, (int(i) for i in pending))
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        iterator = executor.map(evaluate, (int(i) for i in pending))
    try:
        for index, value in iterator:
            tensors[index] = value
            completed[index] = True
            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every:
                _atomic_partial(partial, states=a, geometries=g, tensors=tensors, completed=completed)
                since_checkpoint = 0
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        _atomic_partial(partial, states=a, geometries=g, tensors=tensors, completed=completed)

    if not np.all(completed):
        raise RuntimeError("snapshot generation did not complete")
    dataset = QuadraticJouleDataset.from_tensors(
        a,
        g,
        tensors,
        split_seed=split_seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        metadata={} if metadata is None else dict(metadata),
    )
    if final_path is not None:
        dataset.save(final_path)
    return dataset


__all__ = ["generate_snapshots_resumable"]
