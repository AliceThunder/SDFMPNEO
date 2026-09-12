"""Resumable, embarrassingly-parallel physics snapshot generation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .dataset import QuadraticJouleDataset

_CHECKPOINT_FORMAT_VERSION = 2


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".tensors.npy")


def _validate_tensor_shape(tensor_shape, thermal_rank: int) -> tuple[int, int, int]:
    shape = tuple(int(v) for v in tensor_shape)
    if (
        len(shape) != 3
        or shape[0] != int(thermal_rank)
        or shape[1] < 1
        or shape[1] != shape[2]
    ):
        raise ValueError("partial snapshot checkpoint tensor shape is incompatible")
    return shape


def _atomic_checkpoint(
    path: Path,
    *,
    states: np.ndarray,
    geometries: np.ndarray,
    completed: np.ndarray,
    tensor_shape,
    physical_signature: str | None,
) -> None:
    """Atomically update only small checkpoint metadata/bitmap.

    Tensor values live in a separately flushed NPY memmap and are never
    recompressed/re-written at each checkpoint boundary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(
            output,
            format_version=np.array(_CHECKPOINT_FORMAT_VERSION, dtype=np.int64),
            state_shape=np.asarray(states.shape, dtype=np.int64),
            geometry_shape=np.asarray(geometries.shape, dtype=np.int64),
            state_hash=np.array(_sha256_array(states)),
            geometry_hash=np.array(_sha256_array(geometries)),
            tensor_shape=np.asarray(tuple(int(v) for v in tensor_shape), dtype=np.int64),
            completed=np.asarray(completed, dtype=bool),
            physical_signature=np.array("" if physical_signature is None else str(physical_signature)),
        )
    temporary.replace(path)


def _validate_checkpoint_samples(data, states: np.ndarray, geometries: np.ndarray) -> None:
    version = int(data["format_version"]) if "format_version" in data.files else -1
    if version != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "legacy snapshot checkpoint is not provenance-safe; delete it and regenerate"
        )
    if tuple(np.asarray(data["state_shape"], dtype=int)) != states.shape:
        raise ValueError("partial snapshot checkpoint belongs to a different state sample set")
    if tuple(np.asarray(data["geometry_shape"], dtype=int)) != geometries.shape:
        raise ValueError("partial snapshot checkpoint belongs to a different geometry sample set")
    if str(data["state_hash"]) != _sha256_array(states):
        raise ValueError("partial snapshot checkpoint belongs to a different state sample set")
    if str(data["geometry_hash"]) != _sha256_array(geometries):
        raise ValueError("partial snapshot checkpoint belongs to a different geometry sample set")


def _physical_signature(metadata: Mapping[str, object] | None) -> str | None:
    if metadata is None:
        return None
    value = metadata.get("physical_signature")
    return None if value is None else str(value)


def _close_memmap(value) -> None:
    if isinstance(value, np.memmap):
        value.flush()
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()


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
    """Generate ``G(a,g)`` snapshots with bounded checkpoint I/O.

    Resume is bound to the exact state/geometry sample arrays and, when present,
    the physical signature. Tensor values are stored once in an NPY memmap;
    checkpoint updates rewrite only a small completed bitmap and hashes. Threads
    operate only across independent snapshots. Use ``max_workers=1`` for
    application models whose geometry/context cache is not thread-safe.
    """
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or len(a) != len(g) or len(a) < 1:
        raise ValueError("states/geometries must be aligned nonempty matrices")
    if np.any(~np.isfinite(a)) or np.any(~np.isfinite(g)):
        raise ValueError("snapshot sample locations must be finite")
    checkpoint_every = int(checkpoint_every)
    max_workers = int(max_workers)
    if checkpoint_every < 1 or max_workers < 1:
        raise ValueError("checkpoint_every and max_workers must be positive")
    partial = Path(checkpoint_path)
    sidecar = _sidecar_path(partial)
    expected_signature = _physical_signature(metadata)

    tensors = None
    completed = np.zeros(len(a), dtype=bool)
    if partial.exists():
        if not sidecar.exists():
            raise ValueError("snapshot checkpoint tensor sidecar is missing")
        with np.load(partial, allow_pickle=False) as data:
            _validate_checkpoint_samples(data, a, g)
            saved_signature = str(data["physical_signature"])
            if saved_signature != ("" if expected_signature is None else expected_signature):
                raise ValueError("partial snapshot checkpoint physical signature differs")
            tensor_shape = _validate_tensor_shape(
                np.asarray(data["tensor_shape"], dtype=int).tolist(),
                a.shape[1],
            )
            completed = np.asarray(data["completed"], dtype=bool)
        if completed.shape != (len(a),):
            raise ValueError("partial snapshot checkpoint is malformed")
        tensors = np.lib.format.open_memmap(sidecar, mode="r+")
        expected_full_shape = (len(a),) + tensor_shape
        if tensors.dtype != np.dtype(np.float64) or tensors.shape != expected_full_shape:
            _close_memmap(tensors)
            raise ValueError("partial snapshot tensor sidecar shape/dtype is malformed")

    pending = np.flatnonzero(~completed)
    if tensors is None:
        first = int(pending[0])
        first_tensor = np.asarray(tensor_factory(a[first], g[first]), dtype=float)
        if (
            first_tensor.ndim != 3
            or first_tensor.shape[0] != a.shape[1]
            or first_tensor.shape[1] < 1
            or first_tensor.shape[-1] != first_tensor.shape[-2]
            or np.any(~np.isfinite(first_tensor))
        ):
            raise ValueError("tensor_factory returned an incompatible tensor")
        tensor_shape = _validate_tensor_shape(first_tensor.shape, a.shape[1])
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        tensors = np.lib.format.open_memmap(
            sidecar,
            mode="w+",
            dtype=np.float64,
            shape=(len(a),) + tensor_shape,
        )
        tensors[first] = first_tensor
        completed[first] = True
        tensors.flush()
        _atomic_checkpoint(
            partial,
            states=a,
            geometries=g,
            completed=completed,
            tensor_shape=tensor_shape,
            physical_signature=expected_signature,
        )
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
                tensors.flush()
                _atomic_checkpoint(
                    partial,
                    states=a,
                    geometries=g,
                    completed=completed,
                    tensor_shape=expected_shape,
                    physical_signature=expected_signature,
                )
                since_checkpoint = 0
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        tensors.flush()
        _atomic_checkpoint(
            partial,
            states=a,
            geometries=g,
            completed=completed,
            tensor_shape=expected_shape,
            physical_signature=expected_signature,
        )

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
        # A successfully written frozen dataset supersedes the resumable working
        # files. Remove the large duplicate sidecar only after save succeeds.
        _close_memmap(tensors)
        tensors = None
        partial.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    return dataset


__all__ = ["generate_snapshots_resumable"]