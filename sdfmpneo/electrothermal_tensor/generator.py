"""Resumable, embarrassingly-parallel physics snapshot generation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from .dataset import QuadraticJouleDataset, frozen_split_indices
from .disk_dataset import DiskQuadraticJouleDataset
from .symmetric import tensor_svec

_CHECKPOINT_FORMAT_VERSION = 3
_DEFAULT_DISK_BACKED_THRESHOLD_BYTES = 256 << 20


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".packed.npy")


def _atomic_checkpoint(
    path: Path,
    *,
    states: np.ndarray,
    geometries: np.ndarray,
    completed: np.ndarray,
    tensor_shape,
    output_width: int,
    physical_signature: str | None,
) -> None:
    """Atomically update only small checkpoint metadata/bitmap."""
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
            output_width=np.array(int(output_width), dtype=np.int64),
            completed=np.asarray(completed, dtype=bool),
            physical_signature=np.array("" if physical_signature is None else str(physical_signature)),
        )
    temporary.replace(path)


def _validate_checkpoint_samples(data, states: np.ndarray, geometries: np.ndarray) -> None:
    version = int(data["format_version"]) if "format_version" in data.files else -1
    if version != _CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "legacy snapshot checkpoint is not provenance-safe for packed out-of-core labels; "
            "delete it and regenerate"
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


def _final_paths(final_path: str | Path):
    base = Path(final_path)
    if base.suffix == ".npz":
        return base, base.with_suffix(".store")
    if base.suffix == ".store":
        return base.with_suffix(".npz"), base
    return base.with_suffix(".npz"), base.with_suffix(".store")


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
    disk_backed_threshold_bytes: int = _DEFAULT_DISK_BACKED_THRESHOLD_BYTES,
):
    """Generate ``svec(G(a,g))`` snapshots with bounded RAM and checkpoint I/O.

    Each physics tensor is packed immediately and written once to a memmapped NPY
    sidecar. Resume is bound to the exact sample arrays and physical signature.
    On completion, small datasets freeze to the compact NPZ backend; datasets
    whose packed output matrix exceeds ``disk_backed_threshold_bytes`` are frozen
    as a directory-backed read-only memmap store.
    """
    a = np.asarray(states, dtype=np.float64)
    g = np.asarray(geometries, dtype=np.float64)
    if a.ndim != 2 or g.ndim != 2 or len(a) != len(g) or len(a) < 1:
        raise ValueError("states/geometries must be aligned nonempty matrices")
    checkpoint_every = int(checkpoint_every)
    max_workers = int(max_workers)
    threshold = int(disk_backed_threshold_bytes)
    if checkpoint_every < 1 or max_workers < 1 or threshold < 0:
        raise ValueError("checkpoint_every/max_workers must be positive and threshold non-negative")
    partial = Path(checkpoint_path)
    sidecar = _sidecar_path(partial)
    expected_signature = _physical_signature(metadata)

    packed = None
    completed = np.zeros(len(a), dtype=bool)
    tensor_shape = None
    output_width = None
    if partial.exists():
        if not sidecar.exists():
            raise ValueError("snapshot checkpoint packed-output sidecar is missing")
        with np.load(partial, allow_pickle=False) as data:
            _validate_checkpoint_samples(data, a, g)
            saved_signature = str(data["physical_signature"])
            if saved_signature != ("" if expected_signature is None else expected_signature):
                raise ValueError("partial snapshot checkpoint physical signature differs")
            tensor_shape = tuple(np.asarray(data["tensor_shape"], dtype=int).tolist())
            output_width = int(data["output_width"])
            completed = np.asarray(data["completed"], dtype=bool)
        if completed.shape != (len(a),):
            raise ValueError("partial snapshot checkpoint is malformed")
        packed = np.load(sidecar, mmap_mode="r+", allow_pickle=False)
        if packed.shape != (len(a), output_width) or packed.dtype != np.float64:
            raise ValueError("partial snapshot packed-output sidecar shape/dtype mismatch")

    pending = np.flatnonzero(~completed)
    if packed is None:
        first = int(pending[0])
        first_tensor = np.asarray(tensor_factory(a[first], g[first]), dtype=np.float64)
        if (
            first_tensor.ndim != 3
            or first_tensor.shape[0] != a.shape[1]
            or first_tensor.shape[-1] != first_tensor.shape[-2]
            or np.any(~np.isfinite(first_tensor))
        ):
            raise ValueError("tensor_factory returned an incompatible tensor")
        tensor_shape = first_tensor.shape
        first_packed = tensor_svec(first_tensor).reshape(-1)
        output_width = int(first_packed.size)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        packed = np.lib.format.open_memmap(
            sidecar,
            mode="w+",
            dtype=np.float64,
            shape=(len(a), output_width),
        )
        packed[first] = first_packed
        completed[first] = True
        packed.flush()
        _atomic_checkpoint(
            partial,
            states=a,
            geometries=g,
            completed=completed,
            tensor_shape=tensor_shape,
            output_width=output_width,
            physical_signature=expected_signature,
        )
        pending = np.flatnonzero(~completed)

    assert tensor_shape is not None and output_width is not None

    def evaluate(index: int):
        value = np.asarray(tensor_factory(a[index], g[index]), dtype=np.float64)
        if value.shape != tensor_shape or np.any(~np.isfinite(value)):
            raise ValueError("tensor_factory changed shape or returned non-finite values")
        row = tensor_svec(value).reshape(-1)
        if row.shape != (output_width,) or np.any(~np.isfinite(row)):
            raise ValueError("packed quadratic tensor row is invalid")
        return index, row

    since_checkpoint = 0
    if max_workers == 1:
        iterator = map(evaluate, (int(i) for i in pending))
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        iterator = executor.map(evaluate, (int(i) for i in pending))
    try:
        for index, value in iterator:
            packed[index] = value
            completed[index] = True
            since_checkpoint += 1
            if since_checkpoint >= checkpoint_every:
                packed.flush()
                _atomic_checkpoint(
                    partial,
                    states=a,
                    geometries=g,
                    completed=completed,
                    tensor_shape=tensor_shape,
                    output_width=output_width,
                    physical_signature=expected_signature,
                )
                since_checkpoint = 0
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        packed.flush()
        _atomic_checkpoint(
            partial,
            states=a,
            geometries=g,
            completed=completed,
            tensor_shape=tensor_shape,
            output_width=output_width,
            physical_signature=expected_signature,
        )

    if not np.all(completed):
        raise RuntimeError("snapshot generation did not complete")
    current_dimension = int(tensor_shape[-1] - 1)
    split = frozen_split_indices(
        len(a),
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=split_seed,
    )
    output_bytes = int(len(a)) * int(output_width) * np.dtype(np.float64).itemsize
    use_disk = output_bytes > threshold

    if use_disk:
        if final_path is None:
            raise ValueError("large out-of-core snapshot datasets require final_path")
        _, store_path = _final_paths(final_path)
        _close_memmap(packed)
        packed = None
        dataset = DiskQuadraticJouleDataset.create(
            store_path,
            states=a,
            geometries=g,
            outputs=None,
            split=split,
            thermal_rank=a.shape[1],
            current_dimension=current_dimension,
            metadata={} if metadata is None else dict(metadata),
            move_outputs_file=sidecar,
        )
        partial.unlink(missing_ok=True)
        return dataset

    outputs = np.asarray(packed, dtype=np.float64).copy()
    dataset = QuadraticJouleDataset(
        states=a,
        geometries=g,
        outputs=outputs,
        split=split,
        thermal_rank=a.shape[1],
        current_dimension=current_dimension,
        metadata={} if metadata is None else dict(metadata),
    )
    if final_path is not None:
        npz_path, _ = _final_paths(final_path)
        dataset.save(npz_path)
        _close_memmap(packed)
        packed = None
        partial.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
    return dataset


__all__ = ["generate_snapshots_resumable"]
