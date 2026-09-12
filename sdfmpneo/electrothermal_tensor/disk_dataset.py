"""Out-of-core frozen datasets for wide quadratic Joule tensors.

The in-memory ``QuadraticJouleDataset`` remains the convenient backend for small
problems.  This module provides the same public data interface while keeping the
wide packed tensor matrix in a read-only NPY memmap.  A directory store contains
four NPY arrays plus a JSON manifest with per-file SHA-256 digests and a canonical
dataset hash.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from typing import Mapping

import numpy as np

from .dataset import SnapshotManifest, frozen_split_indices

_DISK_DATASET_FORMAT_VERSION = 2


def _sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(int(chunk_bytes))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _canonical_dataset_hash(file_hashes: Mapping[str, str], metadata: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    payload = {
        "format_version": _DISK_DATASET_FORMAT_VERSION,
        "files": {str(key): str(value) for key, value in sorted(file_hashes.items())},
        "metadata": dict(metadata),
    }
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )
    return digest.hexdigest()


def _write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)
    temporary.replace(path)


class DiskQuadraticJouleDataset:
    """Frozen directory-backed ``(a,g)->svec(G)`` dataset.

    ``states``, ``geometries``, ``outputs`` and ``split`` are loaded with
    ``mmap_mode='r'``.  The class intentionally mirrors the subset of
    ``QuadraticJouleDataset`` used by POD, training, validation and certification.
    """

    def __init__(self, directory: str | Path, manifest: SnapshotManifest, *, verify: bool = True):
        self.directory = Path(directory)
        self._manifest = manifest
        self.metadata = dict(manifest.metadata)
        self.thermal_rank = int(manifest.thermal_rank)
        self.current_dimension = int(manifest.current_dimension)
        self.states = np.load(self.directory / "states.npy", mmap_mode="r", allow_pickle=False)
        self.geometries = np.load(self.directory / "geometries.npy", mmap_mode="r", allow_pickle=False)
        self.outputs = np.load(self.directory / "outputs.npy", mmap_mode="r", allow_pickle=False)
        self.split = np.load(self.directory / "split.npy", mmap_mode="r", allow_pickle=False)
        self._validate_shapes()
        if verify:
            self.verify_integrity()

    def _validate_shapes(self) -> None:
        n = int(self._manifest.n_samples)
        if self.states.shape != (n, self.thermal_rank):
            raise ValueError("disk dataset state shape disagrees with manifest")
        if self.geometries.shape != (n, int(self._manifest.geometry_dimension)):
            raise ValueError("disk dataset geometry shape disagrees with manifest")
        width = self.thermal_rank * int(self._manifest.packed_symmetric_size)
        if self.outputs.shape != (n, width):
            raise ValueError("disk dataset output shape disagrees with manifest")
        if self.split.shape != (n,):
            raise ValueError("disk dataset split shape disagrees with manifest")
        if self.states.dtype != np.float64 or self.geometries.dtype != np.float64 or self.outputs.dtype != np.float64:
            raise ValueError("disk dataset floating arrays must use float64")
        if self.split.dtype != np.int8:
            raise ValueError("disk dataset split array must use int8")
        if any(np.count_nonzero(self.split == k) == 0 for k in range(3)):
            raise ValueError("disk dataset train/validation/test splits must all be nonempty")

    @property
    def n_samples(self) -> int:
        return int(self._manifest.n_samples)

    @property
    def geometry_dimension(self) -> int:
        return int(self._manifest.geometry_dimension)

    @property
    def packed_symmetric_size(self) -> int:
        return int(self._manifest.packed_symmetric_size)

    @property
    def inputs(self) -> np.ndarray:
        # State/geometry inputs are narrow compared with G and are safe to form
        # in memory for ordinary network training.
        return np.hstack([np.asarray(self.states), np.asarray(self.geometries)])

    def indices(self, split: str) -> np.ndarray:
        names = ("train", "validation", "test")
        try:
            label = names.index(str(split))
        except ValueError as exc:
            raise ValueError(f"split must be one of {names}") from exc
        return np.flatnonzero(np.asarray(self.split) == label)

    def arrays_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self.indices(split)
        return self.inputs[ids], np.asarray(self.outputs[ids])

    def manifest(self) -> SnapshotManifest:
        return self._manifest

    def verify_integrity(self) -> None:
        manifest_path = self.directory / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = payload.get("file_hashes")
        if not isinstance(files, dict):
            raise ValueError("disk dataset manifest lacks file hashes")
        actual = {
            name: _sha256_file(self.directory / name)
            for name in ("states.npy", "geometries.npy", "outputs.npy", "split.npy")
        }
        if actual != {str(k): str(v) for k, v in files.items()}:
            raise ValueError("disk quadratic Joule dataset file hash mismatch")
        expected = _canonical_dataset_hash(actual, dict(payload.get("metadata") or {}))
        if expected != str(payload.get("dataset_hash")):
            raise ValueError("disk quadratic Joule dataset hash mismatch")
        if expected != self._manifest.dataset_hash:
            raise ValueError("disk dataset in-memory manifest hash mismatch")

    @classmethod
    def load(cls, directory: str | Path, *, verify: bool = True) -> "DiskQuadraticJouleDataset":
        root = Path(directory)
        payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if int(payload.get("format_version", -1)) != _DISK_DATASET_FORMAT_VERSION:
            raise ValueError("unsupported disk quadratic Joule dataset format")
        manifest = SnapshotManifest(
            format_version=int(payload["format_version"]),
            n_samples=int(payload["n_samples"]),
            thermal_rank=int(payload["thermal_rank"]),
            geometry_dimension=int(payload["geometry_dimension"]),
            current_dimension=int(payload["current_dimension"]),
            packed_symmetric_size=int(payload["packed_symmetric_size"]),
            train_count=int(payload["train_count"]),
            validation_count=int(payload["validation_count"]),
            test_count=int(payload["test_count"]),
            dataset_hash=str(payload["dataset_hash"]),
            metadata=dict(payload.get("metadata") or {}),
        )
        return cls(root, manifest, verify=verify)

    @classmethod
    def create(
        cls,
        directory: str | Path,
        *,
        states: np.ndarray,
        geometries: np.ndarray,
        outputs: np.ndarray,
        split: np.ndarray | None = None,
        split_seed: int = 0,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        thermal_rank: int,
        current_dimension: int,
        metadata: Mapping[str, object] | None = None,
        move_outputs_file: str | Path | None = None,
    ) -> "DiskQuadraticJouleDataset":
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        a = np.asarray(states, dtype=np.float64)
        g = np.asarray(geometries, dtype=np.float64)
        y = np.asarray(outputs)
        n = len(a)
        p = int(current_dimension) + 1
        n_sym = p * (p + 1) // 2
        width = int(thermal_rank) * n_sym
        if a.shape != (n, int(thermal_rank)) or g.ndim != 2 or g.shape[0] != n:
            raise ValueError("disk dataset states/geometries are incompatible")
        if y.shape != (n, width):
            raise ValueError("disk dataset packed output shape mismatch")
        labels = (
            frozen_split_indices(
                n,
                validation_fraction=validation_fraction,
                test_fraction=test_fraction,
                seed=split_seed,
            )
            if split is None
            else np.asarray(split, dtype=np.int8)
        )
        if labels.shape != (n,):
            raise ValueError("disk dataset split shape mismatch")

        _write_npy(root / "states.npy", a)
        _write_npy(root / "geometries.npy", g)
        _write_npy(root / "split.npy", labels.astype(np.int8, copy=False))
        output_path = root / "outputs.npy"
        if move_outputs_file is not None:
            source = Path(move_outputs_file)
            # Verify the NPY header before moving the potentially huge file.
            probe = np.load(source, mmap_mode="r", allow_pickle=False)
            if probe.shape != (n, width) or probe.dtype != np.float64:
                raise ValueError("working packed-output file has incompatible shape/dtype")
            del probe
            shutil.move(str(source), str(output_path))
        else:
            # ``np.save`` can stream a memmap/ndarray without creating another
            # centered or packed copy in memory.
            with output_path.open("wb") as handle:
                np.save(handle, y.astype(np.float64, copy=False), allow_pickle=False)

        metadata_dict = {} if metadata is None else dict(metadata)
        file_hashes = {
            name: _sha256_file(root / name)
            for name in ("states.npy", "geometries.npy", "outputs.npy", "split.npy")
        }
        dataset_hash = _canonical_dataset_hash(file_hashes, metadata_dict)
        manifest = SnapshotManifest(
            format_version=_DISK_DATASET_FORMAT_VERSION,
            n_samples=n,
            thermal_rank=int(thermal_rank),
            geometry_dimension=g.shape[1],
            current_dimension=int(current_dimension),
            packed_symmetric_size=n_sym,
            train_count=int(np.count_nonzero(labels == 0)),
            validation_count=int(np.count_nonzero(labels == 1)),
            test_count=int(np.count_nonzero(labels == 2)),
            dataset_hash=dataset_hash,
            metadata=metadata_dict,
        )
        payload = asdict(manifest)
        payload["file_hashes"] = file_hashes
        (root / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return cls(root, manifest, verify=False)


__all__ = ["DiskQuadraticJouleDataset"]
