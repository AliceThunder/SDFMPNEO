"""Deterministic snapshot datasets for current-quadratic Joule tensors."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from scipy.stats import qmc

from .symmetric import tensor_svec

_DATASET_FORMAT_VERSION = 1
_SPLIT_NAMES = ("train", "validation", "test")


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _dataset_hash(arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_array(arrays[name]).encode("ascii"))
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest.update(encoded.encode("utf-8"))
    return digest.hexdigest()


def frozen_split_indices(
    n_samples: int,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> np.ndarray:
    """Return immutable integer split labels 0=train, 1=validation, 2=test."""
    n = int(n_samples)
    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    if n < 3:
        raise ValueError("at least three samples are required for frozen splits")
    if (
        not 0.0 < validation_fraction < 1.0
        or not 0.0 < test_fraction < 1.0
        or validation_fraction + test_fraction >= 1.0
    ):
        raise ValueError("invalid validation/test fractions")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    n_validation = max(1, int(round(n * validation_fraction)))
    if n_test + n_validation >= n:
        n_validation = 1
        n_test = 1
    split = np.zeros(n, dtype=np.int8)
    split[order[:n_test]] = 2
    split[order[n_test:n_test + n_validation]] = 1
    return split


def latin_hypercube_box(
    lower: np.ndarray,
    upper: np.ndarray,
    n_samples: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Sample a finite box with a scrambled Latin hypercube."""
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if lo.shape != hi.shape or np.any(~np.isfinite(lo + hi)) or np.any(hi <= lo):
        raise ValueError("sampling bounds must be finite and strictly ordered")
    n = int(n_samples)
    if n < 1:
        raise ValueError("n_samples must be positive")
    if lo.size == 0:
        return np.empty((n, 0), dtype=float)
    unit = qmc.LatinHypercube(lo.size, scramble=True, seed=int(seed)).random(n)
    return qmc.scale(unit, lo, hi)


@dataclass(frozen=True)
class SnapshotManifest:
    format_version: int
    n_samples: int
    thermal_rank: int
    geometry_dimension: int
    current_dimension: int
    packed_symmetric_size: int
    train_count: int
    validation_count: int
    test_count: int
    dataset_hash: str
    metadata: dict


@dataclass(frozen=True)
class QuadraticJouleDataset:
    """Offline samples of ``(thermal state, geometry) -> svec(G)``.

    ``outputs`` is flattened as ``(n_samples, thermal_rank * n_sym)``. The
    representation is Frobenius-isometric because it is produced by ``svec``.
    For large datasets, :meth:`load` transparently returns the compatible
    directory-backed ``DiskQuadraticJouleDataset`` instead.
    """

    states: np.ndarray
    geometries: np.ndarray
    outputs: np.ndarray
    split: np.ndarray
    thermal_rank: int
    current_dimension: int
    metadata: dict

    def __post_init__(self) -> None:
        states = np.asarray(self.states, dtype=float)
        geometry = np.asarray(self.geometries, dtype=float)
        outputs = np.asarray(self.outputs, dtype=float)
        split = np.asarray(self.split, dtype=np.int8)
        n = states.shape[0] if states.ndim == 2 else -1
        if n < 1 or geometry.ndim != 2 or outputs.ndim != 2:
            raise ValueError("dataset arrays must be nonempty matrices")
        if geometry.shape[0] != n or outputs.shape[0] != n or split.shape != (n,):
            raise ValueError("dataset sample counts do not match")
        if states.shape[1] != int(self.thermal_rank):
            raise ValueError("state width does not match thermal_rank")
        p = int(self.current_dimension) + 1
        n_sym = p * (p + 1) // 2
        if outputs.shape[1] != int(self.thermal_rank) * n_sym:
            raise ValueError("output width does not match thermal/current dimensions")
        if np.any(~np.isfinite(states)) or np.any(~np.isfinite(geometry)) or np.any(~np.isfinite(outputs)):
            raise ValueError("dataset values must be finite")
        if np.any((split < 0) | (split > 2)):
            raise ValueError("split labels must be 0, 1 or 2")
        if any(np.count_nonzero(split == k) == 0 for k in range(3)):
            raise ValueError("train, validation and test splits must all be nonempty")
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "geometries", geometry)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "thermal_rank", int(self.thermal_rank))
        object.__setattr__(self, "current_dimension", int(self.current_dimension))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_samples(self) -> int:
        return self.states.shape[0]

    @property
    def geometry_dimension(self) -> int:
        return self.geometries.shape[1]

    @property
    def packed_symmetric_size(self) -> int:
        p = self.current_dimension + 1
        return p * (p + 1) // 2

    @property
    def inputs(self) -> np.ndarray:
        return np.hstack([self.states, self.geometries])

    def indices(self, split: str) -> np.ndarray:
        try:
            label = _SPLIT_NAMES.index(str(split))
        except ValueError as exc:
            raise ValueError(f"split must be one of {_SPLIT_NAMES}") from exc
        return np.flatnonzero(self.split == label)

    def arrays_for(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        ids = self.indices(split)
        return self.inputs[ids], self.outputs[ids]

    def manifest(self) -> SnapshotManifest:
        arrays = {
            "states": self.states,
            "geometries": self.geometries,
            "outputs": self.outputs,
            "split": self.split,
        }
        payload = dict(self.metadata)
        digest = _dataset_hash(arrays, payload)
        return SnapshotManifest(
            format_version=_DATASET_FORMAT_VERSION,
            n_samples=self.n_samples,
            thermal_rank=self.thermal_rank,
            geometry_dimension=self.geometry_dimension,
            current_dimension=self.current_dimension,
            packed_symmetric_size=self.packed_symmetric_size,
            train_count=int(np.count_nonzero(self.split == 0)),
            validation_count=int(np.count_nonzero(self.split == 1)),
            test_count=int(np.count_nonzero(self.split == 2)),
            dataset_hash=digest,
            metadata=payload,
        )

    def save(self, path: str | Path) -> tuple[Path, Path]:
        base = Path(path)
        if base.suffix == ".npz":
            npz_path = base
            json_path = base.with_suffix(".json")
        else:
            npz_path = base.with_suffix(".npz")
            json_path = base.with_suffix(".json")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        with npz_path.open("wb") as output:
            np.savez_compressed(
                output,
                states=self.states,
                geometries=self.geometries,
                outputs=self.outputs,
                split=self.split,
                thermal_rank=np.array(self.thermal_rank, dtype=np.int64),
                current_dimension=np.array(self.current_dimension, dtype=np.int64),
            )
        json_path.write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return npz_path, json_path

    @classmethod
    def load(cls, path: str | Path):
        base = Path(path)
        # Delay the import to avoid a module cycle: disk_dataset imports the
        # manifest definition above but can safely be imported after this module
        # has finished initialization.
        if base.is_dir() or base.suffix == ".store":
            from .disk_dataset import DiskQuadraticJouleDataset

            return DiskQuadraticJouleDataset.load(base, verify=True)
        if base.suffix not in {"", ".npz"}:
            raise ValueError("quadratic Joule dataset must be an .npz file or .store directory")
        if base.suffix == "":
            npz_candidate = base.with_suffix(".npz")
            store_candidate = base.with_suffix(".store")
            if npz_candidate.exists():
                base = npz_candidate
            elif store_candidate.exists():
                from .disk_dataset import DiskQuadraticJouleDataset

                return DiskQuadraticJouleDataset.load(store_candidate, verify=True)
        npz_path = base if base.suffix == ".npz" else base.with_suffix(".npz")
        json_path = npz_path.with_suffix(".json")
        manifest = json.loads(json_path.read_text(encoding="utf-8"))
        if int(manifest.get("format_version", -1)) != _DATASET_FORMAT_VERSION:
            raise ValueError("unsupported quadratic Joule dataset format")
        with np.load(npz_path, allow_pickle=False) as data:
            result = cls(
                states=data["states"],
                geometries=data["geometries"],
                outputs=data["outputs"],
                split=data["split"],
                thermal_rank=int(data["thermal_rank"]),
                current_dimension=int(data["current_dimension"]),
                metadata=dict(manifest.get("metadata") or {}),
            )
        if result.manifest().dataset_hash != manifest.get("dataset_hash"):
            raise ValueError("quadratic Joule dataset hash mismatch")
        return result

    @classmethod
    def from_tensors(
        cls,
        states: np.ndarray,
        geometries: np.ndarray,
        tensors: np.ndarray,
        *,
        split_seed: int = 0,
        validation_fraction: float = 0.1,
        test_fraction: float = 0.1,
        metadata: Mapping[str, object] | None = None,
    ) -> "QuadraticJouleDataset":
        a = np.asarray(states, dtype=float)
        g = np.asarray(geometries, dtype=float)
        G = np.asarray(tensors, dtype=float)
        if a.ndim != 2 or g.ndim != 2 or G.ndim != 4:
            raise ValueError("states/geometries/tensors have incompatible ranks")
        if a.shape[0] != g.shape[0] or a.shape[0] != G.shape[0]:
            raise ValueError("snapshot sample counts do not match")
        if G.shape[1] != a.shape[1] or G.shape[-1] != G.shape[-2]:
            raise ValueError("tensor thermal/current dimensions do not match states")
        packed = tensor_svec(G).reshape(len(a), -1)
        split = frozen_split_indices(
            len(a),
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=split_seed,
        )
        return cls(
            states=a,
            geometries=g,
            outputs=packed,
            split=split,
            thermal_rank=a.shape[1],
            current_dimension=G.shape[-1] - 1,
            metadata={} if metadata is None else dict(metadata),
        )


def generate_snapshot_dataset(
    states: np.ndarray,
    geometries: np.ndarray,
    tensor_factory: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    split_seed: int = 0,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    metadata: Mapping[str, object] | None = None,
) -> QuadraticJouleDataset:
    """Generate deterministic physics snapshots with no trajectory integration."""
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or a.shape[0] != g.shape[0] or len(a) < 1:
        raise ValueError("states and geometries must be aligned nonempty matrices")
    tensors = [np.asarray(tensor_factory(ai, gi), dtype=float) for ai, gi in zip(a, g)]
    return QuadraticJouleDataset.from_tensors(
        a,
        g,
        np.stack(tensors),
        split_seed=split_seed,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        metadata=metadata,
    )


__all__ = [
    "QuadraticJouleDataset",
    "SnapshotManifest",
    "frozen_split_indices",
    "generate_snapshot_dataset",
    "latin_hypercube_box",
]
