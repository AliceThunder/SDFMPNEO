"""Unified loading for in-memory and directory-backed quadratic Joule datasets."""
from __future__ import annotations

from pathlib import Path

from .dataset import QuadraticJouleDataset
from .disk_dataset import DiskQuadraticJouleDataset


def load_quadratic_joule_dataset(path: str | Path, *, verify: bool = True):
    target = Path(path).expanduser()
    if target.is_dir():
        return DiskQuadraticJouleDataset.load(target, verify=verify)
    if target.suffix == ".store":
        if not target.exists():
            raise FileNotFoundError(target)
        return DiskQuadraticJouleDataset.load(target, verify=verify)
    if target.suffix == ".npz":
        return QuadraticJouleDataset.load(target)
    if target.exists() and target.is_file():
        return QuadraticJouleDataset.load(target)
    npz = target.with_suffix(".npz")
    store = target.with_suffix(".store")
    if npz.exists():
        return QuadraticJouleDataset.load(npz)
    if store.exists():
        return DiskQuadraticJouleDataset.load(store, verify=verify)
    raise FileNotFoundError(target)


__all__ = ["load_quadratic_joule_dataset"]
