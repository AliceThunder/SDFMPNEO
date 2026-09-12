"""Frobenius-isometric coordinates for real symmetric current tensors."""
from __future__ import annotations

import numpy as np

_SQRT2 = float(np.sqrt(2.0))


def symmetric_packed_size(order: int) -> int:
    order = int(order)
    if order < 1:
        raise ValueError("symmetric matrix order must be positive")
    return order * (order + 1) // 2


def svec(matrix: np.ndarray) -> np.ndarray:
    """Pack symmetric matrices with sqrt(2) off-diagonal scaling.

    For symmetric A and B, ``dot(svec(A), svec(B)) == trace(A.T @ B)``.
    Leading batch dimensions are preserved.
    """
    value = np.asarray(matrix, dtype=float)
    if value.ndim < 2 or value.shape[-1] != value.shape[-2]:
        raise ValueError("matrix must end in a square pair of dimensions")
    if np.any(~np.isfinite(value)):
        raise ValueError("matrix must be finite")
    order = value.shape[-1]
    i, j = np.triu_indices(order)
    out = value[..., i, j].copy()
    out[..., i != j] *= _SQRT2
    return out


def smat(vector: np.ndarray, order: int) -> np.ndarray:
    """Inverse of :func:`svec` for one or many packed matrices."""
    order = int(order)
    value = np.asarray(vector, dtype=float)
    size = symmetric_packed_size(order)
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError("packed symmetric vector has incompatible size")
    if np.any(~np.isfinite(value)):
        raise ValueError("packed symmetric vector must be finite")
    result = np.zeros(value.shape[:-1] + (order, order), dtype=float)
    i, j = np.triu_indices(order)
    entries = value.copy()
    entries[..., i != j] /= _SQRT2
    result[..., i, j] = entries
    result[..., j, i] = entries
    return result


def tensor_svec(tensor: np.ndarray) -> np.ndarray:
    """Pack ``(..., n_thermal, p, p)`` tensors into ``(..., n_thermal, n_sym)``."""
    value = np.asarray(tensor, dtype=float)
    if value.ndim < 3 or value.shape[-1] != value.shape[-2]:
        raise ValueError("tensor must end in (n_thermal,p,p)")
    return svec(value)


def tensor_smat(packed: np.ndarray, order: int) -> np.ndarray:
    """Inverse of :func:`tensor_svec`."""
    value = np.asarray(packed, dtype=float)
    if value.ndim < 2:
        raise ValueError("packed tensor must end in (n_thermal,n_sym)")
    return smat(value, order)


def quadratic_feature(operating: np.ndarray, *, include_offset: bool = True) -> np.ndarray:
    """Return ``svec(zeta zeta^T)`` for one or many real operating vectors."""
    u = np.asarray(operating, dtype=float)
    if u.ndim == 1:
        u = u[None, :]
        squeeze = True
    elif u.ndim == 2:
        squeeze = False
    else:
        raise ValueError("operating must be a vector or matrix")
    if np.any(~np.isfinite(u)):
        raise ValueError("operating must be finite")
    zeta = np.column_stack([np.ones(len(u)), u]) if include_offset else u
    outer = np.einsum("bi,bj->bij", zeta, zeta, optimize=True)
    result = svec(outer)
    return result[0] if squeeze else result


def quadratic_from_svec(packed_tensor: np.ndarray, operating: np.ndarray) -> np.ndarray:
    """Evaluate modal quadratic forms without reconstructing dense matrices."""
    packed = np.asarray(packed_tensor, dtype=float)
    if packed.ndim != 2:
        raise ValueError("packed_tensor must have shape (n_thermal,n_sym)")
    feature = quadratic_feature(operating)
    if feature.shape[-1] != packed.shape[-1]:
        raise ValueError("operating dimension does not match packed tensor")
    return np.asarray(packed @ feature, dtype=float)


__all__ = [
    "quadratic_feature",
    "quadratic_from_svec",
    "smat",
    "svec",
    "symmetric_packed_size",
    "tensor_smat",
    "tensor_svec",
]
