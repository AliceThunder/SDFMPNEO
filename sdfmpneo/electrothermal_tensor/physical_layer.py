"""Hard current-quadratic physics layers for compressed Joule tensors."""
from __future__ import annotations

import numpy as np

from .pod import TensorPOD
from .symmetric import quadratic_feature


def _pod_modes_numpy(pod: TensorPOD):
    n_sym = pod.packed_symmetric_size
    mean_modes = pod.mean.reshape(pod.thermal_rank, n_sym)
    basis_modes = pod.basis.reshape(pod.thermal_rank, n_sym, pod.rank)
    return mean_modes, basis_modes


def decode_heat_source_numpy(
    coefficients: np.ndarray,
    pod: TensorPOD,
    operating: np.ndarray,
) -> np.ndarray:
    """Apply POD and the exact quadratic-current contraction without decoding ``G``.

    With Frobenius-isometric coordinates, the operation is reordered as

    ``q = mean_G:f(u) + sum_k beta_k (U_k:f(u))``.

    This is algebraically identical to decoding the full packed tensor first,
    but online cost scales with ``thermal_rank * POD_rank`` after contracting the
    small current feature rather than materializing the often much wider
    ``thermal_rank * n_sym`` tensor for every ETD step.
    """
    beta = np.asarray(coefficients, dtype=float)
    if beta.ndim < 1 or beta.shape[-1] != pod.rank or np.any(~np.isfinite(beta)):
        raise ValueError("POD coefficient width mismatch or non-finite coefficients")
    feature = quadratic_feature(operating)
    if feature.ndim != 1 or feature.shape[0] != pod.packed_symmetric_size:
        raise ValueError("decode_heat_source_numpy expects one compatible operating vector")
    mean_modes, basis_modes = _pod_modes_numpy(pod)
    mean_q = mean_modes @ feature
    contracted_basis = np.einsum("rsk,s->rk", basis_modes, feature, optimize=True)
    return mean_q + np.tensordot(beta, contracted_basis, axes=([-1], [1]))


def decode_heat_source_batch_numpy(
    coefficients: np.ndarray,
    pod: TensorPOD,
    operating: np.ndarray,
) -> np.ndarray:
    """Aligned batch version without full packed-tensor reconstruction."""
    beta = np.asarray(coefficients, dtype=float)
    u = np.asarray(operating, dtype=float)
    if beta.ndim != 2 or beta.shape[1] != pod.rank:
        raise ValueError("coefficients must have shape (batch,pod_rank)")
    if u.ndim != 2 or beta.shape[0] != u.shape[0]:
        raise ValueError("coefficients and operating must be aligned matrices")
    if np.any(~np.isfinite(beta)) or np.any(~np.isfinite(u)):
        raise ValueError("coefficients and operating must be finite")
    feature = quadratic_feature(u)
    if feature.shape[1] != pod.packed_symmetric_size:
        raise ValueError("operating dimension does not match POD tensor order")
    mean_modes, basis_modes = _pod_modes_numpy(pod)
    mean_q = np.einsum("rs,bs->br", mean_modes, feature, optimize=True)
    contracted_basis = np.einsum("rsk,bs->brk", basis_modes, feature, optimize=True)
    return mean_q + np.einsum("brk,bk->br", contracted_basis, beta, optimize=True)


def torch_quadratic_feature(operating):
    """Torch equivalent of ``svec(zeta zeta^T)``; imported lazily."""
    import torch

    if operating.ndim == 1:
        operating = operating.unsqueeze(0)
        squeeze = True
    elif operating.ndim == 2:
        squeeze = False
    else:
        raise ValueError("operating must be rank one or two")
    ones = torch.ones((operating.shape[0], 1), dtype=operating.dtype, device=operating.device)
    zeta = torch.cat([ones, operating], dim=-1)
    p = zeta.shape[-1]
    i, j = torch.triu_indices(p, p, device=zeta.device)
    weights = torch.where(
        i == j,
        torch.ones_like(i, dtype=operating.dtype),
        torch.full_like(i, np.sqrt(2.0), dtype=operating.dtype),
    )
    result = (zeta[:, i] * zeta[:, j]) * weights.unsqueeze(0)
    return result[0] if squeeze else result


def decode_heat_source_torch(coefficients, mean, basis, thermal_rank: int, operating):
    """Differentiable direct POD/current contraction without decoding full ``G``."""
    import torch

    beta = coefficients
    if beta.ndim == 1:
        beta = beta.unsqueeze(0)
        u = operating.unsqueeze(0) if operating.ndim == 1 else operating
        squeeze = True
    elif beta.ndim == 2:
        u = operating
        squeeze = False
    else:
        raise ValueError("coefficients must be rank one or two")
    if u.ndim != 2 or u.shape[0] != beta.shape[0]:
        raise ValueError("operating batch does not match coefficients")
    if basis.ndim != 2 or mean.ndim != 1 or basis.shape[0] != mean.shape[0]:
        raise ValueError("POD mean/basis dimensions are incompatible")
    if beta.shape[1] != basis.shape[1]:
        raise ValueError("POD coefficient width mismatch")
    r = int(thermal_rank)
    if r < 1 or mean.shape[0] % r:
        raise ValueError("POD width is incompatible with thermal rank")
    n_sym = mean.shape[0] // r
    feature = torch_quadratic_feature(u)
    if feature.shape[1] != n_sym:
        raise ValueError("operating dimension does not match POD tensor order")
    mean_modes = mean.reshape(r, n_sym)
    basis_modes = basis.reshape(r, n_sym, basis.shape[1])
    mean_q = torch.einsum("rs,bs->br", mean_modes, feature)
    contracted_basis = torch.einsum("rsk,bs->brk", basis_modes, feature)
    result = mean_q + torch.einsum("brk,bk->br", contracted_basis, beta)
    return result[0] if squeeze else result


__all__ = [
    "decode_heat_source_batch_numpy",
    "decode_heat_source_numpy",
    "decode_heat_source_torch",
    "torch_quadratic_feature",
]
