"""Hard current-quadratic physics layers for compressed Joule tensors."""
from __future__ import annotations

import numpy as np

from .pod import TensorPOD
from .symmetric import quadratic_feature


def decode_heat_source_numpy(
    coefficients: np.ndarray,
    pod: TensorPOD,
    operating: np.ndarray,
) -> np.ndarray:
    """Decode POD coefficients and apply the exact quadratic-current contraction.

    The dense ``(p,p)`` matrices are never reconstructed.  Because the dataset
    uses the Frobenius-isometric svec coordinates,

        zeta^T G_j zeta = svec(G_j)^T svec(zeta zeta^T).
    """
    beta = np.asarray(coefficients, dtype=float)
    packed = pod.decode(beta)
    n_sym = pod.packed_symmetric_size
    modes = packed.reshape(packed.shape[:-1] + (pod.thermal_rank, n_sym))
    feature = quadratic_feature(operating)
    if feature.ndim != 1:
        raise ValueError("decode_heat_source_numpy expects one operating vector")
    return np.einsum("...rs,s->...r", modes, feature, optimize=True)


def decode_heat_source_batch_numpy(
    coefficients: np.ndarray,
    pod: TensorPOD,
    operating: np.ndarray,
) -> np.ndarray:
    """Batch version with aligned leading sample dimension."""
    beta = np.asarray(coefficients, dtype=float)
    u = np.asarray(operating, dtype=float)
    if beta.ndim != 2 or u.ndim != 2 or beta.shape[0] != u.shape[0]:
        raise ValueError("coefficients and operating must be aligned matrices")
    packed = pod.decode(beta).reshape(len(beta), pod.thermal_rank, pod.packed_symmetric_size)
    feature = quadratic_feature(u)
    return np.einsum("brs,bs->br", packed, feature, optimize=True)


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
    result = zeta[:, i] * zeta[:, j]
    off = i != j
    if bool(torch.any(off)):
        result[:, off] = result[:, off] * np.sqrt(2.0)
    return result[0] if squeeze else result


def decode_heat_source_torch(coefficients, mean, basis, thermal_rank: int, operating):
    """Differentiable POD decode + exact current-quadratic contraction."""
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
    packed = mean.unsqueeze(0) + beta @ basis.T
    n_sym = packed.shape[-1] // int(thermal_rank)
    if n_sym * int(thermal_rank) != packed.shape[-1]:
        raise ValueError("POD width is incompatible with thermal rank")
    modes = packed.reshape(beta.shape[0], int(thermal_rank), n_sym)
    feature = torch_quadratic_feature(u)
    result = torch.einsum("brs,bs->br", modes, feature)
    return result[0] if squeeze else result


__all__ = [
    "decode_heat_source_batch_numpy",
    "decode_heat_source_numpy",
    "decode_heat_source_torch",
    "torch_quadratic_feature",
]
