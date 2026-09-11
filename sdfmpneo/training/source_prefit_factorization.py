"""Data-driven low-rank factorization for the first analytic-response source bank."""
from __future__ import annotations

import numpy as np


def _truncated_matrix_factors(matrix, rank):
    matrix = np.asarray(matrix, dtype=float)
    rank = min(int(rank), min(matrix.shape, default=0))
    if rank <= 0 or not np.any(matrix):
        return (
            np.zeros((matrix.shape[0], 0), dtype=float),
            np.zeros((matrix.shape[1], 0), dtype=float),
        )
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    keep = min(rank, int(np.count_nonzero(s > np.finfo(float).eps * max(float(s[0]), 1.0))))
    if keep <= 0:
        return (
            np.zeros((matrix.shape[0], 0), dtype=float),
            np.zeros((matrix.shape[1], 0), dtype=float),
        )
    return u[:, :keep], (s[:keep, None] * vt[:keep]).T


def _cp_greedy(tensor, rank, *, seed=913, iterations=8):
    """Greedy rank-one CP approximation of a modest dense 3-way tensor."""
    residual = np.asarray(tensor, dtype=float).copy()
    if residual.ndim != 3:
        raise ValueError("CP source tensor must be three-dimensional")
    d0, d1, d2 = residual.shape
    rank = max(0, int(rank))
    rng = np.random.default_rng(seed)
    components = []
    tiny = np.finfo(float).tiny
    for _ in range(rank):
        if not np.any(residual):
            break
        a = rng.normal(size=d0); a /= max(float(np.linalg.norm(a)), tiny)
        b = rng.normal(size=d1); b /= max(float(np.linalg.norm(b)), tiny)
        c = rng.normal(size=d2); c /= max(float(np.linalg.norm(c)), tiny)
        for _ in range(int(iterations)):
            a = np.einsum("ijk,j,k->i", residual, b, c, optimize=True)
            na = float(np.linalg.norm(a))
            if na <= tiny:
                break
            a /= na
            b = np.einsum("ijk,i,k->j", residual, a, c, optimize=True)
            nb = float(np.linalg.norm(b))
            if nb <= tiny:
                break
            b /= nb
            c = np.einsum("ijk,i,j->k", residual, a, b, optimize=True)
            nc = float(np.linalg.norm(c))
            if nc <= tiny:
                break
            c /= nc
        sigma = float(np.einsum("ijk,i,j,k->", residual, a, b, c, optimize=True))
        if not np.isfinite(sigma) or abs(sigma) <= 1e-14 * max(1.0, float(np.linalg.norm(residual))):
            break
        components.append((a.copy(), b.copy(), (sigma * c).copy()))
        residual -= np.einsum("i,j,k->ijk", a, b, sigma * c, optimize=True)
    return tuple(components)


def fit_source_factors(network, samples, desired_source):
    """Learn first-layer low-rank source directions, then refit all amplitudes.

    The factorization mirrors the analytic source bank exactly:

    * affine/linear source: truncated SVD of the full linear regression map;
    * modewise-square source: truncated SVD in squared thermal coordinates;
    * thermal/static bilinear source: greedy CP factorization of the fitted
      ``input x static x output`` coefficient tensor.

    The final call to ``fit_first_layer_source`` solves all output amplitudes
    jointly with the learned factor directions fixed, removing ordering bias
    from the sequential factor estimates.
    """
    samples = np.asarray(samples, dtype=float)
    desired = np.asarray(desired_source, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != network.input_dimension:
        raise ValueError("source samples do not match network input dimension")
    if desired.shape != (len(samples), network.n_modes):
        raise ValueError("source targets do not match thermal rank")
    normalized = (samples - network.input_center[None, :]) / network.input_scale[None, :]
    n = network.n_modes
    static = np.column_stack([
        np.ones(len(samples), dtype=float),
        normalized[:, n:],
    ])

    theta = network.parameters.copy()
    # Fit an affine source first.  The low-rank truncation learns physically
    # relevant input directions instead of keeping random projections forever.
    design = np.column_stack([np.ones(len(samples), dtype=float), normalized])
    affine, *_ = np.linalg.lstsq(design, desired, rcond=None)
    bias = affine[0]
    linear_map = affine[1:]
    lin_in, lin_out = _truncated_matrix_factors(linear_map, network.linear_rank)
    theta[network._indices("input_linear_in")] = 0.0
    theta[network._indices("input_linear_out")] = 0.0
    if lin_in.shape[1]:
        theta[network._indices("input_linear_in")[:lin_in.shape[1]]] = lin_in.T
        theta[network._indices("input_linear_out")[:, :lin_out.shape[1]]] = lin_out
    theta[network._indices("bias_0")] = bias[network.layer_targets[0]]
    linear_prediction = bias[None, :] + normalized @ (lin_in @ lin_out.T if lin_in.shape[1] else np.zeros_like(linear_map))
    residual = desired - linear_prediction

    # Modewise thermal squares are a separate physically meaningful source bank.
    squared = normalized[:, :n] ** 2
    square_map, *_ = np.linalg.lstsq(squared, residual, rcond=None)
    sq_in, sq_out = _truncated_matrix_factors(square_map, network.square_rank)
    theta[network._indices("square_in")] = 0.0
    theta[network._indices("square_out")] = 0.0
    if sq_in.shape[1]:
        theta[network._indices("square_in")[:sq_in.shape[1]]] = sq_in.T
        theta[network._indices("square_out")[:, :sq_out.shape[1]]] = sq_out
        residual = residual - squared @ (sq_in @ sq_out.T)

    # Remaining thermal/static and static/static interactions are represented by
    # the existing CP-like quadratic bank.
    if network.quadratic_rank > 0 and normalized.shape[1] and static.shape[1]:
        bilinear = np.einsum("ni,nj->nij", normalized, static, optimize=True).reshape(len(samples), -1)
        tensor_map, *_ = np.linalg.lstsq(bilinear, residual, rcond=None)
        tensor = tensor_map.reshape(normalized.shape[1], static.shape[1], network.n_modes)
        components = _cp_greedy(tensor, network.quadratic_rank)
        theta[network._indices("quadratic_u")] = 0.0
        theta[network._indices("quadratic_v")] = 0.0
        theta[network._indices("quadratic_out")] = 0.0
        for q, (left, right, output) in enumerate(components):
            theta[network._indices("quadratic_u")[q]] = left
            theta[network._indices("quadratic_v")[q]] = right
            theta[network._indices("quadratic_out")[:, q]] = output[network.layer_targets[0]]

    factored = network.with_parameters(theta)
    # Joint amplitude LS is cheap and restores the best fit for the learned
    # feature directions after the sequential factor estimates above.
    return factored.fit_first_layer_source(samples, desired)


__all__ = ["fit_source_factors"]
