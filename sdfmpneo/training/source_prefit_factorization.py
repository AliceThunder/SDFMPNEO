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
    keep = min(
        rank,
        int(np.count_nonzero(s > np.finfo(float).eps * max(float(s[0]), 1.0))),
    )
    if keep <= 0:
        return (
            np.zeros((matrix.shape[0], 0), dtype=float),
            np.zeros((matrix.shape[1], 0), dtype=float),
        )
    return u[:, :keep], (s[:keep, None] * vt[:keep]).T


def _cp_greedy(tensor, rank, *, seed=913, iterations=16):
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
        # Deterministic high-energy initialization from a matrix unfolding makes
        # simple rank-one polynomial interactions (notably u^2) exact instead of
        # depending on a random power-iteration start.  Random fallback handles a
        # numerically empty leading singular vector defensively.
        unfold = residual.reshape(d0, d1 * d2)
        try:
            left, singular, right = np.linalg.svd(unfold, full_matrices=False)
            a = left[:, 0]
            tail = (singular[0] * right[0]).reshape(d1, d2)
            ub, sb, vtb = np.linalg.svd(tail, full_matrices=False)
            b = ub[:, 0]
            c = vtb[0]
        except np.linalg.LinAlgError:
            a = rng.normal(size=d0)
            b = rng.normal(size=d1)
            c = rng.normal(size=d2)
        for vector in (a, b, c):
            norm = float(np.linalg.norm(vector))
            if norm <= tiny:
                vector[:] = rng.normal(size=vector.shape)
                norm = float(np.linalg.norm(vector))
            vector /= max(norm, tiny)
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
        sigma = float(
            np.einsum("ijk,i,j,k->", residual, a, b, c, optimize=True)
        )
        if (
            not np.isfinite(sigma)
            or abs(sigma) <= 1e-14 * max(1.0, float(np.linalg.norm(residual)))
        ):
            break
        output = (sigma * c).copy()
        components.append((a.copy(), b.copy(), output))
        residual -= np.einsum("i,j,k->ijk", a, b, output, optimize=True)
    return tuple(components)


def _identifiable_source_regression(normalized, desired, n_thermal):
    """Fit the unique polynomial families represented by the first source bank.

    The historical sequential regression used ``x_i * 1`` inside the quadratic
    bank even though those columns are identical to linear features.  On finite
    collocation sets this lets linear, square and quadratic regressions absorb one
    another and makes the subsequent CP tensor non-identifiable.  We instead fit
    one non-redundant polynomial design:

      1, x_i, x_thermal_i^2, x_thermal_i*u_j, u_i*u_j (i<=j).

    It spans exactly the intended first-layer functional families while assigning
    every monomial to one block.  The returned rectangular interaction tensor uses
    a symmetric split for off-diagonal operating products so its contraction with
    ``normalized x [1,u]`` reproduces the fitted polynomial exactly.
    """
    z = np.asarray(normalized, dtype=float)
    y = np.asarray(desired, dtype=float)
    n = int(n_thermal)
    m = z.shape[1] - n
    columns = [np.ones(len(z), dtype=float)]
    columns.extend(z[:, i] for i in range(z.shape[1]))
    columns.extend(z[:, i] * z[:, i] for i in range(n))

    interaction_descriptors = []
    for i in range(n):
        for j in range(m):
            columns.append(z[:, i] * z[:, n + j])
            interaction_descriptors.append(("thermal_operating", i, j))
    for i in range(m):
        for j in range(i, m):
            columns.append(z[:, n + i] * z[:, n + j])
            interaction_descriptors.append(("operating_operating", i, j))

    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    cursor = 0
    bias = coefficients[cursor]
    cursor += 1
    linear_map = coefficients[cursor:cursor + z.shape[1]]
    cursor += z.shape[1]
    square_map = coefficients[cursor:cursor + n]
    cursor += n

    tensor = np.zeros((z.shape[1], 1 + m, y.shape[1]), dtype=float)
    for descriptor in interaction_descriptors:
        coeff = coefficients[cursor]
        cursor += 1
        kind, i, j = descriptor
        if kind == "thermal_operating":
            tensor[i, 1 + j] += coeff
        elif i == j:
            tensor[n + i, 1 + j] += coeff
        else:
            # u_i*u_j appears twice in the rectangular factorization.  A 1/2-1/2
            # symmetric split is unique, preserves the polynomial exactly, and
            # avoids arbitrary least-squares choices between duplicate columns.
            tensor[n + i, 1 + j] += 0.5 * coeff
            tensor[n + j, 1 + i] += 0.5 * coeff
    return bias, linear_map, square_map, tensor


def fit_source_factors(network, samples, desired_source):
    """Learn identifiable low-rank source directions, then refit all amplitudes.

    One joint polynomial regression first separates functional families exactly;
    low-rank SVD/CP compression is then applied independently to those families.
    This removes ordering/correlation bias from the old sequential residual fits.
    """
    samples = np.asarray(samples, dtype=float)
    desired = np.asarray(desired_source, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != network.input_dimension:
        raise ValueError("source samples do not match network input dimension")
    if desired.shape != (len(samples), network.n_modes):
        raise ValueError("source targets do not match thermal rank")

    normalized = (
        samples - network.input_center[None, :]
    ) / network.input_scale[None, :]
    n = network.n_modes
    bias, linear_map, square_map, interaction_tensor = (
        _identifiable_source_regression(normalized, desired, n)
    )

    theta = network.parameters.copy()
    theta[network._indices("bias_0")] = bias[network.layer_targets[0]]

    lin_in, lin_out = _truncated_matrix_factors(
        linear_map, network.linear_rank
    )
    theta[network._indices("input_linear_in")] = 0.0
    theta[network._indices("input_linear_out")] = 0.0
    if lin_in.shape[1]:
        theta[network._indices("input_linear_in")[:lin_in.shape[1]]] = lin_in.T
        theta[network._indices("input_linear_out")[:, :lin_out.shape[1]]] = (
            lin_out[network.layer_targets[0]]
        )

    sq_in, sq_out = _truncated_matrix_factors(
        square_map, network.square_rank
    )
    theta[network._indices("square_in")] = 0.0
    theta[network._indices("square_out")] = 0.0
    if sq_in.shape[1]:
        theta[network._indices("square_in")[:sq_in.shape[1]]] = sq_in.T
        theta[network._indices("square_out")[:, :sq_out.shape[1]]] = (
            sq_out[network.layer_targets[0]]
        )

    theta[network._indices("quadratic_u")] = 0.0
    theta[network._indices("quadratic_v")] = 0.0
    theta[network._indices("quadratic_out")] = 0.0
    components = _cp_greedy(interaction_tensor, network.quadratic_rank)
    for q, (left, right, output) in enumerate(components):
        theta[network._indices("quadratic_u")[q]] = left
        theta[network._indices("quadratic_v")[q]] = right
        theta[network._indices("quadratic_out")[:, q]] = output[
            network.layer_targets[0]
        ]

    factored = network.with_parameters(theta)
    # The directions are now identifiable, but rank truncation can still couple
    # their optimal output amplitudes.  One final joint LS gives the best source
    # fit for the retained directions without changing any feature direction.
    return factored.fit_first_layer_source(samples, desired)


__all__ = ["fit_source_factors"]
