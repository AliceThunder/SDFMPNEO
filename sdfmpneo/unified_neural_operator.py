"""Operator-aware polynomial neural smoother for the full-space Maxwell solve.

The neural model does not predict a Maxwell field.  It reads exact residual and
operator-action features, then predicts a few bounded complex coefficients for
a short Jacobi-preconditioned operator polynomial.  FGMRES consumes the
resulting direction and still accepts a solution only from the true sparse
Maxwell residual.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


FEATURE_SCHEMA = "edge-residual-v4-polynomial-smoother"


@dataclass(frozen=True)
class EdgeMultiscaleConfig:
    width: int = 32
    levels: int = 3
    blocks_per_level: int = 1
    activation: str = "silu"
    polynomial_order: int = 3
    coefficient_limit: float = 2.0

    def __post_init__(self):
        if self.width < 4 or self.levels < 1 or self.blocks_per_level < 1:
            raise ValueError("invalid edge multiscale network dimensions")
        if self.activation not in {"silu", "gelu", "tanh"}:
            raise ValueError("activation must be silu, gelu, or tanh")
        if not 1 <= int(self.polynomial_order) <= 6:
            raise ValueError("polynomial_order must lie in [1, 6]")
        if not np.isfinite(self.coefficient_limit) or self.coefficient_limit <= 0:
            raise ValueError("coefficient_limit must be finite and positive")

    def to_dict(self):
        return asdict(self)


def _edge_centers(background):
    centers = np.empty((background.n_edges, 3), dtype=float)
    axes = np.empty(background.n_edges, dtype=np.int64)
    for e, (axis, i, j, k) in enumerate(background.edge_tuples):
        axes[e] = axis
        if axis == 0:
            centers[e] = [
                0.5 * (background.x[i] + background.x[i + 1]),
                background.y[j],
                background.z[k],
            ]
        elif axis == 1:
            centers[e] = [
                background.x[i],
                0.5 * (background.y[j] + background.y[j + 1]),
                background.z[k],
            ]
        else:
            centers[e] = [
                background.x[i],
                background.y[j],
                0.5 * (background.z[k] + background.z[k + 1]),
            ]
    return centers, axes


def edge_static_features(background):
    """Orientation and normalized physical position for every fixed edge DOF."""
    centers, axes = _edge_centers(background)
    lo = np.array([background.x[0], background.y[0], background.z[0]], dtype=float)
    hi = np.array([background.x[-1], background.y[-1], background.z[-1]], dtype=float)
    span = np.maximum(hi - lo, np.finfo(float).tiny)
    xyz = 2.0 * (centers - lo) / span - 1.0
    orient = np.eye(3, dtype=float)[axes]
    return np.concatenate([xyz, orient], axis=1)


def edge_group_ids(background, levels):
    """Nested orientation-preserving geometric aggregates for coarse processing."""
    groups = []
    for level in range(1, int(levels) + 1):
        stride = 2**level
        lookup = {}
        ids = np.empty(background.n_edges, dtype=np.int64)
        for e, (axis, i, j, k) in enumerate(background.edge_tuples):
            key = (axis, i // stride, j // stride, k // stride)
            if key not in lookup:
                lookup[key] = len(lookup)
            ids[e] = lookup[key]
        groups.append((ids, len(lookup)))
    return tuple(groups)


def _activation(torch, name):
    if name == "silu":
        return torch.nn.SiLU()
    if name == "gelu":
        return torch.nn.GELU()
    return torch.nn.Tanh()


def build_edge_residual_operator(background, config=None):
    """Build the coefficient predictor for the short operator polynomial."""
    import torch

    cfg = (
        config
        if isinstance(config, EdgeMultiscaleConfig)
        else EdgeMultiscaleConfig(**dict(config or {}))
    )
    static = edge_static_features(background)
    groups = edge_group_ids(background, cfg.levels)

    class EdgePolynomialCoefficientOperator(torch.nn.Module):
        # r, MR-Jacobi direction, post-smoother residual: 3 complex fields,
        # followed by five scalar local operator statistics.
        dynamic_dimension = 11
        static_dimension = 6
        feature_schema = FEATURE_SCHEMA

        def __init__(self):
            super().__init__()
            self.config = cfg
            self.n_edges = int(background.n_edges)
            self.polynomial_order = int(cfg.polynomial_order)
            self.register_buffer(
                "static_features",
                torch.as_tensor(static, dtype=torch.float64),
                persistent=False,
            )
            self._group_names = []
            self._group_counts = []
            for level, (ids, count) in enumerate(groups):
                name = f"group_ids_{level}"
                self.register_buffer(
                    name, torch.as_tensor(ids, dtype=torch.long), persistent=False
                )
                self._group_names.append(name)
                self._group_counts.append(int(count))

            act = lambda: _activation(torch, cfg.activation)
            self.input = torch.nn.Linear(
                self.dynamic_dimension + self.static_dimension, cfg.width
            )
            self.local = torch.nn.ModuleList()
            self.coarse = torch.nn.ModuleList()
            for _ in range(cfg.levels):
                local_layers = []
                coarse_layers = []
                for _ in range(cfg.blocks_per_level):
                    local_layers.extend(
                        [
                            torch.nn.Linear(cfg.width, cfg.width),
                            act(),
                            torch.nn.Linear(cfg.width, cfg.width),
                            act(),
                        ]
                    )
                    coarse_layers.extend(
                        [
                            torch.nn.Linear(cfg.width, cfg.width),
                            act(),
                            torch.nn.Linear(cfg.width, cfg.width),
                            act(),
                        ]
                    )
                self.local.append(torch.nn.Sequential(*local_layers))
                self.coarse.append(torch.nn.Sequential(*coarse_layers))

            self.head_norm = torch.nn.LayerNorm(cfg.width)
            self.head_hidden = torch.nn.Sequential(
                torch.nn.Linear(cfg.width, cfg.width), act()
            )
            self.output = torch.nn.Linear(cfg.width, 2 * self.polynomial_order)
            # A fresh network is exactly the physical MR-Jacobi baseline.
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

        def _pool(self, h, ids, count):
            batch, n, width = h.shape
            pooled = h.new_zeros((batch, count, width))
            pooled.index_add_(1, ids, h)
            counts = h.new_zeros(count)
            counts.index_add_(0, ids, h.new_ones(n))
            return pooled / counts.clamp_min(1.0)[None, :, None]

        def forward(self, dynamic):
            if dynamic.ndim == 2:
                dynamic = dynamic.unsqueeze(0)
            if (
                dynamic.ndim != 3
                or dynamic.shape[1] != self.n_edges
                or dynamic.shape[2] != self.dynamic_dimension
            ):
                raise ValueError("edge residual features have the wrong shape")
            static_features = self.static_features.to(
                dtype=dynamic.dtype, device=dynamic.device
            )
            static_batch = static_features.unsqueeze(0).expand(
                dynamic.shape[0], -1, -1
            )
            h = self.input(torch.cat([dynamic, static_batch], dim=-1))
            multiscale = h
            for level, (local, coarse) in enumerate(zip(self.local, self.coarse)):
                h = h + local(h)
                ids = getattr(self, self._group_names[level])
                pooled = self._pool(h, ids, self._group_counts[level])
                pooled = pooled + coarse(pooled)
                multiscale = multiscale + pooled[:, ids, :]

            # Only a handful of global complex coefficients are learned.  The
            # spatial directions themselves are generated by the exact A and D.
            summary = multiscale.mean(dim=1)
            summary = self.head_hidden(self.head_norm(summary))
            raw = self.output(summary).reshape(
                dynamic.shape[0], self.polynomial_order, 2
            )
            return float(cfg.coefficient_limit) * torch.tanh(raw)

    return EdgePolynomialCoefficientOperator()


def safe_diagonal_values(diagonal):
    diagonal = np.asarray(diagonal, complex).reshape(-1)
    if diagonal.size == 0 or np.any(~np.isfinite(diagonal)):
        raise ValueError("invalid Maxwell diagonal")
    magnitude = np.abs(diagonal)
    scale = max(float(np.max(magnitude, initial=0.0)), np.finfo(float).tiny)
    mask = magnitude <= 1e-12 * scale
    if np.any(mask):
        diagonal = diagonal.copy()
        diagonal[mask] = scale + 0j
    return diagonal


def safe_diagonal(A):
    diagonal = np.asarray(A.diagonal(), complex)
    if diagonal.ndim != 1 or diagonal.shape[0] != A.shape[0]:
        raise ValueError("invalid Maxwell diagonal")
    return safe_diagonal_values(diagonal)


def sparse_row_statistics(A):
    """Return exact sparse row |A| sums and row nonzero counts without densifying A."""
    matrix = A.tocsr()
    row_abs_sum = np.asarray(np.abs(matrix).sum(axis=1), float).reshape(-1)
    row_nnz = np.diff(matrix.indptr).astype(float, copy=False)
    if (
        row_abs_sum.shape != (matrix.shape[0],)
        or row_nnz.shape != (matrix.shape[0],)
        or np.any(~np.isfinite(row_abs_sum))
    ):
        raise ValueError("invalid sparse Maxwell row statistics")
    return row_abs_sum, row_nnz


def operator_feature_statistics(A):
    """Compute immutable O(n_edges) statistics once for one assembled A."""
    diagonal = safe_diagonal(A)
    row_abs_sum, row_nnz = sparse_row_statistics(A)
    return diagonal, row_abs_sum, row_nnz


def _as_residual_matrix(residual, size):
    R = np.asarray(residual, complex)
    if R.ndim == 1:
        R = R[:, None]
    if R.ndim != 2 or R.shape[0] != size or np.any(~np.isfinite(R)):
        raise ValueError("residual dimension does not match Maxwell operator")
    return R


def _mr_jacobi_state(diagonal, residual, apply_operator):
    diagonal = safe_diagonal_values(diagonal)
    R = _as_residual_matrix(residual, diagonal.size)
    jacobi = R / diagonal[:, None]
    image = np.asarray(apply_operator(jacobi), complex)
    if image.ndim == 1:
        image = image[:, None]
    if image.shape != R.shape or np.any(~np.isfinite(image)):
        raise FloatingPointError("Maxwell operator action is invalid")
    denominator = np.sum(np.abs(image) ** 2, axis=0)
    numerator = np.sum(np.conj(image) * R, axis=0)
    tiny = np.finfo(float).tiny
    alpha = np.zeros(R.shape[1], complex)
    valid = np.isfinite(denominator) & (denominator > tiny) & np.isfinite(numerator)
    alpha[valid] = numerator[valid] / denominator[valid]
    baseline = jacobi * alpha[None, :]
    post = R - image * alpha[None, :]
    if np.any(~np.isfinite(baseline)) or np.any(~np.isfinite(post)):
        raise FloatingPointError("minimum-residual Jacobi baseline is non-finite")
    return R, jacobi, image, baseline, post


def minimum_residual_jacobi_from_action(diagonal, residual, apply_operator):
    """Best scalar Jacobi correction and its true post-correction residual."""
    _, _, _, baseline, post = _mr_jacobi_state(
        diagonal, residual, apply_operator
    )
    return baseline, post


def _operator_features(diagonal, row_abs_sum, row_nnz):
    dabs = np.abs(diagonal)
    tiny = np.finfo(float).tiny
    median = max(float(np.median(dabs)), tiny)
    logdiag = np.clip(np.log(np.maximum(dabs, tiny) / median), -12.0, 12.0)
    phase = np.angle(diagonal) / np.pi
    loss_ratio = np.clip(
        np.log1p(np.abs(diagonal.imag) / np.maximum(np.abs(diagonal.real), tiny)),
        0.0,
        12.0,
    )
    row_abs = (
        dabs.copy()
        if row_abs_sum is None
        else np.asarray(row_abs_sum, float).reshape(-1)
    )
    nnz = (
        np.ones(diagonal.size, float)
        if row_nnz is None
        else np.asarray(row_nnz, float).reshape(-1)
    )
    if (
        row_abs.shape != (diagonal.size,)
        or nnz.shape != (diagonal.size,)
        or np.any(~np.isfinite(row_abs))
        or np.any(~np.isfinite(nnz))
        or np.any(row_abs < 0)
        or np.any(nnz < 0)
    ):
        raise ValueError("invalid Maxwell row statistics")
    offdiag_abs = np.maximum(row_abs - dabs, 0.0)
    coupling_ratio = np.clip(
        np.log1p(offdiag_abs / np.maximum(dabs, tiny)), 0.0, 12.0
    )
    degree_scale = max(float(np.median(np.maximum(nnz, 1.0))), 1.0)
    degree = np.clip(
        np.log1p(np.maximum(nnz - 1.0, 0.0)) / np.log1p(degree_scale),
        0.0,
        4.0,
    )
    return np.column_stack(
        [logdiag, phase, loss_ratio, coupling_ratio, degree]
    )


def polynomial_state_from_operator_action(
    diagonal,
    residual,
    apply_operator,
    *,
    polynomial_order=3,
    row_abs_sum=None,
    row_nnz=None,
):
    """Build exact features, MR baseline and normalized polynomial directions.

    The spatial basis is
    ``q0=D^-1 r`` and ``q{k+1}=D^-1 A qk``.  Each q is RMS-normalized to
    the scale of q0, while the zero-neural-output baseline is the optimal
    one-dimensional MR-Jacobi correction.
    """
    order = int(polynomial_order)
    if not 1 <= order <= 6:
        raise ValueError("polynomial_order must lie in [1, 6]")
    diagonal = safe_diagonal_values(diagonal)
    R, jacobi, first_image, baseline, post = _mr_jacobi_state(
        diagonal, residual, apply_operator
    )
    n = max(1, diagonal.size)
    tiny = np.finfo(float).tiny
    rscale = np.maximum(np.linalg.norm(R, axis=0) / np.sqrt(n), tiny)
    zscale = np.maximum(np.linalg.norm(jacobi, axis=0) / np.sqrt(n), tiny)
    rn = R / rscale[None, :]
    bn = baseline / zscale[None, :]
    pn = post / rscale[None, :]
    operator = _operator_features(diagonal, row_abs_sum, row_nnz)

    features = []
    for p in range(R.shape[1]):
        features.append(
            np.column_stack(
                [
                    rn[:, p].real,
                    rn[:, p].imag,
                    bn[:, p].real,
                    bn[:, p].imag,
                    pn[:, p].real,
                    pn[:, p].imag,
                    operator,
                ]
            )
        )

    directions = []
    q = jacobi
    image = first_image
    for k in range(order):
        qscale = np.maximum(np.linalg.norm(q, axis=0) / np.sqrt(n), tiny)
        directions.append(q * (zscale / qscale)[None, :])
        if k + 1 < order:
            q = image / diagonal[:, None]
            image = np.asarray(apply_operator(q), complex)
            if image.ndim == 1:
                image = image[:, None]
            if image.shape != R.shape or np.any(~np.isfinite(image)):
                raise FloatingPointError("polynomial Maxwell operator action is invalid")

    basis = np.stack(directions, axis=2)  # [edge, rhs, polynomial-order]
    return np.asarray(features, np.float64), baseline, basis, zscale


def residual_features_from_operator_action(
    diagonal,
    residual,
    apply_operator,
    row_abs_sum=None,
    row_nnz=None,
):
    """Compatibility view of the exact operator-action feature state."""
    features, baseline, _, zscale = polynomial_state_from_operator_action(
        diagonal,
        residual,
        apply_operator,
        polynomial_order=1,
        row_abs_sum=row_abs_sum,
        row_nnz=row_nnz,
    )
    return features, baseline, zscale


def residual_features_from_diagonal(
    diagonal, residual, row_abs_sum=None, row_nnz=None
):
    diagonal = safe_diagonal_values(diagonal)

    def diagonal_action(Z):
        return diagonal[:, None] * np.asarray(Z, complex)

    return residual_features_from_operator_action(
        diagonal,
        residual,
        diagonal_action,
        row_abs_sum=row_abs_sum,
        row_nnz=row_nnz,
    )


def residual_features(A, residual, *, operator_stats=None):
    """Runtime features using exact sparse statistics and exact ``A @ z``."""
    if operator_stats is None:
        operator_stats = operator_feature_statistics(A)
    diagonal, row_abs_sum, row_nnz = operator_stats
    return residual_features_from_operator_action(
        diagonal,
        residual,
        lambda Z: A @ Z,
        row_abs_sum=row_abs_sum,
        row_nnz=row_nnz,
    )


def neural_correction(network, A, residual, *, operator_stats=None):
    """MR-Jacobi plus a learned short exact-operator polynomial direction."""
    import torch

    if operator_stats is None:
        operator_stats = operator_feature_statistics(A)
    diagonal, row_abs_sum, row_nnz = operator_stats
    order = int(getattr(network, "polynomial_order", 3))
    features, baseline, basis, _ = polynomial_state_from_operator_action(
        diagonal,
        residual,
        lambda Z: A @ Z,
        polynomial_order=order,
        row_abs_sum=row_abs_sum,
        row_nnz=row_nnz,
    )
    parameter = next(network.parameters())
    x = torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        y = network(x).detach().cpu().double().numpy()
    if y.shape != (features.shape[0], order, 2) or np.any(~np.isfinite(y)):
        return baseline
    coefficients = y[..., 0] + 1j * y[..., 1]
    learned = np.einsum("nmk,mk->nm", basis, coefficients, optimize=True)
    correction = baseline + learned
    if np.any(~np.isfinite(correction)):
        return baseline
    return correction


__all__ = [
    "FEATURE_SCHEMA",
    "EdgeMultiscaleConfig",
    "build_edge_residual_operator",
    "edge_group_ids",
    "edge_static_features",
    "minimum_residual_jacobi_from_action",
    "neural_correction",
    "operator_feature_statistics",
    "polynomial_state_from_operator_action",
    "residual_features",
    "residual_features_from_diagonal",
    "residual_features_from_operator_action",
    "safe_diagonal",
    "safe_diagonal_values",
    "sparse_row_statistics",
]
