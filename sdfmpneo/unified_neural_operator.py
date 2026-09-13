"""Edge-space multiscale neural residual corrector for the unified Maxwell solve.

The network never predicts a final Maxwell field. It maps local sparse-operator
features and a current residual to a correction that is consumed by FGMRES.
Topology is fixed by :class:`FixedMultiscaleBackground`; geometry and material
changes enter only through the assembled operator features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


FEATURE_SCHEMA = "edge-residual-v2-coupling"


@dataclass(frozen=True)
class EdgeMultiscaleConfig:
    width: int = 32
    levels: int = 3
    blocks_per_level: int = 1
    activation: str = "silu"

    def __post_init__(self):
        if self.width < 4 or self.levels < 1 or self.blocks_per_level < 1:
            raise ValueError("invalid edge multiscale network dimensions")
        if self.activation not in {"silu", "gelu", "tanh"}:
            raise ValueError("activation must be silu, gelu, or tanh")

    def to_dict(self):
        return asdict(self)


def _edge_centers(background):
    centers = np.empty((background.n_edges, 3), dtype=float)
    axes = np.empty(background.n_edges, dtype=np.int64)
    for e, (axis, i, j, k) in enumerate(background.edge_tuples):
        axes[e] = axis
        if axis == 0:
            centers[e] = [0.5 * (background.x[i] + background.x[i + 1]), background.y[j], background.z[k]]
        elif axis == 1:
            centers[e] = [background.x[i], 0.5 * (background.y[j] + background.y[j + 1]), background.z[k]]
        else:
            centers[e] = [background.x[i], background.y[j], 0.5 * (background.z[k] + background.z[k + 1])]
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
    """Build the fixed-topology edge-space neural multigrid corrector."""
    import torch

    cfg = config if isinstance(config, EdgeMultiscaleConfig) else EdgeMultiscaleConfig(**dict(config or {}))
    static = edge_static_features(background)
    groups = edge_group_ids(background, cfg.levels)

    class EdgeResidualOperator(torch.nn.Module):
        dynamic_dimension = 9
        static_dimension = 6
        feature_schema = FEATURE_SCHEMA

        def __init__(self):
            super().__init__()
            self.config = cfg
            self.n_edges = int(background.n_edges)
            self.register_buffer("static_features", torch.as_tensor(static, dtype=torch.float64), persistent=False)
            self._group_names = []
            self._group_counts = []
            for level, (ids, count) in enumerate(groups):
                name = f"group_ids_{level}"
                self.register_buffer(name, torch.as_tensor(ids, dtype=torch.long), persistent=False)
                self._group_names.append(name)
                self._group_counts.append(int(count))

            act = lambda: _activation(torch, cfg.activation)
            self.input = torch.nn.Linear(self.dynamic_dimension + self.static_dimension, cfg.width)
            self.local = torch.nn.ModuleList()
            self.coarse = torch.nn.ModuleList()
            for _ in range(cfg.levels):
                self.local.append(torch.nn.Sequential(*sum(([
                    torch.nn.Linear(cfg.width, cfg.width), act(),
                    torch.nn.Linear(cfg.width, cfg.width), act(),
                ] for _ in range(cfg.blocks_per_level)), [])))
                self.coarse.append(torch.nn.Sequential(*sum(([
                    torch.nn.Linear(cfg.width, cfg.width), act(),
                    torch.nn.Linear(cfg.width, cfg.width), act(),
                ] for _ in range(cfg.blocks_per_level)), [])))
            self.output = torch.nn.Linear(cfg.width, 2)
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
            if dynamic.ndim != 3 or dynamic.shape[1] != self.n_edges or dynamic.shape[2] != self.dynamic_dimension:
                raise ValueError("edge residual features have the wrong shape")
            static_features = self.static_features.to(dtype=dynamic.dtype, device=dynamic.device)
            static_batch = static_features.unsqueeze(0).expand(dynamic.shape[0], -1, -1)
            h = self.input(torch.cat([dynamic, static_batch], dim=-1))
            multiscale = h
            for level, (local, coarse) in enumerate(zip(self.local, self.coarse)):
                h = h + local(h)
                ids = getattr(self, self._group_names[level])
                pooled = self._pool(h, ids, self._group_counts[level])
                pooled = pooled + coarse(pooled)
                multiscale = multiscale + pooled[:, ids, :]
            return self.output(multiscale)

    return EdgeResidualOperator()


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
    """Compute immutable O(n_edges) feature statistics once for one assembled A."""
    diagonal = safe_diagonal(A)
    row_abs_sum, row_nnz = sparse_row_statistics(A)
    return diagonal, row_abs_sum, row_nnz


def residual_features_from_diagonal(diagonal, residual, row_abs_sum=None, row_nnz=None):
    """Local operator/residual features for a full edge-space correction."""
    R = np.asarray(residual, complex)
    if R.ndim == 1:
        R = R[:, None]
    diagonal = safe_diagonal_values(diagonal)
    if R.ndim != 2 or R.shape[0] != diagonal.size:
        raise ValueError("residual dimension does not match Maxwell operator")
    jacobi = R / diagonal[:, None]
    n = max(1, diagonal.size)
    tiny = np.finfo(float).tiny
    rscale = np.maximum(np.linalg.norm(R, axis=0) / np.sqrt(n), tiny)
    zscale = np.maximum(np.linalg.norm(jacobi, axis=0) / np.sqrt(n), tiny)
    rn = R / rscale[None, :]
    zn = jacobi / zscale[None, :]

    dabs = np.abs(diagonal)
    median = max(float(np.median(dabs)), tiny)
    logdiag = np.clip(np.log(np.maximum(dabs, tiny) / median), -12.0, 12.0)
    phase = np.angle(diagonal) / np.pi
    loss_ratio = np.clip(
        np.log1p(np.abs(diagonal.imag) / np.maximum(np.abs(diagonal.real), tiny)),
        0.0,
        12.0,
    )

    row_abs = dabs.copy() if row_abs_sum is None else np.asarray(row_abs_sum, float).reshape(-1)
    nnz = np.ones(diagonal.size, float) if row_nnz is None else np.asarray(row_nnz, float).reshape(-1)
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
    coupling_ratio = np.clip(np.log1p(offdiag_abs / np.maximum(dabs, tiny)), 0.0, 12.0)
    degree_scale = max(float(np.median(np.maximum(nnz, 1.0))), 1.0)
    degree = np.clip(np.log1p(np.maximum(nnz - 1.0, 0.0)) / np.log1p(degree_scale), 0.0, 4.0)
    operator = np.column_stack([logdiag, phase, loss_ratio, coupling_ratio, degree])

    features = []
    for p in range(R.shape[1]):
        features.append(
            np.column_stack([
                rn[:, p].real,
                rn[:, p].imag,
                zn[:, p].real,
                zn[:, p].imag,
                operator,
            ])
        )
    return np.asarray(features, np.float64), jacobi, zscale


def residual_features(A, residual, *, operator_stats=None):
    """Runtime wrapper using exact sparse operator statistics."""
    if operator_stats is None:
        operator_stats = operator_feature_statistics(A)
    diagonal, row_abs_sum, row_nnz = operator_stats
    return residual_features_from_diagonal(diagonal, residual, row_abs_sum, row_nnz)


def neural_correction(network, A, residual, *, operator_stats=None):
    """Physical Jacobi correction plus the learned multiscale edge correction."""
    import torch

    features, jacobi, scale = residual_features(A, residual, operator_stats=operator_stats)
    parameter = next(network.parameters())
    x = torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device)
    with torch.no_grad():
        y = network(x).detach().cpu().double().numpy()
    if np.any(~np.isfinite(y)):
        return jacobi
    learned = (y[..., 0] + 1j * y[..., 1]).T * scale[None, :]
    correction = jacobi + learned
    if np.any(~np.isfinite(correction)):
        return jacobi
    return correction


__all__ = [
    "FEATURE_SCHEMA",
    "EdgeMultiscaleConfig",
    "build_edge_residual_operator",
    "edge_group_ids",
    "edge_static_features",
    "neural_correction",
    "operator_feature_statistics",
    "residual_features",
    "residual_features_from_diagonal",
    "safe_diagonal",
    "safe_diagonal_values",
    "sparse_row_statistics",
]
