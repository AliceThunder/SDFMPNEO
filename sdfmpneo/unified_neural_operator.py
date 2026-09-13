"""Sparse-operator neural residual solver for the full-space Maxwell system.

The neural network is the Maxwell accelerator. It reads the current full-edge
residual together with the exact sparse Maxwell coupling graph and directly
predicts a full-edge correction. FGMRES is retained only as a true-residual
closure; it is not the learned model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import scipy.sparse as sp


FEATURE_SCHEMA = "edge-residual-v5-sparse-message-passing"


@dataclass(frozen=True)
class EdgeMultiscaleConfig:
    width: int = 48
    message_passing_steps: int = 4
    solver_steps: int = 3
    activation: str = "silu"

    def __post_init__(self):
        if self.width < 8 or self.message_passing_steps < 1 or self.solver_steps < 1:
            raise ValueError("invalid sparse neural solver dimensions")
        if self.activation not in {"silu", "gelu", "tanh"}:
            raise ValueError("activation must be silu, gelu, or tanh")

    def to_dict(self):
        return asdict(self)


@dataclass
class OperatorGraph:
    """Normalized sparse Maxwell coupling plus immutable node statistics."""

    diagonal: np.ndarray
    row_abs_sum: np.ndarray
    row_nnz: np.ndarray
    normalized_coupling: sp.coo_matrix
    node_features: np.ndarray
    _torch_cache: dict = field(default_factory=dict, init=False, repr=False)

    def torch_coupling(self, torch, device, network_dtype):
        complex_dtype = torch.complex64 if network_dtype == torch.float32 else torch.complex128
        key = (str(device), str(complex_dtype))
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached
        matrix = self.normalized_coupling
        indices = torch.as_tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long, device=device)
        values = torch.as_tensor(matrix.data, dtype=complex_dtype, device=device)
        value = torch.sparse_coo_tensor(indices, values, size=matrix.shape, device=device).coalesce()
        self._torch_cache[key] = value
        return value


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
    centers, axes = _edge_centers(background)
    lo = np.array([background.x[0], background.y[0], background.z[0]], dtype=float)
    hi = np.array([background.x[-1], background.y[-1], background.z[-1]], dtype=float)
    span = np.maximum(hi - lo, np.finfo(float).tiny)
    xyz = 2.0 * (centers - lo) / span - 1.0
    orient = np.eye(3, dtype=float)[axes]
    return np.concatenate([xyz, orient], axis=1)


def _activation(torch, name):
    if name == "silu": return torch.nn.SiLU()
    if name == "gelu": return torch.nn.GELU()
    return torch.nn.Tanh()


def _spmm_batch(torch, matrix, h):
    batch, nodes, width = h.shape
    flat = h.permute(1, 0, 2).reshape(nodes, batch * width)
    message = torch.sparse.mm(matrix, flat.to(dtype=matrix.dtype))
    return message.reshape(nodes, batch, width).permute(1, 0, 2)


def build_edge_residual_operator(background, config=None):
    import torch
    cfg = config if isinstance(config, EdgeMultiscaleConfig) else EdgeMultiscaleConfig(**dict(config or {}))
    static = edge_static_features(background)

    class SparseMaxwellResidualSolver(torch.nn.Module):
        dynamic_dimension = 9
        static_dimension = 6
        feature_schema = FEATURE_SCHEMA

        def __init__(self):
            super().__init__()
            self.config = cfg
            self.n_edges = int(background.n_edges)
            self.message_passing_steps = int(cfg.message_passing_steps)
            self.solver_steps = int(cfg.solver_steps)
            self.register_buffer("static_features", torch.as_tensor(static, dtype=torch.float64), persistent=False)
            act = lambda: _activation(torch, cfg.activation)
            self.input = torch.nn.Sequential(
                torch.nn.Linear(self.dynamic_dimension + self.static_dimension, cfg.width), act(),
                torch.nn.Linear(cfg.width, cfg.width), act(),
            )
            self.message_blocks = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()
            for _ in range(self.message_passing_steps):
                self.message_blocks.append(torch.nn.Sequential(
                    torch.nn.Linear(3 * cfg.width, cfg.width), act(), torch.nn.Linear(cfg.width, cfg.width)
                ))
                self.norms.append(torch.nn.LayerNorm(cfg.width))
            self.output = torch.nn.Linear(cfg.width, 2)
            torch.nn.init.zeros_(self.output.weight)
            torch.nn.init.zeros_(self.output.bias)

        def forward(self, dynamic, normalized_coupling):
            if dynamic.ndim == 2: dynamic = dynamic.unsqueeze(0)
            if dynamic.ndim != 3 or dynamic.shape[1] != self.n_edges or dynamic.shape[2] != self.dynamic_dimension:
                raise ValueError("edge residual features have the wrong shape")
            if not getattr(normalized_coupling, "is_sparse", False):
                raise ValueError("Maxwell coupling must be a torch sparse tensor")
            static_features = self.static_features.to(dtype=dynamic.dtype, device=dynamic.device)
            h = self.input(torch.cat([dynamic, static_features.unsqueeze(0).expand(dynamic.shape[0], -1, -1)], dim=-1))
            for block, norm in zip(self.message_blocks, self.norms):
                message = _spmm_batch(torch, normalized_coupling, h)
                update = block(torch.cat([h, message.real.to(h.dtype), message.imag.to(h.dtype)], dim=-1))
                h = norm(h + update)
            return self.output(h)

    return SparseMaxwellResidualSolver()


def safe_diagonal_values(diagonal):
    diagonal = np.asarray(diagonal, complex).reshape(-1)
    if diagonal.size == 0 or np.any(~np.isfinite(diagonal)): raise ValueError("invalid Maxwell diagonal")
    magnitude = np.abs(diagonal)
    scale = max(float(np.max(magnitude, initial=0.0)), np.finfo(float).tiny)
    mask = magnitude <= 1e-12 * scale
    if np.any(mask):
        diagonal = diagonal.copy(); diagonal[mask] = scale + 0j
    return diagonal


def safe_diagonal(A):
    diagonal = np.asarray(A.diagonal(), complex)
    if diagonal.ndim != 1 or diagonal.shape[0] != A.shape[0]: raise ValueError("invalid Maxwell diagonal")
    return safe_diagonal_values(diagonal)


def sparse_row_statistics(A):
    matrix = A.tocsr()
    row_abs_sum = np.asarray(np.abs(matrix).sum(axis=1), float).reshape(-1)
    row_nnz = np.diff(matrix.indptr).astype(float, copy=False)
    if row_abs_sum.shape != (matrix.shape[0],) or row_nnz.shape != (matrix.shape[0],) or np.any(~np.isfinite(row_abs_sum)):
        raise ValueError("invalid sparse Maxwell row statistics")
    return row_abs_sum, row_nnz


def _operator_node_features(diagonal, row_abs_sum, row_nnz):
    dabs = np.abs(diagonal); tiny = np.finfo(float).tiny
    median = max(float(np.median(dabs)), tiny)
    logdiag = np.clip(np.log(np.maximum(dabs, tiny) / median), -12.0, 12.0)
    phase = np.angle(diagonal) / np.pi
    loss_ratio = np.clip(np.log1p(np.abs(diagonal.imag) / np.maximum(np.abs(diagonal.real), tiny)), 0.0, 12.0)
    offdiag_abs = np.maximum(row_abs_sum - dabs, 0.0)
    coupling_ratio = np.clip(np.log1p(offdiag_abs / np.maximum(dabs, tiny)), 0.0, 12.0)
    degree_scale = max(float(np.median(np.maximum(row_nnz, 1.0))), 1.0)
    degree = np.clip(np.log1p(np.maximum(row_nnz - 1.0, 0.0)) / np.log1p(degree_scale), 0.0, 4.0)
    return np.column_stack([logdiag, phase, loss_ratio, coupling_ratio, degree]).astype(np.float64, copy=False)


def operator_feature_statistics(A):
    matrix = A.tocsr().astype(complex)
    diagonal = safe_diagonal_values(matrix.diagonal())
    row_abs_sum, row_nnz = sparse_row_statistics(matrix)
    coupling = matrix.copy(); coupling.setdiag(0); coupling.eliminate_zeros()
    offdiag_abs_sum = np.asarray(np.abs(coupling).sum(axis=1), float).reshape(-1)
    inverse = np.zeros_like(offdiag_abs_sum); mask = offdiag_abs_sum > np.finfo(float).tiny; inverse[mask] = 1.0 / offdiag_abs_sum[mask]
    normalized = (sp.diags(inverse) @ coupling).tocoo()
    if np.any(~np.isfinite(normalized.data)): raise ValueError("invalid normalized Maxwell coupling")
    return OperatorGraph(diagonal, row_abs_sum, row_nnz, normalized, _operator_node_features(diagonal, row_abs_sum, row_nnz))


def _as_residual_matrix(residual, size):
    R = np.asarray(residual, complex)
    if R.ndim == 1: R = R[:, None]
    if R.ndim != 2 or R.shape[0] != size or np.any(~np.isfinite(R)): raise ValueError("residual dimension does not match Maxwell operator")
    return R


def residual_features(residual, operator_stats):
    graph = operator_stats; R = _as_residual_matrix(residual, graph.diagonal.size)
    n = max(1, graph.diagonal.size); tiny = np.finfo(float).tiny
    jacobi = R / graph.diagonal[:, None]
    rscale = np.maximum(np.linalg.norm(R, axis=0) / np.sqrt(n), tiny)
    zscale = np.maximum(np.linalg.norm(jacobi, axis=0) / np.sqrt(n), tiny)
    rn = R / rscale[None, :]; zn = jacobi / zscale[None, :]
    features = [np.column_stack([rn[:, p].real, rn[:, p].imag, zn[:, p].real, zn[:, p].imag, graph.node_features]) for p in range(R.shape[1])]
    return np.asarray(features, np.float64), jacobi, zscale


def neural_correction(network, A, residual, *, operator_stats=None):
    import torch
    graph = operator_feature_statistics(A) if operator_stats is None else operator_stats
    features, jacobi, scale = residual_features(residual, graph)
    parameter = next(network.parameters())
    x = torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device)
    coupling = graph.torch_coupling(torch, parameter.device, parameter.dtype)
    with torch.no_grad(): y = network(x, coupling).detach().cpu().double().numpy()
    if y.shape != (features.shape[0], features.shape[1], 2) or np.any(~np.isfinite(y)): return jacobi
    correction = (y[..., 0] + 1j * y[..., 1]).T * scale[None, :]
    if np.any(~np.isfinite(correction)): return jacobi
    learned_norm = np.linalg.norm(correction, axis=0)
    reference_norm = np.maximum(np.linalg.norm(jacobi, axis=0), np.finfo(float).tiny)
    inactive = learned_norm <= 1e-12 * reference_norm
    if np.any(inactive):
        correction = correction.copy(); correction[:, inactive] = jacobi[:, inactive]
    return correction


__all__ = ["FEATURE_SCHEMA", "EdgeMultiscaleConfig", "OperatorGraph", "build_edge_residual_operator", "edge_static_features",
           "neural_correction", "operator_feature_statistics", "residual_features", "safe_diagonal", "safe_diagonal_values", "sparse_row_statistics"]
