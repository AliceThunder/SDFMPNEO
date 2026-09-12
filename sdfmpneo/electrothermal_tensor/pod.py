"""POD/SVD compression for Frobenius-isometric quadratic Joule tensors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .symmetric import quadratic_feature


@dataclass(frozen=True)
class TensorPOD:
    mean: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    thermal_rank: int
    current_dimension: int

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        basis = np.asarray(self.basis, dtype=float)
        singular = np.asarray(self.singular_values, dtype=float).reshape(-1)
        if basis.ndim != 2 or basis.shape[0] != mean.size or basis.shape[1] < 1:
            raise ValueError("POD basis dimensions do not match mean")
        if singular.size < basis.shape[1] or np.any(singular < 0) or np.any(~np.isfinite(singular)):
            raise ValueError("invalid POD singular values")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(basis)):
            raise ValueError("POD arrays must be finite")
        gram = basis.T @ basis
        if not np.allclose(gram, np.eye(basis.shape[1]), rtol=1e-10, atol=1e-12):
            raise ValueError("POD basis must be orthonormal")
        p = int(self.current_dimension) + 1
        n_sym = p * (p + 1) // 2
        if mean.size != int(self.thermal_rank) * n_sym:
            raise ValueError("POD vector width does not match thermal/current dimensions")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "thermal_rank", int(self.thermal_rank))
        object.__setattr__(self, "current_dimension", int(self.current_dimension))

    @property
    def rank(self) -> int:
        return self.basis.shape[1]

    @property
    def output_dimension(self) -> int:
        return self.mean.size

    @property
    def packed_symmetric_size(self) -> int:
        p = self.current_dimension + 1
        return p * (p + 1) // 2

    def encode(self, outputs: np.ndarray) -> np.ndarray:
        value = np.asarray(outputs, dtype=float)
        if value.shape[-1] != self.output_dimension:
            raise ValueError("POD encode width mismatch")
        return (value - self.mean) @ self.basis

    def decode(self, coefficients: np.ndarray) -> np.ndarray:
        value = np.asarray(coefficients, dtype=float)
        if value.shape[-1] != self.rank:
            raise ValueError("POD coefficient width mismatch")
        return self.mean + value @ self.basis.T

    def relative_rms_error(self, outputs: np.ndarray) -> float:
        value = np.asarray(outputs, dtype=float)
        reconstructed = self.decode(self.encode(value))
        denominator = max(float(np.linalg.norm(value)), np.finfo(float).tiny)
        return float(np.linalg.norm(reconstructed - value) / denominator)

    def maximum_relative_sample_error(self, outputs: np.ndarray) -> float:
        value = np.asarray(outputs, dtype=float)
        reconstructed = self.decode(self.encode(value))
        error = np.linalg.norm(reconstructed - value, axis=-1)
        scale = np.maximum(np.linalg.norm(value, axis=-1), np.finfo(float).tiny)
        return float(np.max(error / scale))

    def energy_fraction(self) -> float:
        all_energy = float(np.sum(self.singular_values**2))
        if all_energy == 0.0:
            return 1.0
        return float(np.sum(self.singular_values[: self.rank] ** 2) / all_energy)


@dataclass(frozen=True)
class PODRankDiagnostic:
    rank: int
    energy_fraction: float
    validation_relative_rms: float
    validation_percentile_99: float
    validation_maximum_relative_error: float
    sampled_heat_relative_rms: float
    sampled_heat_maximum_relative_error: float


def _select_rank(singular_values: np.ndarray, relative_tail_tolerance: float) -> int:
    singular = np.asarray(singular_values, dtype=float)
    tolerance = float(relative_tail_tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("relative_tail_tolerance must satisfy 0 <= tol < 1")
    energy = singular**2
    total = float(np.sum(energy))
    if total == 0.0:
        return 1
    cumulative = np.cumsum(energy)
    target = (1.0 - tolerance**2) * total
    return int(np.searchsorted(cumulative, target, side="left") + 1)


def fit_tensor_pod(
    outputs: np.ndarray,
    *,
    thermal_rank: int,
    current_dimension: int,
    rank: int | None = None,
    relative_tail_tolerance: float = 1e-4,
) -> TensorPOD:
    """Fit POD on training outputs only.

    Because dataset outputs use the sqrt(2)-scaled symmetric ``svec`` map,
    Euclidean POD error is exactly the Frobenius error of the underlying tensor.
    """
    value = np.asarray(outputs, dtype=float)
    if value.ndim != 2 or value.shape[0] < 2 or np.any(~np.isfinite(value)):
        raise ValueError("POD outputs must be a finite matrix with at least two samples")
    mean = np.mean(value, axis=0)
    centered = value - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    maximum_rank = max(1, vt.shape[0])
    if rank is None:
        selected = _select_rank(singular, relative_tail_tolerance)
    else:
        selected = int(rank)
        if not 1 <= selected <= maximum_rank:
            raise ValueError("requested POD rank is out of range")
    basis = vt[:selected].T.copy()
    return TensorPOD(
        mean=mean,
        basis=basis,
        singular_values=singular,
        thermal_rank=int(thermal_rank),
        current_dimension=int(current_dimension),
    )


def fit_dataset_pod(dataset, *, rank: int | None = None, relative_tail_tolerance: float = 1e-4) -> TensorPOD:
    """Fit POD strictly on the frozen training split."""
    _, outputs = dataset.arrays_for("train")
    return fit_tensor_pod(
        outputs,
        thermal_rank=dataset.thermal_rank,
        current_dimension=dataset.current_dimension,
        rank=rank,
        relative_tail_tolerance=relative_tail_tolerance,
    )


def pod_rank_sweep(
    dataset,
    ranks,
    *,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
    operating_samples_per_state: int = 4,
    seed: int = 0,
) -> tuple[PODRankDiagnostic, ...]:
    """Gate 2: one SVD, many held-out rank diagnostics.

    The sweep reports both tensor reconstruction and sampled worst-current heat
    errors. It never fits to validation/test rows.
    """
    train_ids = dataset.indices("train")
    validation_ids = dataset.indices("validation")
    train = dataset.outputs[train_ids]
    validation = dataset.outputs[validation_ids]
    mean = np.mean(train, axis=0)
    _, singular, vt = np.linalg.svd(train - mean, full_matrices=False)
    requested = sorted(set(int(rank) for rank in ranks))
    if not requested or requested[0] < 1 or requested[-1] > vt.shape[0]:
        raise ValueError("POD sweep ranks are outside available SVD rank")
    lo = np.asarray(operating_lower, dtype=float).reshape(-1)
    hi = np.asarray(operating_upper, dtype=float).reshape(-1)
    if lo.shape != (dataset.current_dimension,) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("operating bounds mismatch")
    rng = np.random.default_rng(int(seed))
    operating_draws = rng.uniform(lo, hi, size=(max(1, int(operating_samples_per_state)), dataset.current_dimension))
    features = np.asarray([quadratic_feature(u) for u in operating_draws])
    total_energy = max(float(np.sum(singular**2)), np.finfo(float).tiny)
    diagnostics = []
    n_sym = dataset.packed_symmetric_size
    true_modes = validation.reshape(len(validation), dataset.thermal_rank, n_sym)

    for rank in requested:
        basis = vt[:rank].T
        beta = (validation - mean) @ basis
        reconstructed = mean + beta @ basis.T
        sample_error = np.linalg.norm(reconstructed - validation, axis=1)
        sample_scale = np.maximum(np.linalg.norm(validation, axis=1), np.finfo(float).tiny)
        relative = sample_error / sample_scale
        reconstructed_modes = reconstructed.reshape(len(validation), dataset.thermal_rank, n_sym)
        heat_relative = []
        heat_sq_error = heat_sq_scale = 0.0
        for feature in features:
            q_true = np.einsum("nrs,s->nr", true_modes, feature, optimize=True)
            q_pred = np.einsum("nrs,s->nr", reconstructed_modes, feature, optimize=True)
            err = np.linalg.norm(q_pred - q_true, axis=1)
            scale = np.maximum(np.linalg.norm(q_true, axis=1), np.finfo(float).tiny)
            heat_relative.extend((err / scale).tolist())
            heat_sq_error += float(np.sum(err**2))
            heat_sq_scale += float(np.sum(scale**2))
        diagnostics.append(
            PODRankDiagnostic(
                rank=rank,
                energy_fraction=float(np.sum(singular[:rank] ** 2) / total_energy),
                validation_relative_rms=float(np.linalg.norm(reconstructed - validation) / max(np.linalg.norm(validation), np.finfo(float).tiny)),
                validation_percentile_99=float(np.percentile(relative, 99)),
                validation_maximum_relative_error=float(np.max(relative)),
                sampled_heat_relative_rms=float(np.sqrt(heat_sq_error / max(heat_sq_scale, np.finfo(float).tiny))),
                sampled_heat_maximum_relative_error=float(max(heat_relative, default=0.0)),
            )
        )
    return tuple(diagnostics)


__all__ = [
    "PODRankDiagnostic",
    "TensorPOD",
    "fit_dataset_pod",
    "fit_tensor_pod",
    "pod_rank_sweep",
]
