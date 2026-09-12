"""POD/SVD compression for Frobenius-isometric quadratic Joule tensors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, svds

from .symmetric import quadratic_feature


@dataclass(frozen=True)
class TensorPOD:
    mean: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    thermal_rank: int
    current_dimension: int
    total_centered_energy: float | None = None

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
        if not np.allclose(gram, np.eye(basis.shape[1]), rtol=1e-9, atol=1e-11):
            raise ValueError("POD basis must be orthonormal")
        p = int(self.current_dimension) + 1
        n_sym = p * (p + 1) // 2
        if mean.size != int(self.thermal_rank) * n_sym:
            raise ValueError("POD vector width does not match thermal/current dimensions")
        total = self.total_centered_energy
        if total is None:
            total = float(np.sum(singular**2))
        total = float(total)
        if not np.isfinite(total) or total < 0.0:
            raise ValueError("POD total centered energy must be finite and non-negative")
        captured = float(np.sum(singular[: basis.shape[1]] ** 2))
        if captured > total * (1.0 + 1e-8) + 1e-12:
            raise ValueError("POD captured energy exceeds total centered energy")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "singular_values", singular)
        object.__setattr__(self, "thermal_rank", int(self.thermal_rank))
        object.__setattr__(self, "current_dimension", int(self.current_dimension))
        object.__setattr__(self, "total_centered_energy", total)

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
        total = float(self.total_centered_energy)
        if total == 0.0:
            return 1.0
        return float(min(1.0, np.sum(self.singular_values[: self.rank] ** 2) / total))


@dataclass(frozen=True)
class PODRankDiagnostic:
    rank: int
    energy_fraction: float
    validation_relative_rms: float
    validation_percentile_99: float
    validation_maximum_relative_error: float
    sampled_heat_relative_rms: float
    sampled_heat_maximum_relative_error: float


def _select_rank(
    singular_values: np.ndarray,
    relative_tail_tolerance: float,
    *,
    total_energy: float | None = None,
) -> int:
    singular = np.asarray(singular_values, dtype=float)
    tolerance = float(relative_tail_tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("relative_tail_tolerance must satisfy 0 <= tol < 1")
    energy = singular**2
    total = float(np.sum(energy) if total_energy is None else total_energy)
    if total == 0.0:
        return 1
    cumulative = np.cumsum(energy)
    target = (1.0 - tolerance**2) * total
    index = int(np.searchsorted(cumulative, target, side="left"))
    if index >= singular.size:
        raise ValueError(
            "requested POD tail tolerance is not reached by the computed out-of-core rank; "
            "increase out_of_core_max_rank or provide an explicit rank"
        )
    return index + 1


def fit_tensor_pod(
    outputs: np.ndarray,
    *,
    thermal_rank: int,
    current_dimension: int,
    rank: int | None = None,
    relative_tail_tolerance: float = 1e-4,
) -> TensorPOD:
    """Fit an exact in-memory POD on training outputs only."""
    value = np.asarray(outputs, dtype=float)
    if value.ndim != 2 or value.shape[0] < 2 or np.any(~np.isfinite(value)):
        raise ValueError("POD outputs must be a finite matrix with at least two samples")
    mean = np.mean(value, axis=0)
    centered = value - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    total = float(np.sum(singular**2))
    maximum_rank = max(1, vt.shape[0])
    if rank is None:
        selected = _select_rank(singular, relative_tail_tolerance, total_energy=total)
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
        total_centered_energy=total,
    )


def _chunked_mean_and_energy(outputs, ids: np.ndarray, *, chunk_rows: int):
    ids = np.asarray(ids, dtype=int)
    if ids.size < 2:
        raise ValueError("POD training split needs at least two samples")
    width = int(outputs.shape[1])
    total = np.zeros(width, dtype=np.float64)
    for start in range(0, len(ids), int(chunk_rows)):
        block = np.asarray(outputs[ids[start:start + int(chunk_rows)]], dtype=np.float64)
        if np.any(~np.isfinite(block)):
            raise ValueError("POD outputs contain non-finite values")
        total += np.sum(block, axis=0)
    mean = total / float(len(ids))
    energy = 0.0
    for start in range(0, len(ids), int(chunk_rows)):
        block = np.asarray(outputs[ids[start:start + int(chunk_rows)]], dtype=np.float64)
        centered = block - mean
        energy += float(np.einsum("ij,ij->", centered, centered, optimize=True))
    return mean, energy


def _centered_linear_operator(outputs, ids: np.ndarray, mean: np.ndarray, *, chunk_rows: int):
    ids = np.asarray(ids, dtype=int)
    n = len(ids)
    width = int(outputs.shape[1])
    chunk = max(1, int(chunk_rows))

    def matmat(value):
        V = np.asarray(value, dtype=np.float64)
        result = np.empty((n, V.shape[1]), dtype=np.float64)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            block = np.asarray(outputs[ids[start:stop]], dtype=np.float64) - mean
            result[start:stop] = block @ V
        return result

    def rmatmat(value):
        U = np.asarray(value, dtype=np.float64)
        result = np.zeros((width, U.shape[1]), dtype=np.float64)
        for start in range(0, n, chunk):
            stop = min(n, start + chunk)
            block = np.asarray(outputs[ids[start:stop]], dtype=np.float64) - mean
            result += block.T @ U[start:stop]
        return result

    def matvec(value):
        return matmat(np.asarray(value, dtype=np.float64).reshape(-1, 1))[:, 0]

    def rmatvec(value):
        return rmatmat(np.asarray(value, dtype=np.float64).reshape(-1, 1))[:, 0]

    return LinearOperator(
        (n, width),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=np.float64,
    )


def _out_of_core_decomposition(
    dataset,
    *,
    max_rank: int,
    chunk_rows: int,
):
    ids = dataset.indices("train")
    mean, total_energy = _chunked_mean_and_energy(dataset.outputs, ids, chunk_rows=chunk_rows)
    min_dimension = min(len(ids), int(dataset.outputs.shape[1]))
    if min_dimension <= 2:
        values = np.asarray(dataset.outputs[ids], dtype=np.float64)
        exact = fit_tensor_pod(
            values,
            thermal_rank=dataset.thermal_rank,
            current_dimension=dataset.current_dimension,
            rank=min(1, min_dimension),
        )
        return exact.mean, exact.singular_values, exact.basis.T, exact.total_centered_energy
    k = min(int(max_rank), min_dimension - 1)
    if k < 1:
        raise ValueError("out-of-core POD has no available singular directions")
    operator = _centered_linear_operator(dataset.outputs, ids, mean, chunk_rows=chunk_rows)
    # ARPACK is deterministic for a fixed starting vector and matrix/operator.
    v0 = np.ones(min_dimension, dtype=np.float64)
    _, singular, vt = svds(
        operator,
        k=k,
        which="LM",
        v0=v0,
        return_singular_vectors=True,
    )
    order = np.argsort(singular)[::-1]
    singular = np.asarray(singular[order], dtype=np.float64)
    vt = np.asarray(vt[order], dtype=np.float64)
    return mean, singular, vt, float(total_energy)


def fit_dataset_pod(
    dataset,
    *,
    rank: int | None = None,
    relative_tail_tolerance: float = 1e-4,
    exact_svd_max_bytes: int = 512 << 20,
    out_of_core_max_rank: int = 256,
    chunk_rows: int = 256,
) -> TensorPOD:
    """Fit POD strictly on the frozen training split.

    Small datasets use exact dense SVD.  Wide/large datasets use a chunked
    ``LinearOperator`` and truncated ``svds`` without materializing the centered
    training matrix.  Automatic rank selection fails closed when the requested
    tail tolerance is not captured within ``out_of_core_max_rank``.
    """
    train_ids = dataset.indices("train")
    bytes_required = int(len(train_ids)) * int(dataset.outputs.shape[1]) * np.dtype(np.float64).itemsize
    if bytes_required <= int(exact_svd_max_bytes):
        outputs = np.asarray(dataset.outputs[train_ids], dtype=np.float64)
        return fit_tensor_pod(
            outputs,
            thermal_rank=dataset.thermal_rank,
            current_dimension=dataset.current_dimension,
            rank=rank,
            relative_tail_tolerance=relative_tail_tolerance,
        )

    requested_compute_rank = int(rank) if rank is not None else int(out_of_core_max_rank)
    mean, singular, vt, total_energy = _out_of_core_decomposition(
        dataset,
        max_rank=requested_compute_rank,
        chunk_rows=chunk_rows,
    )
    if rank is None:
        selected = _select_rank(
            singular,
            relative_tail_tolerance,
            total_energy=total_energy,
        )
    else:
        selected = int(rank)
        if not 1 <= selected <= vt.shape[0]:
            raise ValueError("requested POD rank exceeds computed out-of-core rank")
    return TensorPOD(
        mean=mean,
        basis=vt[:selected].T.copy(),
        singular_values=singular,
        thermal_rank=dataset.thermal_rank,
        current_dimension=dataset.current_dimension,
        total_centered_energy=total_energy,
    )


def _validation_blocks(dataset, ids: np.ndarray, *, chunk_rows: int = 128):
    ids = np.asarray(ids, dtype=int)
    for start in range(0, len(ids), int(chunk_rows)):
        selected = ids[start:start + int(chunk_rows)]
        yield selected, np.asarray(dataset.outputs[selected], dtype=np.float64)


def pod_rank_sweep(
    dataset,
    ranks,
    *,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
    operating_samples_per_state: int = 4,
    seed: int = 0,
    exact_svd_max_bytes: int = 512 << 20,
    chunk_rows: int = 128,
) -> tuple[PODRankDiagnostic, ...]:
    """Gate 2: one decomposition, many held-out rank diagnostics."""
    requested = tuple(sorted(set(int(rank) for rank in ranks)))
    if not requested or requested[0] < 1:
        raise ValueError("POD sweep ranks must be positive")
    maximum_requested = requested[-1]
    train_ids = dataset.indices("train")
    validation_ids = dataset.indices("validation")
    bytes_required = int(len(train_ids)) * int(dataset.outputs.shape[1]) * np.dtype(np.float64).itemsize
    if bytes_required <= int(exact_svd_max_bytes):
        train = np.asarray(dataset.outputs[train_ids], dtype=np.float64)
        mean = np.mean(train, axis=0)
        _, singular, vt = np.linalg.svd(train - mean, full_matrices=False)
        total_energy = float(np.sum(singular**2))
    else:
        mean, singular, vt, total_energy = _out_of_core_decomposition(
            dataset,
            max_rank=maximum_requested,
            chunk_rows=chunk_rows,
        )
    if maximum_requested > vt.shape[0]:
        raise ValueError("POD sweep rank exceeds available/computed SVD rank")

    lo = np.asarray(operating_lower, dtype=float).reshape(-1)
    hi = np.asarray(operating_upper, dtype=float).reshape(-1)
    if lo.shape != (dataset.current_dimension,) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("operating bounds mismatch")
    rng = np.random.default_rng(int(seed))
    operating_draws = rng.uniform(
        lo,
        hi,
        size=(max(1, int(operating_samples_per_state)), dataset.current_dimension),
    )
    features = np.asarray([quadratic_feature(u) for u in operating_draws])
    total_energy = max(float(total_energy), np.finfo(float).tiny)
    n_sym = dataset.packed_symmetric_size
    accum = {
        rank: {
            "sq_error": 0.0,
            "sq_scale": 0.0,
            "relative": [],
            "heat_sq_error": 0.0,
            "heat_sq_scale": 0.0,
            "heat_relative": [],
        }
        for rank in requested
    }

    for _, validation in _validation_blocks(dataset, validation_ids, chunk_rows=chunk_rows):
        centered = validation - mean
        true_modes = validation.reshape(len(validation), dataset.thermal_rank, n_sym)
        for rank in requested:
            basis = vt[:rank].T
            beta = centered @ basis
            reconstructed = mean + beta @ basis.T
            error = np.linalg.norm(reconstructed - validation, axis=1)
            scale = np.maximum(np.linalg.norm(validation, axis=1), np.finfo(float).tiny)
            record = accum[rank]
            record["sq_error"] += float(np.sum(error**2))
            record["sq_scale"] += float(np.sum(scale**2))
            record["relative"].extend((error / scale).tolist())
            reconstructed_modes = reconstructed.reshape(len(validation), dataset.thermal_rank, n_sym)
            for feature in features:
                q_true = np.einsum("nrs,s->nr", true_modes, feature, optimize=True)
                q_pred = np.einsum("nrs,s->nr", reconstructed_modes, feature, optimize=True)
                q_error = np.linalg.norm(q_pred - q_true, axis=1)
                q_scale = np.maximum(np.linalg.norm(q_true, axis=1), np.finfo(float).tiny)
                record["heat_sq_error"] += float(np.sum(q_error**2))
                record["heat_sq_scale"] += float(np.sum(q_scale**2))
                record["heat_relative"].extend((q_error / q_scale).tolist())

    diagnostics = []
    for rank in requested:
        record = accum[rank]
        relative = np.asarray(record["relative"], dtype=float)
        heat_relative = np.asarray(record["heat_relative"], dtype=float)
        diagnostics.append(
            PODRankDiagnostic(
                rank=rank,
                energy_fraction=float(min(1.0, np.sum(singular[:rank] ** 2) / total_energy)),
                validation_relative_rms=float(
                    np.sqrt(record["sq_error"] / max(record["sq_scale"], np.finfo(float).tiny))
                ),
                validation_percentile_99=float(np.percentile(relative, 99)),
                validation_maximum_relative_error=float(np.max(relative)),
                sampled_heat_relative_rms=float(
                    np.sqrt(
                        record["heat_sq_error"]
                        / max(record["heat_sq_scale"], np.finfo(float).tiny)
                    )
                ),
                sampled_heat_maximum_relative_error=float(np.max(heat_relative)),
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
