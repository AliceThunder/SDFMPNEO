"""Compact solution-label-free training data for the neural Maxwell accelerator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def operator_encoding(A, B, V):
    """Encode one physical operator and its full-space residual quadratic form."""
    AV = A @ V
    Ar = V.conj().T @ AV
    br = V.conj().T @ B
    r = V.shape[1]
    tiny = np.finfo(float).tiny
    ascale = max(float(np.linalg.norm(Ar)) / np.sqrt(max(r, 1)), tiny)
    bscale = np.maximum(np.linalg.norm(br, axis=0), tiny)
    Ah = Ar / ascale
    Bh = br / bscale[None, :]
    features = np.concatenate([
        Ah.real.ravel(), Ah.imag.ravel(), Bh.real.ravel(), Bh.imag.ravel(),
        [np.log(ascale)], np.log(bscale),
    ])

    diagonal = np.diag(Ar)
    diagonal_scale = max(float(np.max(np.abs(diagonal))), tiny)
    diagonal = np.where(np.abs(diagonal) > 1e-12 * diagonal_scale, diagonal, diagonal_scale + 0j)
    base = br / diagonal[:, None]
    coefficient_scale = bscale / ascale
    baseline = np.concatenate([base.real.T, base.imag.T], axis=1)

    Q = AV.conj().T @ AV
    S = AV.conj().T @ B
    Qr, Qi = Q.real, Q.imag
    qblock = np.block([[Qr, -Qi], [Qi, Qr]])
    sreal = np.concatenate([S.real.T, S.imag.T], axis=1)
    norm2 = np.sum(np.abs(B) ** 2, axis=0).real
    return tuple(np.asarray(x, np.float64) for x in (
        features, baseline, coefficient_scale, qblock, sreal, norm2,
    ))


@dataclass
class MaxwellOperatorDataset:
    features: np.ndarray
    baseline: np.ndarray
    coefficient_scale: np.ndarray
    residual_gram: np.ndarray
    residual_linear: np.ndarray
    rhs_norm2: np.ndarray
    split: np.ndarray
    reduced_rank: int
    n_rhs: int

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                features=self.features,
                baseline=self.baseline,
                coefficient_scale=self.coefficient_scale,
                residual_gram=self.residual_gram,
                residual_linear=self.residual_linear,
                rhs_norm2=self.rhs_norm2,
                split=self.split,
                reduced_rank=np.array(self.reduced_rank),
                n_rhs=np.array(self.n_rhs),
            )
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["features"], data["baseline"], data["coefficient_scale"],
                data["residual_gram"], data["residual_linear"], data["rhs_norm2"],
                data["split"], int(data["reduced_rank"]), int(data["n_rhs"]),
            )

    def indices(self, name):
        return np.flatnonzero(self.split == {"train": 0, "validation": 1, "test": 2}[name])


def generate_operator_dataset(background, V, geometry_samples, state_samples, *, seed=0, monitor=None):
    pairs = list(zip(geometry_samples, state_samples))
    n = len(pairs)
    if n < 10:
        raise ValueError("at least ten operator samples are required")
    rows = []
    for i, (geometry, state) in enumerate(pairs):
        if monitor is not None:
            monitor.checkpoint()
        context = background.geometry_context(geometry, assemble_thermal=False)
        A = background.em_operator(context, state)
        B = background.rhs_matrix(context)
        rows.append(operator_encoding(A, B, V))
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(phase="maxwell_operator_samples", training_points=i + 1)
        if i == 0 or (i + 1) % max(1, n // 20) == 0 or i + 1 == n:
            print(
                f"生成 Maxwell 残差训练算子……{100 * (i + 1) / n:5.1f}% ({i + 1}/{n})",
                flush=True,
            )
    features, base, scale, Q, S, N = map(np.asarray, zip(*rows))
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    split = np.zeros(n, np.int8)
    n_validation = max(1, int(round(0.15 * n)))
    n_test = max(1, int(round(0.15 * n)))
    split[order[:n_validation]] = 1
    split[order[n_validation:n_validation + n_test]] = 2
    return MaxwellOperatorDataset(
        features, base, scale, Q, S, N, split, V.shape[1], base.shape[1],
    )


__all__ = ["MaxwellOperatorDataset", "generate_operator_dataset", "operator_encoding"]
