"""Compact solution-label-free training data for the neural Maxwell accelerator."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def operator_encoding(A, B, V):
    """Encode the full-space minimum-residual quadratic form.

    The neural input, Jacobi baseline, and training loss are all derived from
    the same least-squares physics

        min_C ||B - A V C||_2^2,

    whose reduced normal data are ``Q=(AV)^H(AV)`` and ``S=(AV)^H B``.
    No Maxwell solution labels are formed.
    """
    AV = np.asarray(A @ V, complex)
    B = np.asarray(B, complex)
    Q = AV.conj().T @ AV
    S = AV.conj().T @ B
    r = V.shape[1]
    tiny = np.finfo(float).tiny

    qscale = max(float(np.linalg.norm(Q)) / np.sqrt(max(r, 1)), tiny)
    sscale = np.maximum(np.linalg.norm(S, axis=0), tiny)
    Qh = Q / qscale
    Sh = S / sscale[None, :]
    features = np.concatenate(
        [
            Qh.real.ravel(),
            Qh.imag.ravel(),
            Sh.real.ravel(),
            Sh.imag.ravel(),
            [np.log(qscale)],
            np.log(sscale),
        ]
    )

    # Cheap physics-consistent initial coefficient estimate. The network learns
    # only the correction from this Jacobi least-squares guess.
    diagonal = np.real(np.diag(Q)).copy()
    diagonal_scale = max(float(np.max(np.abs(diagonal))), tiny)
    diagonal = np.where(diagonal > 1e-12 * diagonal_scale, diagonal, diagonal_scale)
    base = S / diagonal[:, None]
    coefficient_scale = sscale / qscale
    baseline = np.concatenate([base.real.T, base.imag.T], axis=1)

    Qr, Qi = Q.real, Q.imag
    qblock = np.block([[Qr, -Qi], [Qi, Qr]])
    sreal = np.concatenate([S.real.T, S.imag.T], axis=1)
    norm2 = np.sum(np.abs(B) ** 2, axis=0).real
    return tuple(
        np.asarray(x, np.float64)
        for x in (features, baseline, coefficient_scale, qblock, sreal, norm2)
    )


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
                data["features"],
                data["baseline"],
                data["coefficient_scale"],
                data["residual_gram"],
                data["residual_linear"],
                data["rhs_norm2"],
                data["split"],
                int(data["reduced_rank"]),
                int(data["n_rhs"]),
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
        features, base, scale, Q, S, N, split, V.shape[1], base.shape[1]
    )


__all__ = ["MaxwellOperatorDataset", "generate_operator_dataset", "operator_encoding"]
