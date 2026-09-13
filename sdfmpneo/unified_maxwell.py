"""Neural initial guess plus true sparse Maxwell residual correction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse.linalg as spla

from .unified_basis import safe_diag
from .unified_dataset import operator_encoding


@dataclass(frozen=True)
class MaxwellSolveReport:
    initial_relative_residual: tuple
    final_relative_residual: tuple
    correction_iterations: tuple
    direct_corrections: int


class NeuralMaxwellAccelerator:
    def __init__(self, network, basis, *, residual_tolerance=1e-7, max_iterations=200):
        self.network = network
        self.basis = np.asarray(basis, complex)
        self.residual_tolerance = float(residual_tolerance)
        self.max_iterations = int(max_iterations)
        if self.basis.ndim != 2 or self.basis.shape[1] < 1 or np.any(~np.isfinite(self.basis)):
            raise ValueError("basis must be a finite nonempty matrix")
        if self.residual_tolerance <= 0 or self.max_iterations < 1:
            raise ValueError("invalid Maxwell correction settings")

    def guess(self, A, B):
        """Return the neural reduced-space guess; non-finite NN output falls back to zero."""
        import torch

        features, base, scale, _, _, _ = operator_encoding(A, B, self.basis)
        if np.any(~np.isfinite(features)) or np.any(~np.isfinite(base)) or np.any(~np.isfinite(scale)):
            raise FloatingPointError("physical Maxwell operator encoding is non-finite")
        parameter = next(self.network.parameters())
        x = torch.as_tensor(features, dtype=parameter.dtype, device=parameter.device).unsqueeze(0)
        with torch.no_grad():
            y = self.network(x).detach().cpu().double().numpy()
        expected = B.shape[1] * 2 * self.basis.shape[1]
        if y.size != expected:
            raise ValueError("neural Maxwell output dimension does not match the physical reduced space")
        y = y.reshape((B.shape[1], 2 * self.basis.shape[1]))
        if np.any(~np.isfinite(y)):
            return np.zeros((A.shape[0], B.shape[1]), complex)
        coefficients_real = base + scale[:, None] * y
        if np.any(~np.isfinite(coefficients_real)):
            return np.zeros((A.shape[0], B.shape[1]), complex)
        rank = self.basis.shape[1]
        coefficients = (
            coefficients_real[:, :rank] + 1j * coefficients_real[:, rank:]
        ).T
        X = self.basis @ coefficients
        if np.any(~np.isfinite(X)):
            return np.zeros((A.shape[0], B.shape[1]), complex)
        return X

    def solve(self, A, B):
        B = np.asarray(B, complex)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Maxwell operator must be square")
        if self.basis.shape[0] != A.shape[0] or B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError("Maxwell operator, basis, and RHS dimensions do not match")
        if np.any(~np.isfinite(A.data)) or np.any(~np.isfinite(B)):
            raise FloatingPointError("physical Maxwell system is non-finite")

        X = self.guess(A, B)
        denominator = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
        residual = B - A @ X
        initial = np.linalg.norm(residual, axis=0) / denominator
        if np.any(~np.isfinite(initial)):
            X = np.zeros_like(B)
            residual = B.copy()
            initial = np.linalg.norm(residual, axis=0) / denominator

        iterations = []
        failed = []
        diagonal = safe_diag(A)
        preconditioner = spla.LinearOperator(
            A.shape,
            matvec=lambda vector: vector / diagonal,
            dtype=complex,
        )

        for port in range(B.shape[1]):
            if initial[port] <= self.residual_tolerance:
                iterations.append(0)
                continue
            counter = [0]

            def callback(_):
                counter[0] += 1

            try:
                correction, info = spla.gmres(
                    A,
                    residual[:, port],
                    M=preconditioner,
                    rtol=self.residual_tolerance * 0.25,
                    atol=0.0,
                    restart=min(60, A.shape[0]),
                    maxiter=self.max_iterations,
                    callback=callback,
                    callback_type="pr_norm",
                )
            except TypeError:
                correction, info = spla.gmres(
                    A,
                    residual[:, port],
                    M=preconditioner,
                    tol=self.residual_tolerance * 0.25,
                    restart=min(60, A.shape[0]),
                    maxiter=self.max_iterations,
                    callback=callback,
                )
            iterations.append(counter[0])
            if np.all(np.isfinite(correction)):
                X[:, port] += correction
            relative = float(np.linalg.norm(B[:, port] - A @ X[:, port]) / denominator[port])
            if info != 0 or not np.isfinite(relative) or relative > self.residual_tolerance:
                failed.append(port)

        direct_corrections = 0
        if failed:
            factorization = spla.splu(A.tocsc())
            for port in failed:
                remaining = B[:, port] - A @ X[:, port]
                X[:, port] += factorization.solve(remaining)
                direct_corrections += 1

        final = np.linalg.norm(B - A @ X, axis=0) / denominator
        allowed = max(5 * self.residual_tolerance, 1e-12)
        if np.any(~np.isfinite(final)) or np.any(final > allowed):
            value = float(np.nanmax(final)) if final.size else float("nan")
            raise RuntimeError(
                f"Maxwell residual correction failed: max relative residual={value:.3e}"
            )
        return X, MaxwellSolveReport(
            tuple(map(float, initial)),
            tuple(map(float, final)),
            tuple(map(int, iterations)),
            direct_corrections,
        )


__all__ = ["MaxwellSolveReport", "NeuralMaxwellAccelerator"]
