"""Full-background neural preconditioner plus residual-corrected FGMRES Maxwell solve."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .unified_neural_operator import neural_correction


@dataclass(frozen=True)
class MaxwellSolveReport:
    initial_relative_residual: tuple
    final_relative_residual: tuple
    correction_iterations: tuple
    restarts: tuple


def _fgmres(A, b, apply_preconditioner, *, tolerance, max_iterations, restart):
    """Right-preconditioned flexible GMRES with an explicit true-residual stop.

    The preconditioner may be nonlinear or change between iterations. Accuracy
    is accepted only from the true sparse residual ``b-Ax``.
    """
    b = np.asarray(b, complex).reshape(-1)
    n = b.size
    x = np.zeros(n, complex)
    denominator = max(float(np.linalg.norm(b)), np.finfo(float).tiny)
    total_iterations = 0
    restart_count = 0
    m = max(1, min(int(restart), int(max_iterations), n))

    while total_iterations < max_iterations:
        r = b - A @ x
        beta = float(np.linalg.norm(r))
        if beta / denominator <= tolerance:
            return x, total_iterations, restart_count
        V = np.zeros((n, m + 1), complex)
        Z = np.zeros((n, m), complex)
        H = np.zeros((m + 1, m), complex)
        V[:, 0] = r / beta
        rhs = np.zeros(m + 1, complex)
        rhs[0] = beta
        best = None

        for j in range(m):
            if total_iterations >= max_iterations:
                break
            z = np.asarray(apply_preconditioner(V[:, j]), complex).reshape(-1)
            if z.shape != (n,) or np.any(~np.isfinite(z)):
                raise FloatingPointError("Maxwell preconditioner produced a non-finite correction")
            Z[:, j] = z
            w = np.asarray(A @ z, complex).reshape(-1)
            for i in range(j + 1):
                H[i, j] = np.vdot(V[:, i], w)
                w -= H[i, j] * V[:, i]
            for i in range(j + 1):
                correction = np.vdot(V[:, i], w)
                H[i, j] += correction
                w -= correction * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > np.finfo(float).tiny:
                V[:, j + 1] = w / H[j + 1, j]

            y = np.linalg.lstsq(H[: j + 2, : j + 1], rhs[: j + 2], rcond=None)[0]
            candidate = x + Z[:, : j + 1] @ y
            total_iterations += 1
            true_relative = float(np.linalg.norm(b - A @ candidate) / denominator)
            best = candidate
            if np.isfinite(true_relative) and true_relative <= tolerance:
                return candidate, total_iterations, restart_count
            if H[j + 1, j] <= np.finfo(float).tiny:
                break

        if best is None or np.any(~np.isfinite(best)):
            break
        x = best
        restart_count += 1

    return x, total_iterations, restart_count


class NeuralMaxwellAccelerator:
    """One learned full-edge-space variable preconditioner used by FGMRES."""

    def __init__(self, network, *, residual_tolerance=1e-7, max_iterations=200, restart=40):
        self.network = network
        self.residual_tolerance = float(residual_tolerance)
        self.max_iterations = int(max_iterations)
        self.restart = int(restart)
        if self.residual_tolerance <= 0 or self.max_iterations < 1 or self.restart < 1:
            raise ValueError("invalid Maxwell correction settings")

    def precondition(self, A, residual):
        R = np.asarray(residual, complex)
        vector = R.ndim == 1
        if vector:
            R = R[:, None]
        correction = neural_correction(self.network, A, R)
        return correction[:, 0] if vector else correction

    def guess(self, A, B):
        B = np.asarray(B, complex)
        X = self.precondition(A, B)
        if np.any(~np.isfinite(X)):
            return np.zeros_like(B)
        return X

    def solve(self, A, B):
        B = np.asarray(B, complex)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("Maxwell operator must be square")
        if B.ndim != 2 or B.shape[0] != A.shape[0]:
            raise ValueError("Maxwell operator and RHS dimensions do not match")
        if np.any(~np.isfinite(A.data)) or np.any(~np.isfinite(B)):
            raise FloatingPointError("physical Maxwell system is non-finite")
        n_edges = getattr(self.network, "n_edges", A.shape[0])
        if int(n_edges) != A.shape[0]:
            raise ValueError("neural edge topology does not match the Maxwell background")

        X = self.guess(A, B)
        denominator = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
        residual = B - A @ X
        initial = np.linalg.norm(residual, axis=0) / denominator
        if np.any(~np.isfinite(initial)):
            X = np.zeros_like(B)
            residual = B.copy()
            initial = np.linalg.norm(residual, axis=0) / denominator

        iterations = []
        restarts = []
        for port in range(B.shape[1]):
            if initial[port] <= self.residual_tolerance:
                iterations.append(0)
                restarts.append(0)
                continue
            correction, count, restart_count = _fgmres(
                A,
                residual[:, port],
                lambda r: self.precondition(A, r),
                tolerance=self.residual_tolerance,
                max_iterations=self.max_iterations,
                restart=self.restart,
            )
            if np.any(~np.isfinite(correction)):
                raise FloatingPointError("FGMRES produced a non-finite Maxwell correction")
            X[:, port] += correction
            iterations.append(count)
            restarts.append(restart_count)

        final = np.linalg.norm(B - A @ X, axis=0) / denominator
        if np.any(~np.isfinite(final)) or np.any(final > self.residual_tolerance):
            value = float(np.nanmax(final)) if final.size else float("nan")
            raise RuntimeError(
                f"FGMRES Maxwell solve failed: max relative true residual={value:.3e}, "
                f"target={self.residual_tolerance:.3e}"
            )
        return X, MaxwellSolveReport(
            tuple(map(float, initial)),
            tuple(map(float, final)),
            tuple(map(int, iterations)),
            tuple(map(int, restarts)),
        )


__all__ = ["MaxwellSolveReport", "NeuralMaxwellAccelerator"]
