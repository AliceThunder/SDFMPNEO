from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol
import math

import numpy as np
import scipy.sparse as sp


class CertifiedEnergyPreconditioner(Protocol):
    """Preconditioner contract with a proven spectral-equivalence lower bound.

    The declared constant m must satisfy

        H >= m P

    in the Hermitian positive-definite ordering, where ``solve(rhs)`` applies
    P^{-1}. Consequently

        s^H H^{-1} s <= (1/m) s^H P^{-1} s.

    This is the only property used by the Riesz-action certificate.
    """

    lower_spectral_equivalence_bound: float

    def solve(self, rhs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CertifiedRieszActionResult:
    """Certified enclosure for y=H^{-1}r from an approximate Riesz action."""

    vector: np.ndarray
    residual: np.ndarray
    iterations: int
    energy_action_error_bound: float
    approximate_vector_energy_norm: float
    dual_norm_lower_bound: float
    dual_norm_upper_bound: float
    requested_energy_action_error: float | None
    meets_requested_energy_action_error: bool


@dataclass(frozen=True)
class RieszNormDecision:
    """Threshold decision for ||r||_{H^{-1}} without an arbitrary solve tolerance."""

    relation: Literal["below", "above", "indeterminate"]
    threshold: float
    result: CertifiedRieszActionResult


@dataclass(frozen=True)
class DiagonalGershgorinEnergyPreconditioner:
    """Parameter-free diagonal preconditioner with a certified Gershgorin bound.

    P=diag(H).  For B=P^{-1/2} H P^{-1/2}, Gershgorin gives

        lambda_min(B) >= min_i [1-sum_{j!=i}|B_ij|].

    When the computed lower bound is positive, H >= m P follows. If the
    certificate is non-positive the constructor refuses the preconditioner rather
    than inventing a damping/shift parameter. More powerful auxiliary-space
    preconditioners can implement the same protocol with their own proven m.
    """

    diagonal: np.ndarray
    lower_spectral_equivalence_bound: float

    @classmethod
    def build(cls, H: sp.spmatrix) -> "DiagonalGershgorinEnergyPreconditioner":
        matrix = sp.csr_matrix(H, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            raise ValueError("H must be square")

        asym = matrix - matrix.conj().T
        asym_norm = float(np.sqrt(np.sum(np.abs(asym.data) ** 2))) if asym.nnz else 0.0
        matrix_norm = float(np.sqrt(np.sum(np.abs(matrix.data) ** 2)))
        backward = np.finfo(float).eps * max(1, n) * max(1.0, matrix_norm)
        if asym_norm > backward:
            raise ValueError("H must be Hermitian")

        diag_complex = np.asarray(matrix.diagonal(), dtype=complex)
        diag_imag = float(np.max(np.abs(diag_complex.imag))) if n else 0.0
        diag_scale = float(np.max(np.abs(diag_complex.real))) if n else 0.0
        if diag_imag > np.finfo(float).eps * max(1, n) * max(1.0, diag_scale):
            raise ValueError("H diagonal must be real to backward-error scale")
        diagonal = diag_complex.real.astype(float)
        if np.any(diagonal <= 0.0):
            raise ValueError("H must have a positive diagonal")

        eps = np.finfo(float).eps
        row_lower_bounds = np.empty(n, dtype=float)
        for i in range(n):
            start, stop = matrix.indptr[i], matrix.indptr[i + 1]
            indices = matrix.indices[start:stop]
            values = matrix.data[start:stop]
            terms = []
            for j, value in zip(indices, values):
                j = int(j)
                if j == i:
                    continue
                denom = math.sqrt(float(diagonal[i] * diagonal[j]))
                terms.append(float(abs(value) / denom))

            radius_hat = math.fsum(terms)
            # Per off-diagonal term: product, sqrt, division, magnitude plus the
            # summation contribution.  gamma_k is the standard floating-point
            # model bound; using five operations per term is conservative for
            # this scalar enclosure and contains no fitted safety factor.
            operation_count = 5 * len(terms) + 2
            if operation_count * eps >= 1.0:
                raise FloatingPointError("matrix row is too large for the gamma_k bound")
            gamma = operation_count * eps / (1.0 - operation_count * eps)
            radius_upper = np.nextafter(
                radius_hat / (1.0 - gamma),
                np.inf,
            )
            row_lower_bounds[i] = np.nextafter(1.0 - radius_upper, -np.inf)

        lower = float(np.min(row_lower_bounds)) if n else 0.0
        if lower <= 0.0:
            raise ValueError(
                "diagonal Gershgorin preconditioner has no positive certified spectral-equivalence lower bound"
            )
        return cls(diagonal=diagonal, lower_spectral_equivalence_bound=lower)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        vector = np.asarray(rhs, dtype=complex)
        if vector.shape != self.diagonal.shape:
            raise ValueError("preconditioner rhs dimension mismatch")
        return vector / self.diagonal


class CertifiedPCGRieszAction:
    """PCG Riesz action stopped only by a physical/certificate decision.

    The method solves H y=r.  For an iterate y_h with residual s=r-Hy_h and a
    certified preconditioner H>=mP,

        ||y-y_h||_H^2
        = s^H H^{-1}s
        <= (1/m) s^H P^{-1}s.

    Therefore

        max(0, ||y_h||_H-delta)
        <= ||r||_{H^{-1}}
        <= ||y_h||_H+delta.

    No ``rtol``/``atol`` belongs to the scientific interface.  ``solve`` stops
    when a requested energy-action error is proven; ``decide_dual_norm`` stops
    as soon as the certified interval lies wholly on one side of a threshold.
    The dimension n is the deterministic exact-arithmetic CG work bound.
    """

    def __init__(
        self,
        H: sp.spmatrix,
        preconditioner: CertifiedEnergyPreconditioner,
    ) -> None:
        matrix = sp.csr_matrix(H, dtype=complex)
        n = matrix.shape[0]
        if matrix.shape != (n, n):
            raise ValueError("H must be square")
        m = float(preconditioner.lower_spectral_equivalence_bound)
        if m <= 0.0:
            raise ValueError("preconditioner lower spectral-equivalence bound must be positive")
        self.H = matrix
        self.n = n
        self.preconditioner = preconditioner
        self.m = m

    def _energy_norm(self, vector: np.ndarray) -> float:
        v = np.asarray(vector, dtype=complex)
        q = float(np.real(np.vdot(v, self.H @ v)))
        backward = (
            np.finfo(float).eps
            * max(1, self.n)
            * max(1.0, float(np.linalg.norm(v)) * float(np.linalg.norm(self.H @ v)))
        )
        if q < -backward:
            raise np.linalg.LinAlgError("H is not positive definite in the supplied direction")
        return float(np.sqrt(max(q, 0.0)))

    def _certificate(
        self,
        rhs: np.ndarray,
        x: np.ndarray,
        residual: np.ndarray,
        iterations: int,
        requested: float | None,
    ) -> CertifiedRieszActionResult:
        z = self.preconditioner.solve(residual)
        residual_preconditioned = float(np.real(np.vdot(residual, z)))
        backward = (
            np.finfo(float).eps
            * max(1, self.n)
            * max(1.0, float(np.linalg.norm(residual)) * float(np.linalg.norm(z)))
        )
        if residual_preconditioned < -backward:
            raise np.linalg.LinAlgError("preconditioner is not positive definite in the residual direction")
        residual_preconditioned = max(residual_preconditioned, 0.0)
        delta = np.nextafter(
            np.sqrt(residual_preconditioned / self.m),
            np.inf,
        )
        xnorm = self._energy_norm(x)
        lower = np.nextafter(max(0.0, xnorm - delta), 0.0)
        upper = np.nextafter(xnorm + delta, np.inf)
        meets = requested is not None and delta <= requested
        return CertifiedRieszActionResult(
            vector=np.asarray(x, dtype=complex).copy(),
            residual=np.asarray(residual, dtype=complex).copy(),
            iterations=int(iterations),
            energy_action_error_bound=float(delta),
            approximate_vector_energy_norm=float(xnorm),
            dual_norm_lower_bound=float(lower),
            dual_norm_upper_bound=float(upper),
            requested_energy_action_error=requested,
            meets_requested_energy_action_error=bool(meets),
        )

    def _initial_state(self, rhs: np.ndarray):
        b = np.asarray(rhs, dtype=complex)
        if b.shape != (self.n,):
            raise ValueError("Riesz rhs dimension mismatch")
        x = np.zeros(self.n, dtype=complex)
        residual = b.copy()
        z = self.preconditioner.solve(residual)
        rho = float(np.real(np.vdot(residual, z)))
        if rho < 0.0:
            raise np.linalg.LinAlgError("preconditioner must be positive definite")
        return b, x, residual, z, z.copy(), rho

    def _step(self, x, residual, p, rho):
        Hp = self.H @ p
        denominator = float(np.real(np.vdot(p, Hp)))
        backward = (
            np.finfo(float).eps
            * max(1, self.n)
            * max(1.0, float(np.linalg.norm(p)) * float(np.linalg.norm(Hp)))
        )
        if denominator <= backward:
            raise np.linalg.LinAlgError("PCG direction lost positive H energy")
        alpha = rho / denominator
        x_new = x + alpha * p
        residual_new = residual - alpha * Hp
        z_new = self.preconditioner.solve(residual_new)
        rho_new = float(np.real(np.vdot(residual_new, z_new)))
        if rho_new < -backward:
            raise np.linalg.LinAlgError("preconditioned residual lost positivity")
        rho_new = max(rho_new, 0.0)
        beta = 0.0 if rho == 0.0 else rho_new / rho
        p_new = z_new + beta * p
        return x_new, residual_new, p_new, rho_new

    def solve(
        self,
        rhs: np.ndarray,
        *,
        requested_energy_action_error: float,
    ) -> CertifiedRieszActionResult:
        requested = float(requested_energy_action_error)
        if requested <= 0.0:
            raise ValueError("requested_energy_action_error must be positive")

        b, x, residual, _z, p, rho = self._initial_state(rhs)
        result = self._certificate(b, x, residual, 0, requested)
        if result.meets_requested_energy_action_error:
            return result

        for iteration in range(1, self.n + 1):
            x, residual, p, rho = self._step(x, residual, p, rho)
            result = self._certificate(b, x, residual, iteration, requested)
            if result.meets_requested_energy_action_error:
                return result
        return result

    def decide_dual_norm(
        self,
        rhs: np.ndarray,
        *,
        threshold: float,
    ) -> RieszNormDecision:
        target = float(threshold)
        if target < 0.0:
            raise ValueError("threshold must be non-negative")

        b, x, residual, _z, p, rho = self._initial_state(rhs)
        result = self._certificate(b, x, residual, 0, None)
        if result.dual_norm_upper_bound <= target:
            return RieszNormDecision("below", target, result)
        if result.dual_norm_lower_bound > target:
            return RieszNormDecision("above", target, result)

        for iteration in range(1, self.n + 1):
            x, residual, p, rho = self._step(x, residual, p, rho)
            result = self._certificate(b, x, residual, iteration, None)
            if result.dual_norm_upper_bound <= target:
                return RieszNormDecision("below", target, result)
            if result.dual_norm_lower_bound > target:
                return RieszNormDecision("above", target, result)
        return RieszNormDecision("indeterminate", target, result)
