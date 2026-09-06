from __future__ import annotations

import inspect
from typing import Callable, Protocol

import numpy as np
import scipy.sparse as sp

from .certified_riesz import CertifiedRieszActionResult, RieszNormDecision
from .sparse_solver import ApsiEnergyMetric


class CertifiedRieszAction(Protocol):
    """Certificate-preserving action for the physical Riesz inverse.

    Implementations may be direct or iterative.  The reduced-order method only
    consumes the returned enclosure and never assumes that H^{-1} was applied
    exactly.
    """

    def solve(
        self,
        rhs: np.ndarray,
        *,
        requested_energy_action_error: float,
    ) -> CertifiedRieszActionResult: ...

    def decide_dual_norm(
        self,
        rhs: np.ndarray,
        *,
        threshold: float,
    ) -> RieszNormDecision: ...


RieszActionFactory = Callable[..., CertifiedRieszAction]


def instantiate_riesz_action(
    factory: RieszActionFactory,
    H: sp.spmatrix,
    *,
    state: np.ndarray | None = None,
) -> CertifiedRieszAction:
    """Instantiate a Riesz action while preserving legacy ``factory(H)`` callables.

    Production state-dependent auxiliary spaces may declare a keyword-only
    ``state`` parameter.  Legacy/reference factories remain valid without that
    parameter.  Signature inspection is only an API compatibility operation; it
    does not affect any physical or numerical certificate.
    """

    signature = inspect.signature(factory)
    parameters = signature.parameters.values()
    accepts_state = "state" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if accepts_state:
        return factory(H, state=state)
    return factory(H)


class SparseLUReferenceRieszAction:
    """Deterministic sparse-LU reference implementation of CertifiedRieszAction.

    This is the correctness backend retained from the pre-0.8 implementation.
    It is deliberately behind the generic action protocol so production code can
    replace it by a scalable certified auxiliary-space action without changing
    the reduction theorem or its acceptance criterion.
    """

    def __init__(self, H: sp.spmatrix) -> None:
        self.H = sp.csr_matrix(H, dtype=complex)
        self.n = self.H.shape[0]
        if self.H.shape != (self.n, self.n):
            raise ValueError("H must be square")
        self.energy = ApsiEnergyMetric(self.H)

    def _certificate(
        self,
        rhs: np.ndarray,
        x: np.ndarray,
        iterations: int,
        requested: float | None,
    ) -> CertifiedRieszActionResult:
        residual = np.asarray(rhs, dtype=complex) - self.H @ x
        delta = float(self.energy.dual_norm(residual))
        xnorm = float(self.energy.norm(x))
        lower = float(np.nextafter(max(0.0, xnorm - delta), 0.0))
        upper = float(np.nextafter(xnorm + delta, np.inf))
        return CertifiedRieszActionResult(
            vector=np.asarray(x, dtype=complex).copy(),
            residual=np.asarray(residual, dtype=complex).copy(),
            iterations=int(iterations),
            energy_action_error_bound=delta,
            approximate_vector_energy_norm=xnorm,
            dual_norm_lower_bound=lower,
            dual_norm_upper_bound=upper,
            requested_energy_action_error=requested,
            meets_requested_energy_action_error=bool(
                requested is not None and delta <= requested
            ),
        )

    def _initial(self, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = np.asarray(rhs, dtype=complex)
        if b.shape != (self.n,):
            raise ValueError("Riesz rhs dimension mismatch")
        return b, self.energy.solve(b)

    def _refine(self, rhs: np.ndarray, x: np.ndarray) -> np.ndarray:
        residual = rhs - self.H @ x
        return x + self.energy.solve(residual)

    def solve(
        self,
        rhs: np.ndarray,
        *,
        requested_energy_action_error: float,
    ) -> CertifiedRieszActionResult:
        requested = float(requested_energy_action_error)
        if requested <= 0.0:
            raise ValueError("requested_energy_action_error must be positive")
        b, x = self._initial(rhs)
        result = self._certificate(b, x, 0, requested)
        if result.meets_requested_energy_action_error:
            return result
        for iteration in range(1, self.n + 1):
            x = self._refine(b, x)
            result = self._certificate(b, x, iteration, requested)
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
        b, x = self._initial(rhs)
        result = self._certificate(b, x, 0, None)
        if result.dual_norm_upper_bound <= target:
            return RieszNormDecision("below", target, result)
        if result.dual_norm_lower_bound > target:
            return RieszNormDecision("above", target, result)
        for iteration in range(1, self.n + 1):
            x = self._refine(b, x)
            result = self._certificate(b, x, iteration, None)
            if result.dual_norm_upper_bound <= target:
                return RieszNormDecision("below", target, result)
            if result.dual_norm_lower_bound > target:
                return RieszNormDecision("above", target, result)
        return RieszNormDecision("indeterminate", target, result)
