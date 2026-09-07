from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .realization import AnalyticRealization


@dataclass(frozen=True)
class RealizationCompressionReport:
    """Numerical redundancy report for an analytic realization.

    This is deliberately a diagnostic layer: it identifies uncontrollable or
    unobservable state directions. It does not silently remove states unless a
    caller supplies a tolerance appropriate for a certified approximation.
    """

    original_dimension: int
    controllability_rank: int
    observability_rank: int
    redundant_dimension: int

    @property
    def is_minimal(self) -> bool:
        return self.redundant_dimension == 0


def _rank(matrix: np.ndarray, rtol: float) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0:
        return 0
    threshold = float(rtol) * max(1.0, float(singular[0]))
    return int(np.count_nonzero(singular > threshold))


def controllability_matrix(realization: AnalyticRealization) -> np.ndarray:
    """Construct the finite controllability Krylov matrix."""

    n = realization.dimension
    blocks = []
    vector = realization.b
    for _ in range(n):
        blocks.append(vector)
        vector = realization.A @ vector
    return np.column_stack(blocks)


def observability_matrix(realization: AnalyticRealization) -> np.ndarray:
    """Construct the finite observability Krylov matrix."""

    n = realization.dimension
    blocks = []
    row = realization.c
    for _ in range(n):
        blocks.append(row)
        row = row @ realization.A
    return np.vstack(blocks)


def analyze_realization_redundancy(
    realization: AnalyticRealization,
    *,
    rtol: float = 1e-12,
) -> RealizationCompressionReport:
    """Report removable realization dimensions without changing the model.

    The report enables later certified minimal realization algorithms while
    preserving the current exact realization semantics.
    """

    if rtol <= 0:
        raise ValueError("rtol must be positive")
    ctrb = controllability_matrix(realization)
    obsv = observability_matrix(realization)
    c_rank = _rank(ctrb, rtol)
    o_rank = _rank(obsv, rtol)
    n = realization.dimension
    return RealizationCompressionReport(
        original_dimension=n,
        controllability_rank=c_rank,
        observability_rank=o_rank,
        # The effective dimension is the rank of the Hankel map O C, not
        # min(rank(O), rank(C)): reachable directions can all be unobservable.
        redundant_dimension=max(0, n - _rank(obsv @ ctrb, rtol)),
    )
