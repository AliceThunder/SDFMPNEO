"""Deterministic structure-preserving BFZI seawater reduced subspace.

This is an offline response compression step, analogous to vNext's
separation of REFERENCE from FAST artifacts. It intentionally consumes a
teacher result and therefore is NOT yet an independently deployable online
geometry-conditional neural surrogate.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, solve

from .sea_reference import SeawaterResponse


def compress_seawater_response(
    reference: SeawaterResponse, rank: int = 2,
) -> SeawaterResponse:
    """Galerkin projection of the shared R/L/K system onto a compact real basis.

    Uses snapshots of both unit-port complex response vectors and retains
    their leading real principal directions. The same projection is applied
    to current-mode decoding and integration nodes, preserving power closure.
    """
    n = reference.resistance.shape[0]
    if not isinstance(rank, int) or not 1 <= rank <= n:
        raise ValueError("rank must be between 1 and number of reference modes")
    snapshots = np.column_stack((
        reference.unit_current_modes.real,
        reference.unit_current_modes.imag,
    ))
    # The full R is close to I after the reference whitening, but use its
    # actual value so this remains well-defined after future refactoring.
    evals, evecs = eigh(reference.resistance)
    if np.min(evals) <= 0:
        raise ValueError("reference resistance matrix is not positive definite")
    root = (evecs * np.sqrt(evals)[None, :]) @ evecs.T
    # Left singular vectors of the weighted snapshots are the physical
    # reduced directions; do not use right singular vectors (port-space).
    weighted_snapshots = root @ snapshots
    U, _, _ = np.linalg.svd(weighted_snapshots, full_matrices=True)
    reduction = np.linalg.solve(root, U[:, :rank])
    R = reduction.T @ reference.resistance @ reduction
    L = reduction.T @ reference.inductance @ reduction
    K = reduction.T @ reference.coupling
    R = 0.5 * (R + R.T)
    L = 0.5 * (L + L.T)
    coefficients = solve(R + 1j * reference.sources.omega * L,
                         -1j * reference.sources.omega * K,
                         assume_a="gen", check_finite=True)
    z = 1j * reference.sources.omega * K.T @ coefficients
    z = 0.5 * (z + z.T)
    return SeawaterResponse(
        reference.sources,
        reference.basis,
        reference.transform @ reduction,
        R, L, K, coefficients, z,
        reference.integration_points,
        reference.integration_weights,
        reference.integration_modes @ reduction,
        reference.field_softening,
        reference.raw_basis_count,
    )
