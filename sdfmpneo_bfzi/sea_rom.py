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
        np.einsum("nmc,mr->nrc", reference.integration_modes, reduction),
        reference.field_softening,
        reference.raw_basis_count,
        reference.inductance_kernel,
    )



def restrict_seawater_response(
    reference: SeawaterResponse, raw_indices,
) -> SeawaterResponse:
    """Resolve a nested raw-mode subspace on exactly the parent cubature.

    An enriched teacher already contains the Galerkin R/L/K matrices for
    *every* retained trial direction. Restricting those matrices to a
    selected nested subset is an exact discrete Galerkin operation and
    needs no second O(Q**2) Green assembly. It is only possible when
    the parent's resistance whitening has not discarded any of the
    requested raw directions; fail explicitly rather than silently
    presenting a rank-truncated subset as a reference solution.

    This is a diagnostic/projection utility, not an independently
    trained neural surrogate.
    """
    indices = np.asarray(raw_indices)
    n_raw = reference.basis.size
    if (indices.ndim != 1 or indices.size == 0
            or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= n_raw)
            or len(np.unique(indices)) != len(indices)):
        raise ValueError("raw_indices must be unique valid mode indices")
    selection = np.eye(n_raw)[:, indices]
    # The parent's whitened current modes are raw_modes @ transform.
    # Translate the selected raw trial functions back into this space.
    coordinates, _, _, _ = np.linalg.lstsq(
        reference.transform, selection, rcond=None,
    )
    if not np.allclose(
        reference.transform @ coordinates, selection,
        rtol=1e-8, atol=1e-9,
    ):
        raise ValueError(
            "requested raw subspace includes discarded resistance modes; "
            "refine cubature or lower resistance_rank_tolerance"
        )
    R = coordinates.T @ reference.resistance @ coordinates
    L = coordinates.T @ reference.inductance @ coordinates
    K = coordinates.T @ reference.coupling
    R = 0.5 * (R + R.T)
    L = 0.5 * (L + L.T)
    # Resistive normalization improves conditioning of independently
    # requested subspaces even if the parent whitening is well conditioned.
    evals, directions = eigh(R, check_finite=True)
    if np.min(evals) <= 0:
        raise ValueError("selected subspace has nonpositive resistance")
    normalization = directions / np.sqrt(evals)[None, :]
    R = normalization.T @ R @ normalization
    L = normalization.T @ L @ normalization
    K = normalization.T @ K
    R = 0.5 * (R + R.T)
    L = 0.5 * (L + L.T)
    omega = reference.sources.omega
    z_modes = solve(
        R + 1j * omega * L, -1j * omega * K,
        assume_a="gen", check_finite=True,
    )
    z_sea = 1j * omega * K.T @ z_modes
    z_sea = 0.5 * (z_sea + z_sea.T)
    projection = coordinates @ normalization
    selection_transform = selection @ normalization
    fields = np.einsum(
        "nmc,mr->nrc", reference.integration_modes,
        projection, optimize=True,
    )
    return SeawaterResponse(
        reference.sources, reference.basis, selection_transform,
        R, L, K, z_modes, z_sea,
        reference.integration_points, reference.integration_weights,
        fields, reference.field_softening, len(indices),
        reference.inductance_kernel,
    )
