"""Exact current-quadratic Joule tensors for reduced electromagnetic models.

For a fixed thermal/geometry state, the reduced electromagnetic solve is linear
in the real operating parameters.  With

    b(u) = b0 + B u = B_aug zeta,   zeta = [1, u]^T,

the reduced electromagnetic state is

    x_r(u) = C zeta,

and every modal Joule source is exactly

    q_j(u) = zeta^T G_j zeta,

where G_j is the real symmetric part of C^H H_{r,j} C.  No positivity is
imposed on individual modal G_j because a signed thermal test/mode can produce a
negative modal projection even though the underlying physical Joule density is
non-negative.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp


def _validated_state(em_model, state) -> np.ndarray:
    value = np.asarray(state, dtype=float)
    expected = int(em_model.problem.n_thermal)
    if value.shape != (expected,):
        raise ValueError("thermal state dimension mismatch")
    if np.any(~np.isfinite(value)):
        raise ValueError("thermal state must be finite")
    return value


def _validated_rhs_map(em_model, rhs_map):
    if int(rhs_map.n_em) != int(em_model.problem.n_em):
        raise ValueError("rhs map/electromagnetic dimension mismatch")
    return rhs_map


def augmented_operating_vector(operating, n_operating: int | None = None) -> np.ndarray:
    """Return ``zeta=[1,u]`` for a real operating vector."""
    u = np.asarray(operating, dtype=float)
    if u.ndim != 1:
        raise ValueError("operating parameters must be one-dimensional")
    if n_operating is not None and u.shape != (int(n_operating),):
        raise ValueError("operating parameter dimension mismatch")
    if np.any(~np.isfinite(u)):
        raise ValueError("operating parameters must be finite")
    return np.concatenate([np.ones(1, dtype=float), u])


def _loss_operator(problem, mode: int, state: np.ndarray):
    sparse = getattr(problem, "loss_operator_sparse", None)
    if sparse is not None:
        return sp.csr_matrix(sparse(int(mode), state), dtype=complex)
    dense = getattr(problem, "loss_operator", None)
    if dense is None:
        raise TypeError("problem must provide loss_operator_sparse or loss_operator")
    return np.asarray(dense(int(mode), state), dtype=complex)


def quadratic_joule_tensor(em_model, state, rhs_map) -> np.ndarray:
    """Construct the exact reduced-model current-quadratic Joule tensor.

    Parameters
    ----------
    em_model:
        Reduced electromagnetic model with ``V``, ``operator_reduced`` and a
        problem exposing modal loss operators.
    state:
        Thermal reduced coordinates.
    rhs_map:
        Affine real-operating excitation map ``b(u)=b0+B u``.

    Returns
    -------
    numpy.ndarray
        Real array with shape ``(n_thermal, 1+n_operating, 1+n_operating)``.
        Each slice is symmetric and reconstructs the modal heat source exactly
        for the same reduced electromagnetic model up to floating-point error.
    """
    a = _validated_state(em_model, state)
    rhs_map = _validated_rhs_map(em_model, rhs_map)

    V = np.asarray(em_model.V, dtype=complex)
    if V.ndim != 2 or V.shape[0] != int(em_model.problem.n_em):
        raise ValueError("invalid electromagnetic reduction basis")

    # One augmented excitation basis covers the entire real operating space.
    B_aug = np.column_stack(
        [np.asarray(rhs_map.offset, dtype=complex), np.asarray(rhs_map.matrix, dtype=complex)]
    )
    reduced_rhs = V.conj().T @ B_aug
    reduced_operator = np.asarray(em_model.operator_reduced(a), dtype=complex)
    coefficients = scipy.linalg.solve(
        reduced_operator,
        reduced_rhs,
        assume_a="gen",
        check_finite=True,
    )

    p = B_aug.shape[1]
    n_thermal = int(em_model.problem.n_thermal)
    tensor = np.empty((n_thermal, p, p), dtype=float)

    for mode in range(n_thermal):
        H = _loss_operator(em_model.problem, mode, a)
        # Work in the reduced electromagnetic space.  This is algebraically
        # identical to forming X=V*C and X^H H X, but avoids a full-state matrix.
        reduced_loss = V.conj().T @ (H @ V)
        complex_form = coefficients.conj().T @ reduced_loss @ coefficients
        # For real zeta only the real symmetric part contributes to
        # real(zeta^T complex_form zeta).  Symmetrizing also removes harmless
        # roundoff-level anti-symmetry without changing the represented heat.
        real_form = np.real(complex_form)
        tensor[mode] = 0.5 * (real_form + real_form.T)

    return tensor


def quadratic_heat_source(tensor, operating) -> np.ndarray:
    """Evaluate modal heat from one exact/learned quadratic tensor."""
    G = np.asarray(tensor, dtype=float)
    if G.ndim != 3 or G.shape[1] != G.shape[2] or G.shape[1] < 1:
        raise ValueError("tensor must have shape (n_thermal,p,p)")
    if np.any(~np.isfinite(G)):
        raise ValueError("quadratic tensor must be finite")
    zeta = augmented_operating_vector(operating, G.shape[1] - 1)
    return np.einsum("i,kij,j->k", zeta, G, zeta, optimize=True)


def quadratic_heat_source_batch(tensor, operating) -> np.ndarray:
    """Evaluate one quadratic tensor for a batch of real operating vectors."""
    G = np.asarray(tensor, dtype=float)
    U = np.asarray(operating, dtype=float)
    if G.ndim != 3 or G.shape[1] != G.shape[2] or G.shape[1] < 1:
        raise ValueError("tensor must have shape (n_thermal,p,p)")
    if U.ndim != 2 or U.shape[1] != G.shape[1] - 1:
        raise ValueError("operating batch dimension mismatch")
    if np.any(~np.isfinite(G)) or np.any(~np.isfinite(U)):
        raise ValueError("quadratic tensor and operating batch must be finite")
    zeta = np.column_stack([np.ones(len(U), dtype=float), U])
    return np.einsum("bi,kij,bj->bk", zeta, G, zeta, optimize=True)


__all__ = [
    "augmented_operating_vector",
    "quadratic_heat_source",
    "quadratic_heat_source_batch",
    "quadratic_joule_tensor",
]
