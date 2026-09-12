"""Exact current-quadratic Joule tensors for reduced electromagnetic models.

For a fixed thermal/geometry state, the reduced electromagnetic solve is linear
in the real operating parameters. With

    b(u) = b0 + B u = B_aug zeta,   zeta = [1, u]^T,

the electromagnetic state is

    x(u) = X zeta,

and every modal Joule source is exactly

    q_j(u) = zeta^T G_j zeta.

No positivity is imposed on individual modal G_j because a signed thermal
mode/test can produce a negative modal projection even though the physical
Joule density itself is non-negative.
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

    One reduced multi-RHS solve covers the whole real operating space. When the
    number of augmented excitations is smaller than the EM ROM rank, modal loss
    contractions are evaluated on the solved response matrix ``X=V*C`` rather
    than assembling every full reduced loss matrix ``V^H H_j V``. This changes
    cost, not mathematics.
    """
    a = _validated_state(em_model, state)
    rhs_map = _validated_rhs_map(em_model, rhs_map)

    V = np.asarray(em_model.V, dtype=complex)
    if V.ndim != 2 or V.shape[0] != int(em_model.problem.n_em):
        raise ValueError("invalid electromagnetic reduction basis")

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
    use_response_space = p <= V.shape[1]
    response_matrix = V @ coefficients if use_response_space else None

    for mode in range(n_thermal):
        H = _loss_operator(em_model.problem, mode, a)
        if use_response_space:
            complex_form = response_matrix.conj().T @ (H @ response_matrix)
        else:
            reduced_loss = V.conj().T @ (H @ V)
            complex_form = coefficients.conj().T @ reduced_loss @ coefficients
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
