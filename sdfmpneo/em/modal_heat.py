"""Exact batched projection of Joule heat onto all retained thermal modes.

This module evaluates the same quantity as ``x^H H_j(a) x`` for every thermal
mode, but avoids assembling one global sparse loss operator per output mode.
The electric Nedelec field is reconstructed once on each tetrahedron, its local
quadratic energy polynomial is integrated against the certified conductivity
polynomial, and the four P1 moments are projected to all thermal test modes in
one dense contraction.
"""
from __future__ import annotations

import numpy as np

from sdfmpneo.spatial.barycentric_polynomial import (
    simplex_barycentric_monomial_integral,
)
from sdfmpneo.spatial.tetra3d import _barycentric_gradients


def _supported(problem) -> bool:
    return all(
        hasattr(problem, name)
        for name in (
            "mesh",
            "thermal_test_local",
            "electric_extraction_sparse",
            "_weighted_polynomials",
        )
    )


def _geometry_cache(problem):
    cached = getattr(problem, "_modal_heat_geometry_cache", None)
    if cached is not None:
        return cached

    mesh = problem.mesh
    coefficients = np.zeros((mesh.n_tetrahedra, 6, 4, 3), dtype=float)
    for q, tet in enumerate(mesh.tetrahedra):
        gradients = _barycentric_gradients(mesh.vertices, tet)
        for p, (i, j) in enumerate(mesh._local_edge_vertex_indices(tet)):
            coefficients[q, p, i] = gradients[j]
            coefficients[q, p, j] = -gradients[i]

    cached = (
        coefficients,
        np.asarray(mesh.tet_edge_indices, dtype=int),
        np.asarray(mesh.volumes, dtype=float),
        problem.electric_extraction_sparse().tocsr(),
    )
    # NonlinearTetrahedralApsiProblem is intentionally mutable for lazily built
    # FE caches. Keeping this geometry-only tensor on the problem avoids
    # recomputing barycentric gradients on every residual evaluation.
    setattr(problem, "_modal_heat_geometry_cache", cached)
    return cached


def _third_moment_tensor(volume: float, poly) -> np.ndarray:
    """Return I[l,a,b] = integral poly * lambda_l * lambda_a * lambda_b."""
    tensor = np.zeros((4, 4, 4), dtype=float)
    if not poly:
        return tensor
    for l in range(4):
        for a in range(4):
            for b in range(4):
                extra = [0, 0, 0, 0]
                extra[l] += 1
                extra[a] += 1
                extra[b] += 1
                value = 0.0
                for powers, coefficient in poly.items():
                    augmented = tuple(int(powers[i] + extra[i]) for i in range(4))
                    value += float(coefficient) * simplex_barycentric_monomial_integral(
                        volume, augmented
                    )
                tensor[l, a, b] = value
    return tensor


def exact_modal_heat_source(problem, electromagnetic_state, thermal_state) -> np.ndarray:
    """Evaluate every modal Joule source exactly without per-mode FE assembly.

    For a tetrahedron, the first-order Nedelec field has the barycentric form

        E(lambda) = sum_i lambda_i C_i.

    Therefore ``|E|^2`` is quadratic in the four barycentric coordinates.  For
    each cell we compute only the four moments

        m_l = integral sigma(T) |E|^2 lambda_l,

    then project them to all thermal P1 test modes at once. This is algebraically
    identical to assembling ``H_j`` and evaluating ``x^H H_j x`` separately.
    """
    if not _supported(problem):
        raise TypeError("problem does not expose the nonlinear tetrahedral modal-heat interface")

    state = np.asarray(thermal_state, dtype=float)
    x = np.asarray(electromagnetic_state, dtype=complex)
    if state.shape != (problem.n_thermal,):
        raise ValueError("thermal state dimension mismatch")
    if x.shape != (problem.n_em,):
        raise ValueError("electromagnetic state dimension mismatch")

    coefficients, edge_ids, volumes, extraction = _geometry_cache(problem)
    conductivity_polynomials, _ = problem._weighted_polynomials(state)
    edge_field = np.asarray(extraction @ x, dtype=complex).reshape(-1)
    local_edge = edge_field[edge_ids]
    # C[q,i,c] is the Cartesian vector multiplying lambda_i in cell q.
    C = np.einsum("qp,qpic->qic", local_edge, coefficients, optimize=True)

    moments = np.zeros((problem.mesh.n_tetrahedra, 4), dtype=float)
    for q, poly in enumerate(conductivity_polynomials):
        if not poly:
            continue
        gram = np.real(
            np.einsum("ic,jc->ij", np.conj(C[q]), C[q], optimize=True)
        )
        tensor = _third_moment_tensor(float(volumes[q]), poly)
        moments[q] = np.einsum("ab,lab->l", gram, tensor, optimize=True)

    tests = np.asarray(problem.thermal_test_local, dtype=float)
    if tests.shape != (problem.n_thermal, problem.mesh.n_tetrahedra, 4):
        raise ValueError("thermal test-mode shape mismatch")
    return 0.5 * np.einsum("rqi,qi->r", tests, moments, optimize=True)


def heat_source_for_reduced_model(em_model, thermal_state, rhs) -> np.ndarray:
    """Fast exact heat source when supported, otherwise use the model fallback."""
    problem = em_model.problem
    if _supported(problem):
        x = em_model.state_for_rhs(thermal_state, rhs)
        return exact_modal_heat_source(problem, x, thermal_state)
    return np.asarray(em_model.heat_source_for_rhs(thermal_state, rhs), dtype=float)


__all__ = ["exact_modal_heat_source", "heat_source_for_reduced_model"]
