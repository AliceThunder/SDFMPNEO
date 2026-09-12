"""Exact batched projection of Joule heat onto retained thermal modes.

The training hot path has two exact implementations:

* a generic full-state path for arbitrary reduced EM models; and
* a fused tetrahedral reduced path that reuses the same certified conductivity
  polynomial family for both the reduced EM operator and the Joule projection.

The fused path avoids reconstructing the full electromagnetic coordinate vector,
avoids applying the full electric extraction matrix, and most importantly avoids
building the nonlinear conductivity polynomials twice for one physical residual.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg

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


def _modal_heat_from_edge_field(problem, edge_field, conductivity_polynomials) -> np.ndarray:
    """Project exact Joule heat once the electric edge field is already known."""
    coefficients, edge_ids, volumes, _ = _geometry_cache(problem)
    edge_field = np.asarray(edge_field, dtype=complex).reshape(-1)
    if edge_field.shape != (problem.mesh.n_edges,):
        raise ValueError("electric edge field dimension mismatch")

    local_edge = edge_field[edge_ids]
    C = np.einsum("qp,qpic->qic", local_edge, coefficients, optimize=True)
    gram = np.real(np.einsum("qic,qjc->qij", np.conj(C), C, optimize=True))

    moments = np.zeros((problem.mesh.n_tetrahedra, 4), dtype=float)
    for q, poly in enumerate(conductivity_polynomials):
        if not poly:
            continue
        tensor = _third_moment_tensor(float(volumes[q]), poly)
        moments[q] = np.einsum("ab,lab->l", gram[q], tensor, optimize=True)

    tests = np.asarray(problem.thermal_test_local, dtype=float)
    if tests.shape != (problem.n_thermal, problem.mesh.n_tetrahedra, 4):
        raise ValueError("thermal test-mode shape mismatch")
    return 0.5 * np.einsum("rqi,qi->r", tests, moments, optimize=True)


def exact_modal_heat_source(problem, electromagnetic_state, thermal_state) -> np.ndarray:
    """Evaluate every modal Joule source exactly without per-mode FE assembly."""
    if not _supported(problem):
        raise TypeError("problem does not expose the nonlinear tetrahedral modal-heat interface")

    state = np.asarray(thermal_state, dtype=float)
    x = np.asarray(electromagnetic_state, dtype=complex)
    if state.shape != (problem.n_thermal,):
        raise ValueError("thermal state dimension mismatch")
    if x.shape != (problem.n_em,):
        raise ValueError("electromagnetic state dimension mismatch")

    _, _, _, extraction = _geometry_cache(problem)
    conductivity_polynomials, _ = problem._weighted_polynomials(state)
    edge_field = np.asarray(extraction @ x, dtype=complex).reshape(-1)
    return _modal_heat_from_edge_field(problem, edge_field, conductivity_polynomials)


def _can_fuse_reduced_modal_heat(em_model) -> bool:
    problem = getattr(em_model, "problem", None)
    return bool(
        problem is not None
        and _supported(problem)
        and getattr(em_model, "_direct_reduced", False)
        and getattr(em_model, "_conductivity_reduced", None) is not None
        and getattr(em_model, "_loss_reduced_assembler", None) is not None
        and getattr(em_model, "_magnetic_reduced", None) is not None
    )


def _fused_reduced_modal_heat_source(em_model, thermal_state, rhs) -> np.ndarray:
    """Solve the reduced EM system and project Joule heat in one constitutive pass.

    ``SparseEnergyReducedEMModel`` already stores the electric reduced basis on
    mesh edges.  Reusing it here gives exactly

        E_edge = electric_extraction @ V @ c

    without constructing ``V @ c`` in the full EM coordinate space.  The same
    conductivity polynomials are also used to assemble ``V^H A(a) V`` and the
    Joule moments, so the nonlinear constitutive expansion is performed once.
    """
    problem = em_model.problem
    state = np.asarray(thermal_state, dtype=float)
    source = np.asarray(rhs, dtype=complex)
    if state.shape != (problem.n_thermal,):
        raise ValueError("thermal state dimension mismatch")
    if source.shape != (problem.n_em,):
        raise ValueError("rhs dimension mismatch")

    conductivity_polynomials, _ = problem._weighted_polynomials(state)
    conductivity = em_model._conductivity_reduced.assemble(conductivity_polynomials)
    reduced_operator = (
        np.asarray(em_model._magnetic_reduced, dtype=complex)
        + 1j * float(problem.omega) * np.asarray(conductivity, dtype=complex)
    )
    reduced_rhs = em_model.rhs_reduced(source)
    coefficients = scipy.linalg.solve(
        reduced_operator,
        reduced_rhs,
        assume_a="gen",
        check_finite=False,
    )
    reduced_edge_fields = np.asarray(
        em_model._loss_reduced_assembler.fields,
        dtype=complex,
    )
    edge_field = reduced_edge_fields @ coefficients
    return _modal_heat_from_edge_field(
        problem,
        edge_field,
        conductivity_polynomials,
    )


def heat_source_for_reduced_model(em_model, thermal_state, rhs) -> np.ndarray:
    """Fast exact heat source with a fused UWPT reduced-coordinate path."""
    problem = em_model.problem
    if _can_fuse_reduced_modal_heat(em_model):
        return _fused_reduced_modal_heat_source(em_model, thermal_state, rhs)
    if _supported(problem):
        x = em_model.state_for_rhs(thermal_state, rhs)
        return exact_modal_heat_source(problem, x, thermal_state)
    return np.asarray(em_model.heat_source_for_rhs(thermal_state, rhs), dtype=float)


__all__ = [
    "exact_modal_heat_source",
    "heat_source_for_reduced_model",
]
