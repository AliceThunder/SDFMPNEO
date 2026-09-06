from __future__ import annotations

from itertools import product
from math import factorial
from typing import Sequence

import numpy as np
import scipy.sparse as sp

from .tetra3d import TetrahedralComplex3D, _barycentric_gradients


def _simplex_monomial_integral(volume: float, barycentric_indices: Sequence[int]) -> float:
    """Exact integral of a barycentric monomial over one tetrahedron.

    For a tetrahedron T and counts alpha_i,

        integral_T prod_i lambda_i**alpha_i dV
        = 6 |T| prod_i alpha_i! / (3 + sum_i alpha_i)!.

    This is algebraic and introduces no quadrature rule or tuning parameter.
    """

    counts = np.zeros(4, dtype=int)
    for index in barycentric_indices:
        if not 0 <= int(index) < 4:
            raise ValueError("barycentric index out of range")
        counts[int(index)] += 1
    degree = int(np.sum(counts))
    numerator = 6.0 * float(volume)
    for count in counts:
        numerator *= factorial(int(count))
    return numerator / factorial(3 + degree)


def _weighted_lambda_pair_integral(
    volume: float,
    first: int,
    second: int,
    factor_values: np.ndarray,
) -> float:
    """Integrate lambda_first lambda_second times products of P1 factors."""

    factors = np.asarray(factor_values, dtype=float)
    if factors.ndim != 2 or factors.shape[1] != 4:
        raise ValueError("factor_values must have shape (n_factors,4)")
    if factors.shape[0] == 0:
        return _simplex_monomial_integral(volume, (first, second))

    total = 0.0
    for choices in product(range(4), repeat=factors.shape[0]):
        coefficient = 1.0
        for factor_index, local_vertex in enumerate(choices):
            coefficient *= factors[factor_index, local_vertex]
        if coefficient != 0.0:
            total += coefficient * _simplex_monomial_integral(
                volume,
                (first, second, *choices),
            )
    return total


def assemble_weighted_nedelec_mass(
    mesh: TetrahedralComplex3D,
    *,
    scale_tetra: np.ndarray,
    p1_factors: np.ndarray | None = None,
) -> sp.csr_matrix:
    """Assemble an exact first-order Nedelec mass with polynomial P1 weights.

    The scalar coefficient on tetrahedron q is

        scale_q * prod_f (sum_i factor[f,q,i] lambda_i).

    Zero factors therefore give a piecewise-constant coefficient, one factor
    gives an exact P1-weighted mass, and two factors give the exact P1 x P1
    weight required by the temperature derivative of projected Joule heating.
    The integration is performed from the closed simplex monomial formula.
    """

    scale = np.asarray(scale_tetra, dtype=float)
    if scale.shape != (mesh.n_tetrahedra,):
        raise ValueError("scale_tetra must have shape (n_tetrahedra,)")

    if p1_factors is None:
        factors = np.empty((0, mesh.n_tetrahedra, 4), dtype=float)
    else:
        factors = np.asarray(p1_factors, dtype=float)
        if factors.ndim != 3 or factors.shape[1:] != (mesh.n_tetrahedra, 4):
            raise ValueError("p1_factors must have shape (n_factors,n_tetrahedra,4)")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for q, tet in enumerate(mesh.tetrahedra):
        if scale[q] == 0.0:
            continue
        volume = float(mesh.volumes[q])
        gradients = _barycentric_gradients(mesh.vertices, tet)
        local_pairs = mesh._local_edge_vertex_indices(tet)
        local_factors = factors[:, q, :]
        local = np.zeros((6, 6), dtype=float)

        for p, (i, j) in enumerate(local_pairs):
            gi, gj = gradients[i], gradients[j]
            for r, (k, ell) in enumerate(local_pairs):
                gk, gl = gradients[k], gradients[ell]
                value = (
                    np.dot(gj, gl)
                    * _weighted_lambda_pair_integral(volume, i, k, local_factors)
                    - np.dot(gj, gk)
                    * _weighted_lambda_pair_integral(volume, i, ell, local_factors)
                    - np.dot(gi, gl)
                    * _weighted_lambda_pair_integral(volume, j, k, local_factors)
                    + np.dot(gi, gk)
                    * _weighted_lambda_pair_integral(volume, j, ell, local_factors)
                )
                local[p, r] = scale[q] * value

        global_edges = mesh.tet_edge_indices[q]
        for p, ep in enumerate(global_edges):
            for r, er in enumerate(global_edges):
                value = float(local[p, r])
                if value != 0.0:
                    rows.append(int(ep))
                    cols.append(int(er))
                    data.append(value)

    matrix = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(mesh.n_edges, mesh.n_edges),
    ).tocsr()
    matrix.sum_duplicates()
    return matrix
