from __future__ import annotations

from math import factorial
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np
import scipy.sparse as sp

from .tetra3d import TetrahedralComplex3D, _barycentric_gradients

MultiIndex = Tuple[int, int, int, int]
Polynomial = Dict[MultiIndex, float]
_ZERO: MultiIndex = (0, 0, 0, 0)


def polynomial_constant(value: float) -> Polynomial:
    return {} if value == 0.0 else {_ZERO: float(value)}


def polynomial_p1(nodal_values: np.ndarray) -> Polynomial:
    values = np.asarray(nodal_values, dtype=float)
    if values.shape != (4,):
        raise ValueError("nodal_values must have shape (4,)")
    out: Polynomial = {}
    for i, value in enumerate(values):
        if value != 0.0:
            index = [0, 0, 0, 0]
            index[i] = 1
            out[tuple(index)] = float(value)
    return out


def polynomial_add(first: Mapping[MultiIndex, float], second: Mapping[MultiIndex, float]) -> Polynomial:
    out: Polynomial = dict(first)
    for index, value in second.items():
        out[index] = out.get(index, 0.0) + float(value)
        if out[index] == 0.0:
            del out[index]
    return out


def polynomial_scale(poly: Mapping[MultiIndex, float], scale: float) -> Polynomial:
    if scale == 0.0:
        return {}
    return {index: float(scale) * float(value) for index, value in poly.items() if value != 0.0}


def polynomial_multiply(first: Mapping[MultiIndex, float], second: Mapping[MultiIndex, float]) -> Polynomial:
    out: Polynomial = {}
    for a, ca in first.items():
        for b, cb in second.items():
            index = tuple(int(a[i] + b[i]) for i in range(4))
            out[index] = out.get(index, 0.0) + float(ca) * float(cb)
    return {index: value for index, value in out.items() if value != 0.0}


def polynomial_power(poly: Mapping[MultiIndex, float], exponent: int) -> Polynomial:
    if exponent < 0:
        raise ValueError("exponent must be non-negative")
    result = polynomial_constant(1.0)
    if exponent == 0:
        return result
    base = dict(poly)
    n = int(exponent)
    while n:
        if n & 1:
            result = polynomial_multiply(result, base)
        n >>= 1
        if n:
            base = polynomial_multiply(base, base)
    return result


@lru_cache(maxsize=8192)
def _unit_simplex_moment(powers: MultiIndex) -> float:
    degree = int(sum(powers))
    numerator = 6.0
    for power in powers:
        if power < 0:
            raise ValueError("barycentric powers must be non-negative")
        numerator *= factorial(int(power))
    return numerator / factorial(3 + degree)


def simplex_barycentric_monomial_integral(volume: float, powers: MultiIndex) -> float:
    return float(volume) * _unit_simplex_moment(tuple(powers))


def integrate_polynomial_times_lambda_pair(
    volume: float,
    poly: Mapping[MultiIndex, float],
    first: int,
    second: int,
) -> float:
    total = 0.0
    for powers, coefficient in poly.items():
        augmented = list(powers)
        augmented[int(first)] += 1
        augmented[int(second)] += 1
        total += float(coefficient) * simplex_barycentric_monomial_integral(
            volume,
            tuple(augmented),
        )
    return total


def assemble_polynomial_weighted_nedelec_mass(
    mesh: TetrahedralComplex3D,
    tetra_polynomials: Iterable[Mapping[MultiIndex, float]],
) -> sp.csr_matrix:
    """Exactly integrate polynomial scalar weights against first-order Nedelec fields."""

    polynomials = tuple(dict(poly) for poly in tetra_polynomials)
    if len(polynomials) != mesh.n_tetrahedra:
        raise ValueError("one polynomial is required for every tetrahedron")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for q, tet in enumerate(mesh.tetrahedra):
        poly = polynomials[q]
        if not poly:
            continue
        gradients = _barycentric_gradients(mesh.vertices, tet)
        local_pairs = mesh._local_edge_vertex_indices(tet)
        volume = float(mesh.volumes[q])
        # Only ten independent moments are needed, not four recomputations for
        # each of the 36 edge pairs. This is the same exact polynomial integral.
        moments = np.empty((4, 4))
        for i in range(4):
            for j in range(i, 4):
                moments[i, j] = moments[j, i] = integrate_polynomial_times_lambda_pair(volume, poly, i, j)
        coefficients = np.zeros((6, 4, 3))
        for p, (i, j) in enumerate(local_pairs):
            coefficients[p, i] = gradients[j]
            coefficients[p, j] = -gradients[i]
        local = np.einsum('pik,ij,qjk->pq', coefficients, moments, coefficients)

        edges = mesh.tet_edge_indices[q]
        for p, ep in enumerate(edges):
            for r, er in enumerate(edges):
                value = float(local[p, r])
                if value != 0.0:
                    rows.append(int(ep))
                    cols.append(int(er))
                    data.append(value)

    matrix = sp.coo_matrix((data, (rows, cols)), shape=(mesh.n_edges, mesh.n_edges)).tocsr()
    matrix.sum_duplicates()
    return matrix
