from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from sdfmpneo.cpp_training_backend import (
    backend_info,
    nedelec_local,
    p1_thermal_local,
    reduced_assemble_many,
    reduced_assemble_products,
)
from sdfmpneo.spatial import AffineTetrahedralGeometryChart, TetrahedralComplex3D


def _require_backend():
    info = backend_info(auto_build=True)
    if not info["available"]:
        pytest.skip(f"C++ compiler/backend unavailable: {info['error']}")
    return info


def _gradients(vertices, tet):
    x = vertices[tet]
    B = np.column_stack([np.ones(4), x])
    return np.linalg.inv(B)[1:, :].T


def test_cpp_backend_builds_and_matches_p1_and_nedelec_local_forms():
    info = _require_backend()
    assert info["version"] == "sdfmpneo_training_cpp_v1"

    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    for q, tet in enumerate(tets):
        x = vertices[tet]
        det = np.linalg.det(np.column_stack([x[1]-x[0], x[2]-x[0], x[3]-x[0]]))
        if det < 0.0:
            tets[q, [0, 1]] = tets[q, [1, 0]]

    rho = np.array([2.0, 3.0]); kappa = np.array([4.0, 5.0])
    local_M, local_K, volumes = p1_thermal_local(vertices, tets, rho, kappa)
    assert local_M is not None
    for q, tet in enumerate(tets):
        gradients = _gradients(vertices, tet)
        x = vertices[tet]
        volume = np.linalg.det(np.column_stack([x[1]-x[0], x[2]-x[0], x[3]-x[0]])) / 6.0
        expected_M = np.full((4, 4), rho[q] * volume / 20.0)
        np.fill_diagonal(expected_M, rho[q] * volume / 10.0)
        expected_K = kappa[q] * volume * (gradients @ gradients.T)
        assert np.allclose(local_M[q], expected_M, rtol=2e-14, atol=2e-14)
        assert np.allclose(local_K[q], expected_K, rtol=2e-14, atol=2e-14)
        assert volumes[q] == pytest.approx(volume, rel=2e-14, abs=2e-14)

    nu = np.array([1.2, 1.3]); sigma = np.array([2.1, 2.2])
    local_curl, local_mass = nedelec_local(vertices, tets, nu, sigma)
    assert local_curl is not None
    for q, tet in enumerate(tets):
        gradients = _gradients(vertices, tet)
        volume = volumes[q]
        local_of = {int(vertex): i for i, vertex in enumerate(tet)}
        pairs = []
        for u, v in combinations(map(int, tet), 2):
            a, b = (u, v) if u < v else (v, u)
            pairs.append((local_of[a], local_of[b]))
        integrals = np.full((4, 4), volume / 20.0)
        np.fill_diagonal(integrals, volume / 10.0)
        expected_K = np.zeros((6, 6)); expected_M = np.zeros((6, 6))
        curls = [2.0 * np.cross(gradients[i], gradients[j]) for i, j in pairs]
        for p, (i, j) in enumerate(pairs):
            gi, gj = gradients[i], gradients[j]
            for r, (k, ell) in enumerate(pairs):
                gk, gl = gradients[k], gradients[ell]
                mass = (
                    np.dot(gj, gl) * integrals[i, k]
                    - np.dot(gj, gk) * integrals[i, ell]
                    - np.dot(gi, gl) * integrals[j, k]
                    + np.dot(gi, gk) * integrals[j, ell]
                )
                expected_M[p, r] = sigma[q] * mass
                expected_K[p, r] = nu[q] * volume * np.dot(curls[p], curls[r])
        assert np.allclose(local_curl[q], expected_K, rtol=3e-14, atol=3e-14)
        assert np.allclose(local_mass[q], expected_M, rtol=3e-14, atol=3e-14)


def _unit_simplex_moment(volume, powers):
    import math
    degree = sum(powers)
    numerator = 6
    for power in powers:
        numerator *= math.factorial(power)
    return volume * numerator / math.factorial(3 + degree)


def _reduced_reference(volumes, coefficients, fields, families):
    out = np.zeros((len(families), fields.shape[2], fields.shape[2]), dtype=complex)
    for family, polynomials in enumerate(families):
        for q, poly in enumerate(polynomials):
            moments = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    for powers, coefficient in poly.items():
                        augmented = list(powers); augmented[i] += 1; augmented[j] += 1
                        moments[i, j] += coefficient * _unit_simplex_moment(volumes[q], augmented)
            local = np.einsum("pik,ij,qjk->pq", coefficients[q], moments, coefficients[q])
            out[family] += fields[q].conj().T @ (local @ fields[q])
    return out


def test_cpp_reduced_polynomial_assembly_matches_reference():
    _require_backend()
    rng = np.random.default_rng(23)
    n_tet, n_reduced = 5, 3
    volumes = rng.uniform(0.1, 0.8, size=n_tet)
    coefficients = rng.normal(size=(n_tet, 6, 4, 3))
    fields = np.ascontiguousarray(
        rng.normal(size=(n_tet, 6, n_reduced))
        + 1j * rng.normal(size=(n_tet, 6, n_reduced))
    )
    families = []
    for family in range(2):
        values = []
        for q in range(n_tet):
            poly = {(0, 0, 0, 0): rng.normal(), (1, 0, 0, 0): rng.normal()}
            if (family + q) % 2:
                poly[(0, 1, 1, 0)] = rng.normal()
            values.append(poly)
        families.append(tuple(values))
    multipliers = [
        tuple({(0, 0, 0, 0): rng.normal(), (0, 0, 1, 0): rng.normal()} for _ in range(n_tet)),
        tuple({(0, 0, 0, 0): rng.normal(), (0, 0, 0, 1): rng.normal()} for _ in range(n_tet)),
    ]

    actual = reduced_assemble_many(volumes, coefficients, fields, families)
    expected = _reduced_reference(volumes, coefficients, fields, families)
    assert np.allclose(actual, expected, rtol=2e-13, atol=2e-13)

    product_families = []
    for multiplier in multipliers:
        row = []
        for family in families:
            values = []
            for base, test in zip(family, multiplier):
                product = {}
                for a, ca in base.items():
                    for b, cb in test.items():
                        power = tuple(a[k] + b[k] for k in range(4))
                        product[power] = product.get(power, 0.0) + ca * cb
                values.append(product)
            row.append(tuple(values))
        product_families.append(row)
    expected_products = np.stack([
        np.stack([_reduced_reference(volumes, coefficients, fields, [values])[0]
                  for values in row])
        for row in product_families
    ])
    actual_products = reduced_assemble_products(
        volumes, coefficients, fields, families, multipliers
    )
    assert np.allclose(actual_products, expected_products, rtol=3e-13, atol=3e-13)


def test_geometry_chart_reuses_topology_but_updates_geometry():
    reference_vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    tetrahedra = np.array([[0, 1, 2, 3]], dtype=int)
    directions = np.zeros((1, 4, 3)); directions[0, 1, 0] = 0.1
    chart = AffineTetrahedralGeometryChart(
        reference_vertices, tetrahedra, directions, ("scale_x",)
    )
    first = chart.mesh(np.array([0.0])); second = chart.mesh(np.array([0.5]))
    explicit = TetrahedralComplex3D.build(chart.vertices(np.array([0.5])), tetrahedra)
    assert first.grad is second.grad
    assert first.curl is second.curl
    assert first.edge_vertices is second.edge_vertices
    assert first.tet_edge_indices is second.tet_edge_indices
    assert np.array_equal(second.tetrahedra, explicit.tetrahedra)
    assert np.allclose(second.volumes, explicit.volumes, rtol=2e-14, atol=2e-14)
