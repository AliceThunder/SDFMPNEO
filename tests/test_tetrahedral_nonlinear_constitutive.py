import numpy as np

from sdfmpneo.em import (
    AffineConductivity,
    ConductivityRegion,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    ResidualGreedyEMReducer,
    certified_reciprocal_polynomials,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D
from sdfmpneo.thermal import ThermalSpectralModel


def evaluate_barycentric_polynomial(poly, barycentric):
    lam = np.asarray(barycentric, dtype=float)
    total = 0.0
    for powers, coefficient in poly.items():
        term = float(coefficient)
        for i, power in enumerate(powers):
            term *= lam[i] ** power
        total += term
    return total


def test_certified_reciprocal_series_bounds_true_pointwise_error():
    denominator = np.array([0.82, 1.05, 1.31, 0.96])
    inverse, inverse_square, certificate = certified_reciprocal_polynomials(
        denominator,
        requested_relative_error=1e-9,
    )
    assert certificate.certified
    barycentric_points = [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.25, 0.25, 0.25, 0.25]),
        np.array([0.1, 0.2, 0.3, 0.4]),
        np.array([0.55, 0.05, 0.15, 0.25]),
    ]
    for lam in barycentric_points:
        d = float(denominator @ lam)
        exact_inverse = 1.0 / d
        exact_square = 1.0 / (d * d)
        approx_inverse = evaluate_barycentric_polynomial(inverse, lam)
        approx_square = evaluate_barycentric_polynomial(inverse_square, lam)
        inverse_relative = abs(approx_inverse - exact_inverse) / abs(exact_inverse)
        square_relative = abs(approx_square - exact_square) / abs(exact_square)
        assert inverse_relative <= certificate.inverse_relative_bound * (1.0 + 1e-10) + 2e-15
        assert square_relative <= certificate.inverse_square_relative_bound * (1.0 + 1e-10) + 2e-15


def centered_tetrahedral_mesh():
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ]
    )
    tetrahedra = np.array(
        [
            [4, 1, 2, 3],
            [0, 4, 2, 3],
            [0, 1, 4, 3],
            [0, 1, 2, 4],
        ],
        dtype=int,
    )
    return TetrahedralComplex3D.build(vertices, tetrahedra)


def build_nonlinear_problem():
    mesh = centered_tetrahedral_mesh()
    thermal_assembly = mesh.assemble_p1_thermal(
        rho_cp_tetra=np.ones(mesh.n_tetrahedra) * 2.0,
        conductivity_tetra=np.ones(mesh.n_tetrahedra) * 3.0,
        homogeneous_dirichlet_boundary=True,
    )
    thermal = ThermalSpectralModel.build(thermal_assembly.M, thermal_assembly.K)
    full_mode = thermal_assembly.expand_free(thermal.Phi[:, 0])
    local_modes = full_mode[mesh.tetrahedra][None, ...]
    T0 = np.ones((mesh.n_tetrahedra, 4)) * 293.15

    copper = np.array([True, True, False, False])
    seawater = ~copper
    regions = [
        ConductivityRegion(
            "copper",
            copper,
            ReciprocalLinearResistivity(
                sigma_ref=500.0,
                alpha=0.004,
                temperature_ref=293.15,
            ),
        ),
        ConductivityRegion(
            "seawater",
            seawater,
            AffineConductivity(
                sigma_ref=5.0,
                beta=0.01,
                temperature_ref=293.15,
            ),
        ),
    ]
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    problem = NonlinearTetrahedralApsiProblem(
        mesh,
        omega=8.0,
        reluctivity_tetra=np.array([1.1, 1.2, 1.3, 1.4]),
        source_current=source,
        temperature_reference_local=T0,
        thermal_modes_local=local_modes,
        conductivity_regions=regions,
        constitutive_relative_error_budget=1e-13,
    )
    return problem


def test_nonlinear_tetrahedral_operator_and_heat_jacobian_match_difference():
    problem = build_nonlinear_problem()
    a = np.array([0.08])
    certificate = problem.constitutive_certificate(a)
    assert certificate.certified
    assert certificate.maximum_inverse_relative_bound <= 1e-13
    assert certificate.maximum_inverse_square_relative_bound <= 1e-13

    analytic_A = problem.operator_derivatives(a)[0]
    h_operator = 1e-4
    finite_A = (
        problem.operator(a + np.array([h_operator]))
        - problem.operator(a - np.array([h_operator]))
    ) / (2.0 * h_operator)
    assert np.allclose(analytic_A, finite_A, rtol=2e-6, atol=2e-7)

    states = [np.array([value]) for value in (-0.15, 0.0, 0.15)]
    reduced = ResidualGreedyEMReducer(problem).build(states, tolerance=1e-11)
    _, jacobian = reduced.heat_source_and_jacobian(a)
    h_heat = 1e-4
    finite_heat = (
        reduced.heat_source(a + np.array([h_heat]))
        - reduced.heat_source(a - np.array([h_heat]))
    ) / (2.0 * h_heat)
    assert np.allclose(jacobian[:, 0], finite_heat, rtol=2e-5, atol=2e-7)
