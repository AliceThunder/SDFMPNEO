import numpy as np
import scipy.linalg

from sdfmpneo.certification import certify_algebraic_heat_source_error
from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    SparseLinearSolveCertificate,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D


def build_problem():
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
    mesh = TetrahedralComplex3D.build(vertices, tetrahedra)
    copper = np.array([True, True, False, False])
    regions = (
        ConductivityRegion(
            "copper",
            copper,
            ReciprocalLinearResistivity(5.8e7, 3.93e-3, 293.15),
        ),
        ConductivityRegion("seawater", ~copper, ConstantConductivity(5.0)),
    )
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    reference = np.ones((mesh.n_tetrahedra, 4)) * 293.15
    modes = np.zeros((1, mesh.n_tetrahedra, 4))
    modes[0, :, :] = np.array([0.2, 0.4, 0.1, 0.3])
    mu0 = 4.0e-7 * np.pi
    return NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) / mu0,
        source_current=source,
        temperature_reference_local=reference,
        thermal_modes_local=modes,
        conductivity_regions=regions,
        constitutive_relative_error_budget=1e-10,
    )


def test_quadratic_heat_source_change_is_bounded_by_sparse_frobenius_certificate():
    problem = build_problem()
    a = np.array([9.0])
    A = problem.operator_sparse(a).toarray()
    x_exact = scipy.linalg.solve(A, problem.b, assume_a="gen")

    direction = np.arange(1, problem.n_em + 1, dtype=float).astype(complex)
    direction /= np.linalg.norm(direction)
    eps_x = 2e-7
    perturbation = 0.6 * eps_x * direction
    x_approx = x_exact + perturbation

    synthetic_linear_certificate = SparseLinearSolveCertificate(
        requested_state_error=eps_x,
        stability_lower_bound=1.0,
        residual_target=eps_x,
        residual_norm=0.6 * eps_x,
        relative_residual_norm=0.0,
        state_error_bound=eps_x,
        iterations=0,
        iterative_info=0,
        method="controlled_test_state",
        certified=True,
    )
    certificate = certify_algebraic_heat_source_error(
        problem,
        a,
        x_approx,
        synthetic_linear_certificate,
    )

    observed = []
    for j in range(problem.n_thermal):
        H = problem.loss_operator_sparse(j, a)
        q_exact = float(np.real(np.vdot(x_exact, H @ x_exact)))
        q_approx = float(np.real(np.vdot(x_approx, H @ x_approx)))
        observed.append(abs(q_exact - q_approx))

    observed = np.asarray(observed)
    assert certificate.certified
    assert np.all(observed <= certificate.component_error_bounds * (1.0 + 1e-12) + 1e-12)
    assert np.linalg.norm(observed) <= certificate.heat_source_vector_error_bound * (1.0 + 1e-12) + 1e-12
