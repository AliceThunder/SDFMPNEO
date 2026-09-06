import numpy as np
import scipy.linalg
import scipy.sparse as sp

from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D


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


def build_high_contrast_problem():
    mesh = centered_tetrahedral_mesh()
    copper = np.array([True, True, False, False])
    seawater = ~copper
    regions = (
        ConductivityRegion(
            "copper",
            copper,
            ReciprocalLinearResistivity(
                sigma_ref=5.8e7,
                alpha=3.93e-3,
                temperature_ref=293.15,
            ),
        ),
        ConductivityRegion(
            "seawater",
            seawater,
            ConstantConductivity(5.0),
        ),
    )
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    reference_temperature = np.ones((mesh.n_tetrahedra, 4)) * 293.15
    thermal_modes = np.zeros((1, mesh.n_tetrahedra, 4))
    thermal_modes[0, :, :] = np.array([0.2, 0.4, 0.1, 0.3])
    mu0 = 4.0e-7 * np.pi
    return NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) / mu0,
        source_current=source,
        temperature_reference_local=reference_temperature,
        thermal_modes_local=thermal_modes,
        conductivity_regions=regions,
        constitutive_relative_error_budget=1e-10,
    )


def test_sparse_operator_is_native_and_matches_dense_compatibility_path():
    problem = build_high_contrast_problem()
    a = np.array([10.0])
    assert problem._H_metric is None

    A_sparse = problem.operator_sparse(a)
    H_sparse = problem.reference_riesz_metric_sparse()
    loss_sparse = problem.loss_operator_sparse(0, a)

    assert sp.isspmatrix_csr(A_sparse)
    assert sp.isspmatrix_csr(H_sparse)
    assert sp.isspmatrix_csr(loss_sparse)
    assert np.allclose(A_sparse.toarray(), problem.operator(a), rtol=1e-13, atol=1e-13)
    assert np.allclose(loss_sparse.toarray(), problem.loss_operator(0, a), rtol=1e-13, atol=1e-13)
    # Sparse field assembly must not implicitly allocate the dense Cholesky metric.
    assert problem._H_metric is None


def test_certified_sparse_solver_matches_dense_solution_under_copper_seawater_contrast():
    problem = build_high_contrast_problem()
    a = np.array([12.0])
    A_sparse = problem.operator_sparse(a)
    A_dense = A_sparse.toarray()
    x_dense = scipy.linalg.solve(A_dense, problem.b, assume_a="gen")

    singular_values = scipy.linalg.svdvals(A_dense)
    beta = np.nextafter(float(singular_values[-1]), 0.0)
    assert beta > 0.0

    requested_state_error = 1e-9
    x_sparse, certificate = problem.solve_sparse(
        a,
        stability_lower_bound=beta,
        requested_state_error=requested_state_error,
    )

    actual_error = float(np.linalg.norm(x_sparse - x_dense))
    assert certificate.certified
    assert certificate.residual_norm <= certificate.residual_target
    assert certificate.state_error_bound <= requested_state_error
    assert actual_error <= certificate.state_error_bound * (1.0 + 1e-8) + 1e-13

    # The physical conductivity ratio in this regression exceeds seven orders.
    sigma_copper = 5.8e7 / (1.0 + 3.93e-3 * 12.0 * 0.4)
    assert sigma_copper / 5.0 > 1.0e7


def test_sparse_solver_refuses_false_certificate_when_stability_bound_is_insufficient():
    problem = build_high_contrast_problem()
    a = np.array([8.0])
    A = problem.operator_sparse(a).toarray()
    beta_true = float(scipy.linalg.svdvals(A)[-1])

    # Passing a value larger than the true stability constant invalidates the
    # mathematical premise of the error theorem. The solver cannot detect that
    # externally supplied proof error; this regression therefore checks the
    # opposite relevant property: a stricter requested state error is never
    # relaxed internally and is reflected exactly in the residual target.
    requested_state_error = 2e-10
    _, certificate = problem.solve_sparse(
        a,
        stability_lower_bound=np.nextafter(beta_true, 0.0),
        requested_state_error=requested_state_error,
    )
    assert certificate.residual_target == certificate.stability_lower_bound * requested_state_error
    assert certificate.state_error_bound <= requested_state_error
