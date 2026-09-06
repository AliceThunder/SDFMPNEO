import numpy as np

from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
    SparseEnergyResidualGreedyEMReducer,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D


def build_high_contrast_problem():
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
    problem = NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) / mu0,
        source_current=source,
        temperature_reference_local=reference,
        thermal_modes_local=modes,
        conductivity_regions=regions,
        constitutive_relative_error_budget=1e-10,
    )
    return problem


def test_sparse_energy_reducer_never_materializes_dense_riesz_metric():
    problem = build_high_contrast_problem()
    assert problem._H_metric is None

    reducer = SparseEnergyResidualGreedyEMReducer(problem)
    states = [np.array([-10.0]), np.array([0.0]), np.array([15.0])]
    requested = 1e-6
    model = reducer.build(states, requested_energy_state_error=requested)

    assert problem._H_metric is None
    assert model.reduction_certificate.certified
    assert model.reduction_certificate.maximum_energy_state_error_bound <= requested

    gram = model.V.conj().T @ (model.reference_energy_metric @ model.V)
    defect = np.linalg.norm(gram - np.eye(model.n_reduced), ord="fro")
    backward = (
        128.0
        * np.finfo(float).eps
        * max(1, problem.n_em, model.n_reduced)
        * max(1.0, np.linalg.norm(gram, ord="fro"))
    )
    assert defect <= backward

    for state in states:
        cert = model.residual_certificate(state)
        assert cert.energy_state_error_bound <= requested


def test_sparse_energy_multi_rhs_reduction_is_jointly_certified_without_snapshots():
    problem = build_high_contrast_problem()
    mesh = problem.mesh
    rhs = np.column_stack(
        [
            problem.source_coordinate(
                tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
            ),
            problem.source_coordinate(
                tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[1]))
            ),
        ]
    )
    states = [np.array([-8.0]), np.array([0.0]), np.array([12.0])]
    requested = 2e-6
    reducer = SparseEnergyResidualGreedyEMReducer(problem)
    model = reducer.build_multi_rhs(
        states,
        rhs,
        requested_energy_state_error=requested,
    )

    assert model.reduction_certificate.certified
    assert problem._H_metric is None
    for state in states:
        for p in range(rhs.shape[1]):
            cert = model.residual_certificate_for_rhs(state, rhs[:, p])
            assert cert.energy_state_error_bound <= requested


def test_sparse_reduced_heat_and_jacobian_use_only_sparse_full_order_operators():
    problem = build_high_contrast_problem()
    reducer = SparseEnergyResidualGreedyEMReducer(problem)
    model = reducer.build(
        [np.array([-5.0]), np.array([0.0]), np.array([10.0])],
        requested_energy_state_error=1e-6,
    )
    assert model.reduction_certificate.certified

    # Dense compatibility methods are deliberately disabled after construction.
    # The sparse reduced model must still evaluate state, heat source and exact
    # reduced heat-source Jacobian without touching them.
    def dense_forbidden(*_args, **_kwargs):
        raise AssertionError("dense full-order compatibility path was used")

    problem.operator = dense_forbidden
    problem.loss_operator = dense_forbidden
    problem.loss_operator_derivative = dense_forbidden
    problem.operator_derivatives = dense_forbidden

    state = np.array([4.0])
    x = model.state(state)
    q, J = model.heat_source_and_jacobian(state)

    assert x.shape == (problem.n_em,)
    assert q.shape == (problem.n_thermal,)
    assert J.shape == (problem.n_thermal, problem.n_thermal)
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(J))
    assert problem._H_metric is None
