import numpy as np

from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    ImpressedCurrentPortSet,
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


def _gram_roundoff_envelope(H, V):
    """Conditioning-aware standard-model bound for forming V^H H V."""

    Habs = H.copy().tocsr()
    Habs.data = np.abs(Habs.data)
    Vabs = np.abs(V)
    absolute_scale = Vabs.T @ (Habs @ Vabs)
    max_row_nnz = int(np.max(np.diff(Habs.indptr))) if Habs.shape[0] else 0
    n = V.shape[0]
    r = V.shape[1]
    operation_depth = max_row_nnz + n + 4 * r * r
    eps = np.finfo(float).eps
    gamma = operation_depth * eps / (1.0 - operation_depth * eps)
    return float(gamma * np.linalg.norm(absolute_scale, ord="fro"))


def _two_port_set(problem):
    mesh = problem.mesh
    edge_currents = np.column_stack(
        [
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0])).real,
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[1])).real,
        ]
    )
    return ImpressedCurrentPortSet.build(
        mesh,
        a_basis=problem.a_basis,
        n_scalar=problem.n_scalar,
        omega=problem.omega,
        edge_currents=edge_currents,
        names=("p1", "p2"),
    )


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
    roundoff_envelope = _gram_roundoff_envelope(
        model.reference_energy_metric,
        model.V,
    )
    assert defect <= roundoff_envelope

    for state in states:
        cert = model.residual_certificate(state)
        assert cert.energy_state_error_bound <= requested


def test_sparse_energy_multi_rhs_reduction_is_jointly_certified_without_snapshots():
    problem = build_high_contrast_problem()
    ports = _two_port_set(problem)
    rhs = ports.coordinate_rhs
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


def test_reduced_multiport_outputs_are_certified_without_full_order_equilibrium_solve():
    problem = build_high_contrast_problem()
    ports = _two_port_set(problem)
    query_state = np.array([6.0])
    candidate_states = [np.array([-8.0]), np.array([0.0]), query_state, np.array([12.0])]

    reducer = SparseEnergyResidualGreedyEMReducer(problem)
    model = reducer.build_multi_rhs(
        candidate_states,
        ports.coordinate_rhs,
        requested_energy_state_error=1e-9,
    )
    assert model.reduction_certificate.certified

    full = ports.evaluate_sparse_physical_certified(
        problem,
        query_state,
        requested_impedance_element_error=1e-7,
    )
    reduced = ports.evaluate_reduced_physical_certified(
        problem,
        query_state,
        model,
        requested_impedance_element_error=1e-7,
    )
    assert reduced.certified

    observed = np.abs(reduced.impedance - full.impedance)
    combined_bound = reduced.impedance_element_error_bounds + full.impedance_element_error_bounds
    assert np.all(observed <= combined_bound)

    def full_solve_forbidden(*_args, **_kwargs):
        raise AssertionError("full-order equilibrium solve was used")

    problem.solve_full = full_solve_forbidden
    problem.solve_sparse = full_solve_forbidden
    problem.operator = full_solve_forbidden

    reduced_again = ports.evaluate_reduced_physical_certified(
        problem,
        query_state,
        model,
        requested_impedance_element_error=1e-7,
    )
    assert reduced_again.certified
    assert np.allclose(reduced_again.impedance, reduced.impedance)
    assert problem._H_metric is None
