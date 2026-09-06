import numpy as np
import scipy.linalg

from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    ImpressedCurrentPortSet,
    NonlinearTetrahedralApsiProblem,
    PhysicalEnergySparseApsiSolver,
    ReciprocalLinearResistivity,
    apsi_physical_energy_metric,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D


def build_problem_and_ports():
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
    edge_currents = np.column_stack(
        [
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0])).real,
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[1])).real,
        ]
    )
    ports = ImpressedCurrentPortSet.build(
        mesh,
        a_basis=problem.a_basis.toarray(),
        n_scalar=problem.n_scalar,
        omega=problem.omega,
        edge_currents=edge_currents,
        names=("p1", "p2"),
    )
    return problem, ports


def test_physical_apsi_energy_metric_has_structural_one_over_sqrt_two_coercivity():
    problem, _ = build_problem_and_ports()
    a = np.array([11.0])
    A = problem.operator_sparse(a)
    H = apsi_physical_energy_metric(A)

    for shift in range(1, 5):
        x = (np.arange(1, problem.n_em + 1) + 1j * shift).astype(complex)
        x /= np.linalg.norm(x)
        lhs = abs(np.vdot(x, A @ x))
        energy = float(np.real(np.vdot(x, H @ x)))
        rhs = energy / np.sqrt(2.0)
        backward = np.finfo(float).eps * max(1.0, lhs, rhs) * problem.n_em
        assert lhs + backward >= rhs


def test_energy_solver_state_error_bound_contains_actual_dense_reference_error():
    problem, _ = build_problem_and_ports()
    a = np.array([13.0])
    A = problem.operator_sparse(a)
    solver = PhysicalEnergySparseApsiSolver(A)
    requested = 1e-8
    x_sparse, certificate = solver.solve(
        problem.b,
        requested_energy_state_error=requested,
    )
    x_dense = scipy.linalg.solve(A.toarray(), problem.b, assume_a="gen")
    actual_energy_error = solver.energy.norm(x_sparse - x_dense)

    assert certificate.certified
    assert certificate.coercivity_lower_bound == 1.0 / np.sqrt(2.0)
    assert certificate.energy_state_error_bound <= requested
    assert actual_energy_error <= certificate.energy_state_error_bound * (1.0 + 1e-8) + 1e-12


def test_energy_certified_sparse_multiport_needs_no_external_singular_value_bound():
    problem, ports = build_problem_and_ports()
    a = np.array([10.0])
    requested_Z_error = 1e-5

    assert problem._H_metric is None
    sparse_result = ports.evaluate_sparse_physical_certified(
        problem,
        a,
        requested_impedance_element_error=requested_Z_error,
    )
    assert problem._H_metric is None
    assert sparse_result.all_linear_solves_certified
    assert sparse_result.maximum_impedance_error_bound <= requested_Z_error

    dense_result = ports.evaluate(problem, a)
    observed_Z = np.abs(sparse_result.impedance - dense_result.impedance)
    observed_R = np.abs(sparse_result.resistance - dense_result.resistance)
    observed_L = np.abs(sparse_result.inductance - dense_result.inductance)
    assert np.all(observed_Z <= sparse_result.impedance_element_error_bounds + 1e-12)
    assert np.all(observed_R <= sparse_result.resistance_element_error_bounds + 1e-12)
    assert np.all(observed_L <= sparse_result.inductance_element_error_bounds + 1e-18)
