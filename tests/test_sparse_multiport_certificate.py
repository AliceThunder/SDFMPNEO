import numpy as np
import scipy.linalg

from sdfmpneo.em import (
    ConductivityRegion,
    ConstantConductivity,
    ImpressedCurrentPortSet,
    NonlinearTetrahedralApsiProblem,
    ReciprocalLinearResistivity,
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
        ConductivityRegion(
            "seawater",
            ~copper,
            ConstantConductivity(5.0),
        ),
    )
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    reference_temperature = np.ones((mesh.n_tetrahedra, 4)) * 293.15
    thermal_modes = np.zeros((1, mesh.n_tetrahedra, 4))
    thermal_modes[0, :, :] = np.array([0.2, 0.4, 0.1, 0.3])
    mu0 = 4.0e-7 * np.pi
    problem = NonlinearTetrahedralApsiProblem(
        mesh,
        omega=2.0 * np.pi * 1.0e5,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) / mu0,
        source_current=source,
        temperature_reference_local=reference_temperature,
        thermal_modes_local=thermal_modes,
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


def test_sparse_multiport_actual_output_error_is_inside_algebraic_certificate():
    problem, ports = build_problem_and_ports()
    a = np.array([10.0])
    A_dense = problem.operator_sparse(a).toarray()
    beta = np.nextafter(float(scipy.linalg.svdvals(A_dense)[-1]), 0.0)

    requested_Z_error = 1e-5
    assert problem._H_metric is None
    sparse_result = ports.evaluate_sparse_certified(
        problem,
        a,
        stability_lower_bound=beta,
        requested_impedance_element_error=requested_Z_error,
    )
    # The sparse online path must not instantiate the dense Riesz metric.
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
