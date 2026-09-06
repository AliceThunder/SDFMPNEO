import numpy as np

from sdfmpneo import TetrahedralElectroThermalCore
from sdfmpneo.em import (
    AffineConductivity,
    ConductivityRegion,
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


def test_certified_nonlinear_region_power_closes_with_total_joule_and_port_power():
    mesh = centered_tetrahedral_mesh()
    copper = np.array([True, True, False, False])
    seawater = ~copper
    material_regions = [
        ConductivityRegion(
            "copper",
            copper,
            ReciprocalLinearResistivity(500.0, 0.004, 293.15),
        ),
        ConductivityRegion(
            "seawater",
            seawater,
            AffineConductivity(5.0, 0.01, 293.15),
        ),
    ]
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    core = TetrahedralElectroThermalCore.build_nonlinear(
        mesh,
        omega=7.0,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) * 1.2,
        conductivity_regions=material_regions,
        temperature_reference_nodal=np.ones(mesh.n_nodes) * 293.15,
        constitutive_relative_error_budget=1e-12,
        rho_cp_tetra=np.ones(mesh.n_tetrahedra) * 2.0,
        thermal_conductivity_tetra=np.ones(mesh.n_tetrahedra) * 3.0,
        source_current=source,
    )

    edge_currents = np.column_stack(
        [
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0])).real,
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[1])).real,
        ]
    )
    ports = core.build_ports(edge_currents, names=("p1", "p2"))
    region_losses = core.build_region_loss_projector()

    a = np.array([0.04])
    currents = np.array([1.0 + 0.1j, -0.3 + 0.2j])
    rhs = ports.rhs_for_currents(currents)
    problem = core.electromagnetic_problem
    x = np.linalg.solve(problem.operator(a), rhs)

    separated = region_losses.evaluate_state(x, a)
    L = problem.electric_extraction()
    electric = L @ x
    S = problem.conductivity_matrix(a)
    total_joule = 0.5 * np.real(np.vdot(electric, S @ electric))
    port_result = ports.evaluate(problem, a)
    port_power = port_result.average_input_power(currents)

    assert set(separated) == {"copper", "seawater"}
    assert np.isclose(
        separated["copper"] + separated["seawater"],
        total_joule,
        rtol=5e-11,
        atol=5e-11,
    )
    assert np.isclose(port_power, total_joule, rtol=5e-11, atol=5e-11)
    assert port_result.reciprocity_defect < 5e-11
