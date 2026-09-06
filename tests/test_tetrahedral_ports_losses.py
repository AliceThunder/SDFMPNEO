import numpy as np

from sdfmpneo import TetrahedralElectroThermalCore
from sdfmpneo.em import ResidualGreedyEMReducer, tetra_face_loop_source
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


def build_core():
    mesh = centered_tetrahedral_mesh()
    first = int(mesh.boundary_face_indices[0])
    source = tetra_face_loop_source(mesh, first)
    core = TetrahedralElectroThermalCore.build(
        mesh,
        omega=6.0,
        reluctivity_tetra=np.array([1.1, 1.2, 1.3, 1.4]),
        conductivity_reference_tetra=np.array([2.0, 2.2, 1.5, 1.7]),
        conductivity_temperature_slope_tetra=np.array([0.02, 0.018, 0.01, 0.012]),
        rho_cp_tetra=np.ones(mesh.n_tetrahedra) * 2.0,
        thermal_conductivity_tetra=np.ones(mesh.n_tetrahedra) * 3.0,
        source_current=source,
    )
    return mesh, core


def test_tetrahedral_multiport_is_reciprocal_passive_and_reduced_consistent():
    mesh, core = build_core()
    face0 = int(mesh.boundary_face_indices[0])
    face1 = int(mesh.boundary_face_indices[1])
    edge_currents = np.column_stack(
        [
            tetra_face_loop_source(mesh, face0).real,
            tetra_face_loop_source(mesh, face1).real,
        ]
    )
    ports = core.build_ports(edge_currents, names=("p1", "p2"))

    states = [np.array([value]) for value in (-0.2, 0.0, 0.2)]
    reduced = ResidualGreedyEMReducer(core.electromagnetic_problem).build_multi_rhs(
        states,
        ports.coordinate_rhs,
        tolerance=1e-12,
    )
    a = np.array([0.04])
    full = ports.evaluate(core.electromagnetic_problem, a)
    rom = ports.evaluate(
        core.electromagnetic_problem,
        a,
        reduced_basis=reduced.V,
    )

    assert full.reciprocity_defect < 2e-12
    assert rom.reciprocity_defect < 2e-12
    assert full.minimum_resistance_eigenvalue >= -2e-12
    assert np.allclose(rom.impedance, full.impedance, rtol=2e-9, atol=2e-10)
    assert rom.maximum_residual_dual_norm < 1e-9


def test_tetrahedral_region_losses_partition_total_joule_power_and_match_port_power():
    mesh, core = build_core()
    face0 = int(mesh.boundary_face_indices[0])
    face1 = int(mesh.boundary_face_indices[1])
    edge_currents = np.column_stack(
        [
            tetra_face_loop_source(mesh, face0).real,
            tetra_face_loop_source(mesh, face1).real,
        ]
    )
    ports = core.build_ports(edge_currents, names=("p1", "p2"))

    copper = np.array([True, True, False, False])
    seawater = ~copper
    projector = core.build_region_loss_projector(
        {"copper": copper, "seawater": seawater}
    )

    a = np.array([0.03])
    currents = np.array([1.0 + 0.2j, -0.35 + 0.15j])
    rhs = ports.rhs_for_currents(currents)
    A = core.electromagnetic_problem.operator(a)
    x = np.linalg.solve(A, rhs)
    region = projector.evaluate_state(x, a)

    disc = core.electromagnetic_discretization
    S = disc.conductivity0.copy().astype(complex)
    for k, Sk in enumerate(disc.conductivity_state):
        S = S + a[k] * Sk
    L = disc.electric_extraction()
    electric = L @ x
    total_joule = 0.5 * np.real(np.vdot(electric, S @ electric))

    result = ports.evaluate(core.electromagnetic_problem, a)
    port_power = result.average_input_power(currents)

    assert np.isclose(region["copper"] + region["seawater"], total_joule, rtol=3e-12, atol=3e-12)
    assert np.isclose(port_power, total_joule, rtol=3e-12, atol=3e-12)
