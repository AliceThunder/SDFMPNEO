import numpy as np

from sdfmpneo import TetrahedralElectroThermalCore
from sdfmpneo.certification import (
    certify_constitutive_heat_source_error,
    certify_constitutive_multiport_error,
)
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


def build_core(error_budget):
    mesh = centered_tetrahedral_mesh()
    copper = np.array([True, True, False, False])
    seawater = ~copper
    regions = [
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
        conductivity_regions=regions,
        temperature_reference_nodal=np.ones(mesh.n_nodes) * 293.15,
        constitutive_relative_error_budget=error_budget,
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
    return core, ports


def test_constitutive_series_impedance_difference_is_below_combined_certificates():
    coarse_core, coarse_ports = build_core(2e-4)
    tight_core, tight_ports = build_core(1e-12)
    a = np.array([0.08])

    coarse_certificate = certify_constitutive_multiport_error(
        coarse_core.electromagnetic_problem,
        coarse_ports,
        a,
    )
    tight_certificate = certify_constitutive_multiport_error(
        tight_core.electromagnetic_problem,
        tight_ports,
        a,
    )
    assert coarse_certificate.certified
    assert tight_certificate.certified

    coarse_Z = coarse_ports.evaluate(coarse_core.electromagnetic_problem, a).impedance
    tight_Z = tight_ports.evaluate(tight_core.electromagnetic_problem, a).impedance
    observed = np.linalg.norm(coarse_Z - tight_Z, ord=2)
    rigorous_pair_bound = (
        coarse_certificate.impedance_spectral_norm_bound
        + tight_certificate.impedance_spectral_norm_bound
    )
    assert observed <= rigorous_pair_bound * (1.0 + 1e-10) + 1e-13


def test_tighter_constitutive_budget_reduces_certified_impedance_bound():
    coarse_core, coarse_ports = build_core(1e-3)
    tight_core, tight_ports = build_core(1e-8)
    a = np.array([0.06])
    coarse = certify_constitutive_multiport_error(
        coarse_core.electromagnetic_problem,
        coarse_ports,
        a,
    )
    tight = certify_constitutive_multiport_error(
        tight_core.electromagnetic_problem,
        tight_ports,
        a,
    )
    assert coarse.certified and tight.certified
    assert tight.constitutive_relative_bound <= coarse.constitutive_relative_bound
    assert tight.impedance_spectral_norm_bound <= coarse.impedance_spectral_norm_bound


def test_constitutive_heat_source_difference_is_below_combined_certificates():
    coarse_core, _ = build_core(3e-4)
    tight_core, _ = build_core(1e-12)
    a = np.array([0.07])

    coarse_problem = coarse_core.electromagnetic_problem
    tight_problem = tight_core.electromagnetic_problem
    coarse_certificate = certify_constitutive_heat_source_error(coarse_problem, a)
    tight_certificate = certify_constitutive_heat_source_error(tight_problem, a)
    assert coarse_certificate.certified
    assert tight_certificate.certified

    coarse_x = coarse_problem.solve_full(a)
    tight_x = tight_problem.solve_full(a)
    coarse_q = np.array(
        [
            np.real(np.vdot(coarse_x, coarse_problem.loss_operator(j, a) @ coarse_x))
            for j in range(coarse_problem.n_thermal)
        ]
    )
    tight_q = np.array(
        [
            np.real(np.vdot(tight_x, tight_problem.loss_operator(j, a) @ tight_x))
            for j in range(tight_problem.n_thermal)
        ]
    )
    observed = np.linalg.norm(coarse_q - tight_q)
    pair_bound = (
        coarse_certificate.heat_source_vector_error_bound
        + tight_certificate.heat_source_vector_error_bound
    )
    assert observed <= pair_bound * (1.0 + 1e-10) + 1e-13


def test_tighter_constitutive_budget_reduces_heat_source_bound():
    coarse_core, _ = build_core(1e-3)
    tight_core, _ = build_core(1e-8)
    a = np.array([0.05])
    coarse = certify_constitutive_heat_source_error(
        coarse_core.electromagnetic_problem,
        a,
    )
    tight = certify_constitutive_heat_source_error(
        tight_core.electromagnetic_problem,
        a,
    )
    assert coarse.certified and tight.certified
    assert tight.heat_source_vector_error_bound <= coarse.heat_source_vector_error_bound
