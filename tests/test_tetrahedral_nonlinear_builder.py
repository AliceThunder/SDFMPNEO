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


def build_nonlinear_core():
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
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) * 1.3,
        conductivity_regions=regions,
        temperature_reference_nodal=np.ones(mesh.n_nodes) * 293.15,
        constitutive_relative_error_budget=1e-12,
        rho_cp_tetra=np.ones(mesh.n_tetrahedra) * 2.0,
        thermal_conductivity_tetra=np.ones(mesh.n_tetrahedra) * 3.0,
        source_current=source,
    )
    return core


def test_one_call_nonlinear_builder_uses_certified_material_backend():
    core = build_nonlinear_core()
    assert core.material_backend == "certified_nonlinear"
    assert core.conductivity_reference_tetra is None
    assert core.conductivity_regions is not None
    certificate = core.electromagnetic_problem.constitutive_certificate(np.zeros(core.thermal_model.rank))
    assert certificate.certified
    assert core.electromagnetic_problem._H_metric is None

    requested = 1e-8
    reduced = core.build_reduced_electromagnetics(
        [np.array([-0.1]), np.array([0.0]), np.array([0.1])],
        requested_energy_state_error=requested,
    )
    assert reduced.reduction_certificate.certified
    assert reduced.residual_certificate(np.array([0.02])).energy_state_error_bound <= requested
    assert core.electromagnetic_problem._H_metric is None


def test_core_build_ports_never_densifies_full_order_gauge_basis(monkeypatch):
    core = build_nonlinear_core()
    mesh = core.mesh
    edge_currents = np.column_stack(
        [
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0])).real,
            tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[1])).real,
        ]
    )

    matrix_type = type(core.electromagnetic_discretization.a_basis)

    def forbidden_toarray(*_args, **_kwargs):
        raise AssertionError("full-order sparse gauge basis was densified")

    monkeypatch.setattr(matrix_type, "toarray", forbidden_toarray)
    ports = core.build_ports(edge_currents, names=("p1", "p2"))

    assert ports.coordinate_rhs.shape == (core.electromagnetic_problem.n_em, 2)
    assert core.electromagnetic_problem._H_metric is None
