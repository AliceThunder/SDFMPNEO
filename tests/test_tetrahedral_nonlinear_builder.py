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


def test_one_call_nonlinear_builder_uses_certified_material_backend():
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
