import numpy as np

from sdfmpneo import TetrahedralElectroThermalCore
from sdfmpneo.em import tetra_face_loop_source
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


def test_one_call_tetrahedral_core_builds_shared_thermal_and_em_spaces():
    mesh = centered_tetrahedral_mesh()
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    core = TetrahedralElectroThermalCore.build(
        mesh,
        omega=5.0,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra) * 1.4,
        conductivity_reference_tetra=np.ones(mesh.n_tetrahedra) * 2.0,
        conductivity_temperature_slope_tetra=np.ones(mesh.n_tetrahedra) * 0.02,
        rho_cp_tetra=np.ones(mesh.n_tetrahedra) * 3.0,
        thermal_conductivity_tetra=np.ones(mesh.n_tetrahedra) * 4.0,
        source_current=source,
    )

    assert core.thermal_model.rank == 1
    assert core.electromagnetic_problem.n_thermal == core.thermal_model.rank
    assert core.thermal_mode_local_values.shape == (1, mesh.n_tetrahedra, 4)

    reduced = core.build_affine_verification_reduced_electromagnetics(
        [np.array([-0.1]), np.array([0.0]), np.array([0.1])],
        residual_tolerance=1e-11,
    )
    assert reduced.residual_dual_norm(np.array([0.03])) < 1e-9


def test_one_call_tetrahedral_core_uses_certified_thermal_rank_path():
    mesh = centered_tetrahedral_mesh()
    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    core = TetrahedralElectroThermalCore.build(
        mesh,
        omega=5.0,
        reluctivity_tetra=np.ones(mesh.n_tetrahedra),
        conductivity_reference_tetra=np.ones(mesh.n_tetrahedra),
        conductivity_temperature_slope_tetra=np.zeros(mesh.n_tetrahedra),
        rho_cp_tetra=np.ones(mesh.n_tetrahedra),
        thermal_conductivity_tetra=np.ones(mesh.n_tetrahedra),
        source_current=source,
        initial_temperature_deviation_free=np.zeros(1),
        source_dual_bound=0.0,
        requested_state_tolerance=1e-12,
    )
    assert core.thermal_tail_certificate is not None
    assert core.thermal_tail_certificate.certified
    assert core.thermal_model.rank == 1
