import numpy as np

from sdfmpneo.em import (
    ResidualGreedyEMReducer,
    build_tetrahedral_apsi_from_thermal_modes,
    tetra_face_loop_source,
)
from sdfmpneo.spatial import TetrahedralComplex3D
from sdfmpneo.thermal import ThermalSpectralModel


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


def build_problem():
    mesh = centered_tetrahedral_mesh()
    thermal_assembly = mesh.assemble_p1_thermal(
        rho_cp_tetra=np.ones(mesh.n_tetrahedra) * 2.0,
        conductivity_tetra=np.ones(mesh.n_tetrahedra) * 3.0,
        homogeneous_dirichlet_boundary=True,
    )
    thermal = ThermalSpectralModel.build(thermal_assembly.M, thermal_assembly.K)

    local_modes = []
    for k in range(thermal.Phi.shape[1]):
        full = thermal_assembly.expand_free(thermal.Phi[:, k])
        local_modes.append(full[mesh.tetrahedra])
    local_modes = np.asarray(local_modes)

    source = tetra_face_loop_source(mesh, int(mesh.boundary_face_indices[0]))
    assert np.allclose(np.asarray(mesh.grad.T @ source).ravel(), 0.0)

    discretization = build_tetrahedral_apsi_from_thermal_modes(
        mesh,
        omega=7.0,
        reluctivity_tetra=np.array([1.1, 1.3, 1.2, 1.4]),
        conductivity_reference_tetra=np.array([2.0, 1.5, 1.8, 2.2]),
        conductivity_temperature_slope_tetra=np.array([0.02, 0.01, 0.015, 0.018]),
        thermal_mode_local_values=local_modes,
        source_current=source,
    )
    return mesh, thermal, discretization.to_parametric_problem()


def test_tetrahedral_apsi_is_reciprocal_and_reduces_without_snapshots():
    _, thermal, problem = build_problem()
    assert problem.n_thermal == thermal.Phi.shape[1] == 1
    assert np.allclose(problem.A0, problem.A0.T, rtol=2e-13, atol=2e-13)
    assert np.allclose(problem.A_state[0], problem.A_state[0].T, rtol=2e-13, atol=2e-13)

    states = [np.array([value]) for value in (-0.2, 0.0, 0.2)]
    reduced = ResidualGreedyEMReducer(problem).build(states, tolerance=1e-11)
    query = np.array([0.07])
    assert reduced.residual_dual_norm(query) < 1e-9
    assert reduced.V.shape[1] <= problem.n_em


def test_tetrahedral_projected_joule_jacobian_matches_full_nonlinear_difference():
    _, _, problem = build_problem()
    states = [np.array([value]) for value in (-0.25, 0.0, 0.25)]
    reduced = ResidualGreedyEMReducer(problem).build(states, tolerance=1e-12)

    a = np.array([0.05])
    q, jacobian = reduced.heat_source_and_jacobian(a)
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(jacobian))

    h = 1e-5
    finite = (
        reduced.heat_source(a + np.array([h]))
        - reduced.heat_source(a - np.array([h]))
    ) / (2.0 * h)
    assert np.allclose(jacobian[:, 0], finite, rtol=3e-6, atol=3e-8)
