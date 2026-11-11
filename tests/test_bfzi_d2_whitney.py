"""Manufactured uniform field checks for raw oriented Whitney curls."""
import numpy as np

from sdfmpneo_bfzi.direct3d_aphi import MU0, TetraMesh, _topology
from sdfmpneo_bfzi.direct3d_whitney import curl_b, magnetic_energy


def test_curl_orientation_and_gauge():
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                      [0, 0, 1], [0, 0, -1.]], float)
    tet = np.array([[0, 1, 2, 3], [1, 0, 2, 4]])
    mesh = TetraMesh(nodes, tet, np.zeros(2))
    edges, *_ = _topology(mesh)
    b0 = np.array([0., 0., 3.])
    a = np.cross(b0, nodes)/2
    dof = np.array([np.dot((a[i]+a[j])/2, nodes[j]-nodes[i])
                    for i, j in edges])
    b = curl_b(mesh, dof)
    assert b.shape == (1, 2, 3)
    np.testing.assert_allclose(b, [[b0, b0]], atol=1e-14)
    phi = nodes @ np.array([1., -3., 2.])
    grad_dof = np.array([phi[j]-phi[i] for i, j in edges])
    np.testing.assert_allclose(curl_b(mesh, dof+grad_dof), b, atol=1e-14)
    en = magnetic_energy(mesh, b)
    np.testing.assert_allclose(en, [0.5*(2/6)*9/MU0], rtol=1e-14)
    np.testing.assert_allclose(magnetic_energy(mesh, b, average=True), en/2)
    np.testing.assert_allclose(
        magnetic_energy(mesh, b, phase_rad=np.pi/2), [0.], atol=1e-24)


def test_actual_team7_edge_solution_matches_independent_curl():
    """Compare two independent reconstructions on a solved 3-D metal problem."""
    from sdfmpneo_bfzi.direct3d_team7 import (
        graded_team7_smoke_mesh, team7_stranded_source)
    from sdfmpneo_bfzi.direct3d_galerkin import solve_closed_ports

    mesh = graded_team7_smoke_mesh(1)
    _, _, strands = team7_stranded_source(2, 2, 5, return_strands=True)
    res = solve_closed_ports(mesh, 200., [strands])
    med = curl_b(mesh, res.medium.edge_coefficients)
    vac = curl_b(mesh, res.vacuum.edge_coefficients)
    np.testing.assert_allclose(med, res.medium.magnetic_secondary,
                               rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(vac, res.vacuum.magnetic_secondary,
                               rtol=1e-11, atol=1e-10)
    assert np.linalg.norm(med-vac) > 0
    for angle in (0, np.pi/2):
        energy = magnetic_energy(mesh, med, phase_rad=angle)[0]
        assert energy >= 0 and np.isfinite(energy)
    assert magnetic_energy(mesh, med, average=True)[0] > 0
