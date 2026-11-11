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
