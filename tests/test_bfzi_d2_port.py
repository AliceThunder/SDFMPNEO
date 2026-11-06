"""Focused physical port voltage checks for independent D2 A-phi FEM."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.direct3d_aphi import (
    box_tetra_mesh, solve_aphi, closed_polyline_potential,
)
from sdfmpneo_bfzi.direct3d_port import (
    reaction_potential_at_points, closed_source_port_impedance,
)


def _loop():
    return np.array([[-.05,-.05,0.], [.05,-.05,0.],
                     [.05,.05,0.], [-.05,.05,0.],
                     [-.05,-.05,0.]])


def test_total_vector_boundary_and_gauge_exact_gradient():
    mesh=box_tetra_mesh([[-.2,.2]]*3,[3,3,3],lambda p: 0.)
    vec=np.array([.1,-.2,.5])
    def constant_A(x):
        return np.repeat(vec[None,:],len(x),axis=0)
    result=solve_aphi(mesh,1000.,[constant_A],outer_bc='total_zero')
    probes=np.array([[0,0,0],[.015,.022,-.04],[.1,.04,.03]])
    got=reaction_potential_at_points(result,probes)
    assert_allclose(got,np.broadcast_to(-vec,got.shape),rtol=0,atol=1e-12)
    # Gradient potential contributes zero along a physically closed path.
    z=closed_source_port_impedance(result,1000.,[[(1.,_loop())]])
    assert abs(z[0,0])<1e-10


def test_real_source_loop_voltage_orientation_and_no_extrapolation():
    mesh=box_tetra_mesh([[-.2,.2]]*3,[4,4,4],
                        lambda p:5. if max(abs(p))>.115 else 0.)
    path=_loop()
    result=solve_aphi(mesh,10000.,[closed_polyline_potential(path)],
                      outer_bc='total_zero')
    impedance=closed_source_port_impedance(result,10000.,
                                          [[(1.,path)]],subsegments=2)
    reverse=closed_source_port_impedance(result,10000.,
                                        [[(1.,path[::-1])]],subsegments=2)
    assert_allclose(reverse,-impedance,rtol=1e-9,atol=1e-12)
    assert impedance[0,0].real>0
    with pytest.raises(ValueError,match='outside'):
        reaction_potential_at_points(result,np.array([[2.,2.,2.]]))


def test_tetra_exact_closed_port_flux_and_shared_facet():
    """Real analytic Stokes flux, reverse direction, gauge and face tie."""
    from types import SimpleNamespace
    from sdfmpneo_bfzi.direct3d_aphi import _topology
    from sdfmpneo_bfzi.direct3d_port import closed_source_port_impedance_exact

    def whitney_representation(analytic_A, n):
        mesh = box_tetra_mesh([[-.25, .25]]*3, [n]*3, lambda x: 0.)
        edges, *_ = _topology(mesh)
        coefficients = np.zeros(len(edges), complex)
        for (i, j), edge in edges.items():
            a, b = mesh.nodes[i], mesh.nodes[j]
            coefficients[edge] = np.dot(analytic_A((a+b)/2), b-a)
        return SimpleNamespace(mesh=mesh,
                               edge_coefficients=coefficients[None, :])

    B0 = .19
    A = lambda x: .5*B0*np.array([-x[1], x[0], 0.])
    for z, n in ((.013, 5), (0., 4)):
        result = whitney_representation(A, n)
        path = np.array([[-.075,-.065,z],[.08,-.065,z],
                         [.08,.073,z],[-.075,.073,z],
                         [-.075,-.065,z]])
        za = closed_source_port_impedance_exact(result, 400., [[(1.,path)]])
        true_flux = B0*(.08+.075)*(.073+.065)
        assert_allclose(za[0,0]/(1j*2*np.pi*400), true_flux,
                        rtol=0, atol=2e-14)
        zr = closed_source_port_impedance_exact(result, 400.,
                                                [[(1.,path[::-1])]])
        assert_allclose(zr, -za, rtol=0, atol=2e-12)

    result = whitney_representation(
        lambda x: np.array([.3, -.2, .6]), 5)
    zg = closed_source_port_impedance_exact(result, 400., [[(1.,path)]])
    assert abs(zg[0,0]) < 1e-12
