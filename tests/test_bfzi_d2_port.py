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
