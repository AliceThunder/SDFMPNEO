"""Original BFZI oriented package interface conductance diagnostics."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import lil_matrix
from sdfmpneo_bfzi.geom_interface import report


def fixture():
    ns=4; n=8
    H=lil_matrix((n,n))
    for s,p,g in ((0,4,.6),(1,5,.9)):
        H[s,s]+=g;H[p,p]+=g;H[s,p]-=g;H[p,s]-=g
    return SimpleNamespace(
        ns=ns,npkg=1,cap=np.ones(n),H=H.tocsr(),
        xyz=np.array([[.15,0,0],[1.05,0,0],[3,0,0],[4,0,0]]),
        _pkg_mid=np.array([[0,0,0],[1,0,0]]),
        _pkg_half=np.full((2,3),.1),
        _pkg_axes=np.tile(np.eye(3),(2,1,1)))


def test_rotated_package_boundary_and_interface_flux():
    head=fixture()
    theta=np.array([0,0,0,0,2,3,0,0],float)
    result=report(head,theta)
    assert len(result)==2
    assert [r["links"] for r in result]==[1,1]
    assert [r["inside_box_links"] for r in result]==[0,1]
    assert np.allclose([r["conductance_W_K"] for r in result],[.6,.9])
    assert np.allclose([r["package_to_sea_W"] for r in result],
                       [1.2,2.7])
    assert all(r["reciprocity_gap_W_K"]==0 for r in result)
    assert np.isclose(result[0]["effective_h_W_m2K"],.6/.24)


def test_bad_interface_temperature_shape():
    with pytest.raises(ValueError):
        report(fixture(),np.zeros(5))
