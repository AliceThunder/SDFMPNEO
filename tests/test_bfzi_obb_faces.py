"""Exact wet FV face geometry for BFZI oriented package exclusions."""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csr_matrix

from sdfmpneo_bfzi.obb_faces import wet_area,apply


def recs():
    return (
        dict(cell_min=np.array([-.5,-.5,-.5]),
             cell_max=np.array([0.,.5,.5])),
        dict(cell_min=np.array([0.,-.5,-.5]),
             cell_max=np.array([.5,.5,.5])))


def test_face_area_subtracts_exact_oriented_package_section():
    a,b=recs()
    mid=np.zeros(3)
    half=np.array([.2,.2,.2])
    ang=np.deg2rad(27)
    rot=np.array([[1,0,0],
                  [0,np.cos(ang),-np.sin(ang)],
                  [0,np.sin(ang),np.cos(ang)]])
    assert np.isclose(wet_area(a,b,0,[(mid,half,np.eye(3))]),.84)
    assert np.isclose(wet_area(a,b,0,[(mid,half,rot)]),.84,atol=1e-12)
    other=np.array([0.,.42,0.])
    boxes=[(mid,half,np.eye(3)),
           (other,np.full(3,.05),np.eye(3))]
    assert np.isclose(wet_area(a,b,0,boxes),.83,atol=1e-12)


def test_original_FV_projector_face_weights_are_rebuilt():
    r1,r2=recs()
    B=csr_matrix([[-2.,2.]])
    proj=SimpleNamespace(
        edges=[(0,1,0,.5,1.,1.)],weights=np.array([.5]),
        B=B,n=2,meta={})
    info=dict(void_type="oriented",
              physical_package_center_m=[[0,0,0],[0,.42,0]],
              physical_package_half_m=[[.2,.2,.2],[.05,.05,.05]],
              physical_package_rot=[np.eye(3),np.eye(3)])
    result=apply(proj,[r1,r2],info)
    assert result is proj
    assert np.isclose(proj.edges[0][4],.83)
    assert np.isclose(proj.weights[0],.415)
    assert proj.meta["face_area_corrected_edges"]==1
    assert np.all(np.isfinite(proj.solve(np.array([0.,1.]))))


def test_non_oriented_FV_left_unchanged():
    proj=SimpleNamespace(edges=[(0,1,0,.5,1.,1.)],weights=np.array([.5]))
    assert apply(proj,[],{"void_type":"aabb"}) is proj
    assert proj.weights[0]==.5
