"""Check original BFZI AABB sea void versus oriented dual-package solids."""
from types import SimpleNamespace

import numpy as np

from sdfmpneo_bfzi.geom_void import overlap,report,disjoint_y_shift


def sample(sep=.04,tilt=27.):
    a=np.deg2rad(tilt)
    R=np.array([[1.,0.,0.],
                [0.,np.cos(a),-np.sin(a)],
                [0.,np.sin(a),np.cos(a)]])
    mid=np.array([[0.,0.,0.],[0.,sep,0.]])
    half=np.array([[.1,.0055,.1],[.1,.0055,.1]])
    rot=np.stack([np.eye(3),R])
    boxes=[]
    for i in range(2):
        w=np.abs(rot[i])@half[i]
        boxes.append({"min_m":(mid[i]-w).tolist(),
                      "max_m":(mid[i]+w).tolist()})
    head=SimpleNamespace(
        _pkg_mid=mid,_pkg_half=half,_pkg_axes=rot)
    return head,{"inner_boxes":boxes}


def test_rotated_obb_collision_and_aabb_overexclusion():
    head,mesh=sample()
    assert overlap(head._pkg_mid,head._pkg_half,head._pkg_axes)
    d=report(head,mesh)
    assert d["physical_packages_overlap"]
    assert d["BFZI_AABB_over_physical_sum"]>2
    assert d["AABB_void_excess_lower_bound_m3"]>0
    dy=disjoint_y_shift(head._pkg_mid,head._pkg_half,head._pkg_axes)
    assert dy>.005
    new=head._pkg_mid.copy()
    new[1,1]+=dy
    assert not overlap(new,head._pkg_half,head._pkg_axes)
    assert np.isclose(d["BFZI_AABB_to_physical_box_ratio"][0],1)


def test_noncolliding_packages():
    head,mesh=sample(sep=.3)
    assert not overlap(head._pkg_mid,head._pkg_half,head._pkg_axes)
    assert not report(head,mesh)["physical_packages_overlap"]
    assert disjoint_y_shift(head._pkg_mid,head._pkg_half,head._pkg_axes)==0.
