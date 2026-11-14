"""Coil-local BFZI weak-source mask follows independent SE(3) package pose."""
from types import SimpleNamespace

import numpy as np

from sdfmpneo_bfzi.bfzi_v148_adapter import _oriented_mask
from sdfmpneo_bfzi.pose import axes


def coil(center,angle):
    return SimpleNamespace(center=tuple(center),yaw_deg=0.,
        pitch_deg=0.,roll_deg=90.+angle,plane_angle_deg=90.)


def test_weak_mass_uses_rotated_coil_frame():
    c=coil([.15,.05,-.02],27.)
    R=axes(c)
    local=np.array([[.05,.011,.07],[.05,.12,.07],[.16,.011,.07]])
    pts=np.asarray(c.center)+local@R.T
    lo=pts-.01
    hi=pts+.01
    line=pts[[0]]
    keep=_oriented_mask(pts,lo,hi,c,.102,.0055,.102,.102,line,.08)
    assert keep.tolist()==[True,False,False]
    moved=coil([-.10,.25,.15],27.)
    shift=np.asarray(moved.center)-np.asarray(c.center)
    assert np.array_equal(keep,_oriented_mask(
        pts+shift,lo+shift,hi+shift,moved,.102,.0055,.102,.102,
        line+shift,.08))


def test_nonrotated_mask_near_physical_package_face():
    c=coil([0.,0.,0.],0.)
    pt=np.array([[0.,.011,0.],[0.,.15,0.]])
    half=.006
    mask=_oriented_mask(pt,pt-half,pt+half,c,
                         .102,.0055,.102,.102,pt[:1],.08)
    assert mask.tolist()==[True,False]
