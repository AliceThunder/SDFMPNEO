"""Native BFZI pose and independent rotating-package geometry tests."""
from types import SimpleNamespace

import numpy as np

from sdfmpneo_bfzi.pose import axes,desc,default_desc,isinside
from sdfmpneo_bfzi.geom_memory import geom_feat,GeomFrac


def coil(center,roll=90.,yaw=0.,pitch=0.):
    return SimpleNamespace(center=center,roll_deg=roll,
                           yaw_deg=yaw,pitch_deg=pitch,plane_angle_deg=90.)


def test_original_bfzi_rotation_and_mixed_pose_descriptor():
    c1=coil((0.,-.02,0.))
    c2=coil((.005,.02,.003),92.,1.,1.)
    A,B=axes(c1),axes(c2)
    assert np.allclose(A,np.eye(3),atol=1e-14)
    assert np.allclose(B@B.T,np.eye(3),atol=1e-14)
    actual=np.degrees(np.arccos(np.clip(abs(A[:,1]@B[:,1]),-1,1)))
    assert np.isclose(actual,2.2203117576865186,atol=1e-7)
    p=desc([c1,c2],[
        (c1.center,(.102,.0055,.102)),
        (c2.center,(.117,.0055,.117))])
    assert p.shape==(18,)
    assert np.allclose(p[:3],[.005,.04,.003])
    assert np.allclose(p[3:12],B.ravel())
    assert np.allclose(p[-6:],[.102,.0055,.102,.117,.0055,.117])


def test_package_rotated_local_query_and_explicit_model_features():
    c=coil((0.,0.,0.),135.)
    R=axes(c)
    half=np.array([.12,.006,.12])
    local=np.array([[.10,0.,0.],[0.,0.,.10],[.11,0.,.01]])
    xyz=local@R.T
    assert np.all(isinside(xyz,np.zeros(3),half,R))
    assert not np.all(isinside(xyz,np.zeros(3),half,np.eye(3)))
    other=np.array([[.16,0.,0.]])@R.T
    assert not isinside(other,np.zeros(3),half,R)[0]
    g=np.array([.09,.09,0.,0.,1.,1.,10.,10.,.00079,.00079,.04,
                16e-6,5e-6,16e-6,35.,35.])
    z=np.array([.2,10.,.1,4.,.2,10.])
    pose=default_desc(g)
    old=geom_feat(g,z)
    full=geom_feat(g,z,pose)
    assert old.shape==(22,) and full.shape==(40,)
    assert np.allclose(full[:22],old)
    net=GeomFrac(ng=len(full))
    assert net.ng==40
