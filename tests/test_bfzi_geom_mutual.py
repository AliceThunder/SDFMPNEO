"""No-query-physics BFZI mutual-port component regression."""
import numpy as np
from sdfmpneo_bfzi.geom_mutual import MutFix,feat


def case(a=.8,shift=90):
    return dict(geometry=dict(alpha1=a,alpha2=.9,core1_mm=12.,
      core2_mm=18.,shift_mm=[4.,shift,-3.],
      tilt_deg=9.,yaw_deg=4.,pitch_deg=5.))


def test_mutual_pose_and_port_consistency(tmp_path):
    rows=[]
    for a in (.75,.83,.91):
        z=np.array([.25,18.,.05+a*.01,.85+a*.2,.26,19.,
                    1.1e-5,(.86+a*.2)/1256637.,1.2e-5])
        rows.append(dict(case=case(a),y=z))
    m=MutFix.fit(rows)
    ini=dict(z0=np.array([.25,18.,.05,.9,.26,19.]),
             air=np.array([1e-5,1e-6,1.2e-5]),k=np.eye(5))
    out=m.apply(case(.83),ini)
    assert out['z0'].shape==(6,)
    assert np.array_equal(out['z0'][[0,1,4,5]],ini['z0'][[0,1,4,5]])
    assert out['air'][0]==ini['air'][0] and out['air'][2]==ini['air'][2]
    assert np.all(np.isfinite(out['z0']))
    assert out['k'] is ini['k']
    m.save(tmp_path/'m.npz')
    assert np.allclose(MutFix.load(tmp_path/'m.npz').predict(case(.83)),
                       m.predict(case(.83)))
