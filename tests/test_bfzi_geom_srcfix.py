"""Positive source geometry-only rank32 and roundtrip."""
import numpy as np
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def _case(a):
    return dict(geometry=dict(alpha1=a,alpha2=.8,core1_mm=11,
      core2_mm=18,shift_mm=[0,90,1],tilt_deg=12,yaw_deg=3,
      pitch_deg=2,box1_mm=224,box2_mm=228),
      thermal=dict(ins_k=.24,ins_t=.0004,pkg_k=.26,pkg_sea_h=20),
      alpha_per_K=.022,donor='g1')


def test_shared_seawater_source_only(tmp_path):
    rng=np.random.default_rng(3)
    rows=[dict(case=_case(a),src=rng.normal(size=(32,3)))
          for a in (.71,.77,.84,.92)]
    net=SrcFix.fit(rows)
    init=dict(src=np.zeros((32,3)),k=np.eye(32))
    got=net.apply(_case(.85),init)
    assert got['src'].shape==(32,3)
    assert np.isfinite(got['src']).all()
    assert got['k'] is init['k']
    assert np.all(init['src']==0)
    net.save(tmp_path/'s.npz')
    assert np.allclose(SrcFix.load(tmp_path/'s.npz').predict(_case(.85)),
                       net.predict(_case(.85)))
