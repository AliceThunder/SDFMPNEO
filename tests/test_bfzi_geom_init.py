"""Geometry-only cold/S5 shared operator, independent of BFZI runtime."""
import numpy as np
import pytest
from sdfmpneo_bfzi.geom_init import GeomInit,encode


def _case(a=0.8):
    return dict(donor='g1',alpha_per_K=.020,
        geometry=dict(alpha1=a,alpha2=.95,core1_mm=19,core2_mm=15,
            shift_mm=[2,94,0],tilt_deg=10,yaw_deg=2,pitch_deg=3,
            box1_mm=221,box2_mm=230),
        thermal=dict(ins_k=.2,ins_t=.0004,pkg_k=.25,pkg_sea_h=20.))


def test_encode_uses_only_geometric_and_material_inputs():
    assert encode(_case()).shape==(18,)
    assert not np.allclose(encode(_case(.8)),encode(_case(.9)))


def test_shared_operator_stays_symmetric_and_dissipative(tmp_path):
    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((3,14))
    y[:,:6]=[1,10,.1,.5,1,10]
    y[:,6:9]=[1e-5,1e-6,1e-5]
    y[:,9:14]=np.log([1e5,500,500,20,20])
    k=np.stack([np.eye(32)*v for v in (.01,.02,.03)])
    f=np.ones((3,32,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    model=GeomInit.fit(x,y,k,f,ext)
    got=model.predict(_case(.86))
    assert got['k'].shape==(32,32)
    assert np.linalg.eigvalsh(got['k'])[0]>0
    assert np.allclose(got['k'],got['k'].T)
    assert np.all(got['cap']>0)
    assert got['src'].shape==(32,3)
    assert got['z0'].shape==(6,)
    model.save(tmp_path/'atlas.npz')
    restored=GeomInit.load(tmp_path/'atlas.npz')
    assert np.allclose(restored.predict(_case(.86))['k'],got['k'])
    with pytest.raises(ValueError):
        model.predict(_case(),external=np.ones((3,3)))

def test_shared_modal_order_accepts_64_coordinates():
    cases=[_case(.8),_case(.9),_case(.95)]
    x=np.stack([encode(c) for c in cases])
    y=np.zeros((3,14));y[:,9:]=np.log([1e5,400,400,20,20])
    k=np.stack([np.eye(64)*v for v in (.01,.02,.03)])
    f=np.ones((3,64,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    model=GeomInit.fit(x,y,k,f,ext)
    pred=model.predict(_case(.86))
    assert model.rank==64
    assert pred['k'].shape==(64,64)
    assert pred['src'].shape==(64,3)
    assert np.linalg.eigvalsh(pred['k'])[0]>0
