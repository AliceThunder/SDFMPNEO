"""Minimal cold-data and geometry-only cold network consistency tests."""
from __future__ import annotations

import numpy as np

from examples.bfzi_cold_plan import sample,_key,_coil
from sdfmpneo_bfzi.geom_coldnet import feat,target,unpack


def test_cold_plan_determinism_and_disjoint():
    a=sample(seed=17,n=40)
    b=sample(seed=17,n=40)
    assert a==b
    ids=a['train']+a['val']+a['test']
    assert len(ids)==len(set(ids))==40
    assert len(a['train'])==32 and len(a['val'])==len(a['test'])==4
    assert len({_key(g) for g in a['cases'].values()})==40
    for g in a['cases'].values():
        assert all(_coil(g[f'alpha{j}'],g[f'core{j}_mm']) for j in (1,2))


def test_target_roundtrip_and_features():
    g=dict(alpha1=.84,alpha2=.93,core1_mm=13,core2_mm=17,
           shift_mm=[2,90,-4],tilt_deg=6,yaw_deg=-2,pitch_deg=7,
           box1_mm=221,box2_mm=233)
    row=dict(geometry=g,z0=[.25,18.,.03,1.2,.27,19.],
             air=[14.2e-6,1.1e-6,15.0e-6])
    x=feat(g)
    assert x.shape==(16,) and np.isfinite(x).all()
    z,air=unpack(target(row))
    np.testing.assert_allclose(z,row['z0'],rtol=1e-6,atol=1e-6)
    np.testing.assert_allclose(air,row['air'],rtol=1e-6,atol=1e-11)


def test_cold_plan_skips_existing():
    baseline=sample(seed=29,n=20)
    old=list(baseline['cases'].values())
    new=sample(seed=29,n=20,existing=old)
    assert not set(map(_key,old))&set(map(_key,new['cases'].values()))
