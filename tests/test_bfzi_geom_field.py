"""Mesh-free continuous 3-D temperature queries."""
import json
import numpy as np
from sdfmpneo_bfzi.geom_field import FieldAtlas


def test_meshfree_spatial_queries_from_geometry_only():
    plan=json.load(open('runs/geomfrac_s5_v9_plan.json'))
    case=plan['cases']['W01']
    field=FieldAtlas.load('runs/geomfrac_field_v1.npz')
    xyz=np.array([[.18,.02,.12],[-.15,.03,-.1],
                  [.01,0.,0.],[.01,.02,.0],[0.,0.,0.]])
    reg=np.array([0,0,1,2,3])
    state=np.zeros((4,32))
    out=field.query(case,xyz,reg,state)
    assert out.shape==(4,len(xyz))
    assert np.allclose(out,293.15)
    state[:,0]=2.
    state[:,1]=4.
    state[:,2]=6.
    state[:,3]=8.
    state[:,4]=10.
    out=field.query(case,xyz,reg,state)
    assert np.allclose(out[:,0:2],295.15)
    assert np.allclose(out[:,2],297.15)
    assert np.allclose(out[:,3],299.15)
    assert np.allclose(out[:,4],301.15)
    state[:,5]=.5
    assert np.isfinite(field.query(case,xyz,reg,state)).all()
    assert np.allclose(field.query(case,xyz,reg,state)[:,4],301.15)

def test_conservative_spatial_shape_centering():
    case=json.load(open('runs/geomfrac_s5_v9_plan.json'))['cases']['W02']
    field=FieldAtlas.load('runs/geomfrac_field_v1.npz')
    xyz=np.array([[.12,.2,.1],[-.11,.04,.12],[.01,0.,0.],
                  [-.03,.0,.05],[.02,.03,0.],[-.05,.02,.04]])
    reg=np.array([0,5,1,1,2,2])
    cap=np.array([2.,3.,5.,7.,11.,13.])
    sh=field.predict(case,xyz,reg,weights=cap)
    mat=np.where(reg>=5,0,reg)
    for m in (0,1,2):
        idx=mat==m
        target=np.zeros(32);target[m]=1.
        avg=np.average(sh[idx],axis=0,weights=cap[idx])
        assert np.allclose(avg,target,atol=1e-8)
    st=np.random.default_rng(17).normal(size=(4,32))
    temp=field.query(case,xyz,reg,st,weights=cap)
    for m in (0,1,2):
        idx=mat==m
        avg=np.average(temp[:,idx],axis=1,weights=cap[idx])
        assert np.allclose(avg,293.15+st[:,m],atol=1e-8)
