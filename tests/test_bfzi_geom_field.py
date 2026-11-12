"""Focused tests for material-mean/hotspot preserving 3-D BFZI decoder."""
import numpy as np
import torch

from sdfmpneo_bfzi.geom_field import (
    SpaceTemp,space_feat,time_feat,calibrate,infer_field,
)


def _case():
    g=np.array([.09885,.11885,.04385,.05385,
                .5,.5,10.,13.,.00079,.00079,.1,
                18e-6,3.5e-6,32e-6,40.,57.])
    z0=np.array([.36,23.,.21,4.35,.75,41.32])
    z=np.linspace(-.09,.09,4)
    x=np.array([[a,0,b] for a in z for b in z])
    boxes=np.stack([
        np.array([[a,y,b] for a in z for y in (-.05,.05)
                  for b in z[:1]])
        for _ in range(2)])
    boxes[0,:,1]-=.05
    boxes[1,:,1]+=.05
    return g,z0,x,boxes


def test_bfzi_coordinate_and_gradient():
    g,z0,x,boxes=_case()
    features=[space_feat(arr,g,z0,boxes,k)
              for k,arr in enumerate((x,boxes[0],boxes[1]))]
    assert all(a.shape[1]==44 for a in features)
    h=time_feat(np.array([.01,.7,1.,1.5,1.2,2.,.2,8.]),300.)
    assert h.shape==(9,)
    net=SpaceTemp()
    t=torch.tensor(np.column_stack(
        (features[0],np.broadcast_to(h,(len(x),9)))))
    y=net(t)
    assert y.shape==(len(x),)
    assert torch.all((y>0)&(y<1))
    y.mean().backward()
    assert net.net[0].weight.grad is not None


def test_exact_material_hotspot_mean_and_cold():
    raw=np.linspace(.01,1.,384)**2
    weights=1.+np.arange(384)/384.
    values=calibrate(raw,.035,1.2,weights)
    assert np.min(values)>=0
    assert np.isclose(np.max(values),1.2,atol=1e-8)
    assert np.isclose(np.average(values,weights=weights),.035,atol=1e-8)
    assert np.all(calibrate(raw,0,0)==0)
    uniform=calibrate(raw,1.999,2.)
    assert np.isclose(uniform.mean(),1.999,atol=1e-6)
    assert np.isclose(uniform.max(),2.,atol=1e-8)


def test_material_field_inference_uses_any_cell_count():
    g,z0,x,boxes=_case()
    features=[space_feat(arr,g,z0,boxes,k)
              for k,arr in enumerate((x,boxes[0],boxes[1]))]
    net=SpaceTemp()
    states=np.array([[0.]*8,
                     [.06,.5,.7,1.1,.9,1.5,.1,3.]])
    sea,pkg=infer_field(net,features,states,np.array([0.,30.]),
                        np.ones(len(x)))
    assert sea.shape==(2,len(x))
    assert pkg.shape==(2,2,len(boxes[0]))
    assert np.all(sea[0]==0) and np.all(pkg[0]==0)
    assert np.isclose(np.max(sea[1]),.5)
    assert np.isclose(sea[1].mean(),.06,atol=1e-6)
    for p in range(2):
        assert np.isclose(np.max(pkg[1,p]),states[1,4+p])
        assert np.isclose(np.mean(pkg[1,p]),states[1,2+p],atol=1e-6)
