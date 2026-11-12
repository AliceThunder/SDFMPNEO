"""Fractional memory does not require equal-sized original BFZI sea grids."""
import numpy as np
import torch

from sdfmpneo_bfzi.geom_memory import (
    GeomFrac,geom_feat,thermal_summary,rollout,
)


def test_geometry_summary_independent_of_fv_cell_count():
    for ns in (384,2250):
        nt=3
        t=np.full((nt,ns+66),293.15)
        t[:,ns:ns+32]=293.15+np.arange(nt)[:,None]*2.
        t[:,ns+32:ns+64]=293.15+np.arange(nt)[:,None]*3.
        t[:,-2:]=293.15+np.arange(nt)[:,None]*4.
        s=thermal_summary(t,ns,32,np.ones(ns),293.15)
        assert s.shape==(3,8)
        assert np.allclose(s[0],0)
        assert np.allclose(s[2,2:6],[4.,6.,4.,6.])


def test_geom_fractional_memory_and_unseen_geometry_rollout():
    torch.manual_seed(5)
    net=GeomFrac()
    times=np.array([0.,30.,60.,90.])
    geom=np.array([.09,.09,0.,0.,1.,1.,10.,10.,
                   .0008,.0008,.04,
                   15e-6,5e-6,15e-6,.1,.1])
    z0=np.array([.25,20.,.1,6.,.25,20.])
    x=geom_feat(geom,z0)
    assert len(x)==net.ng
    src=torch.ones((2,4,3))
    g=torch.tensor(np.repeat(x[None],2,axis=0),
                   dtype=torch.float32)
    pred=net(src,times,g)
    assert pred.shape==(2,4,14)
    assert torch.all(pred[:,0]==0)
    assert torch.all(pred[:,:,6:]>=0)
    pred.sum().backward()
    assert torch.linalg.vector_norm(net.raw.grad)>0
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14))
    v=np.array([80.,80.,0.,80.])
    out=rollout(net,ck,v,times,geom,z0,np.diag([.5,.5]))
    assert out["z6"].shape==(4,6)
    assert out["temp_rise_K"].shape==(4,8)
    assert np.all(out["temp_rise_K"]>=0)
    assert np.allclose(out["z6"][0],z0)
    assert np.allclose(out["current_A"][2],0)


def test_original_bfzi_geometry_source_partition():
    from sdfmpneo_bfzi.geom_memory import geom_port,heat_drive
    g=np.array([.09,.11,.0,.02,1.,.5,10.,13.,
                .00079,.00079,.08,16e-6,4e-6,31e-6,35.,56.])
    port=geom_port(g)
    t=np.array([293.15,320.])
    sea=np.array([[.12+.002j,.07-.003j],
                  [.07-.003j,.19+.004j]])
    z=port.total(sea,t)
    ex=np.diag([.5-.3j,.5-.3j])
    v=75.
    u=heat_drive(v,z,ex,port,t,g[14:16])
    I=np.linalg.solve(z+ex,np.array([v,0],complex))
    cu=port.copper(t)
    expected=np.r_[
        .5*np.vdot(I,sea@I).real,
        .5*cu.real*abs(I)**2/g[14:16]]
    assert np.allclose(u,expected,rtol=1e-12,atol=1e-12)
    assert np.all(heat_drive(0.,z,ex,port,t,g[14:16])==0.)
