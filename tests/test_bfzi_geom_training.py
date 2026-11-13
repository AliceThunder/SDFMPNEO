"""Focused BFZI causal closed-loop, geometry balancing and ensemble tests."""
import numpy as np
import torch

from sdfmpneo_bfzi.geom_train import zmat,source,closed,weights,loss
from sdfmpneo_bfzi.geom_ensemble import Multi


class Toy:
    ng=4

    def __init__(self):
        self.a=torch.nn.Parameter(torch.tensor(.003))

    def init_state(self,n):
        return torch.zeros(n,1)

    def step(self,h,u,dt,geom):
        h=.97*h+u.mean(-1,keepdim=True)*dt*.01
        z=self.a*(u+geom[:,:3])
        t=torch.relu(h+self.a).repeat(1,8)
        return h,torch.cat((z,z,t),dim=1)


def batch():
    b,n=5,7
    p=dict(
        cold=torch.tensor(np.tile([.3,20.,.02,1.5,.4,30.],(b,1)),
                          dtype=torch.float32),
        ext=torch.diag(torch.tensor([.5,.5],dtype=torch.complex64)
                       ).expand(b,2,2),
        air=torch.diag(torch.tensor([3.,4.],dtype=torch.complex64)
                       ).expand(b,2,2),
        cu=torch.full((b,2),.08+.002j,dtype=torch.complex64),
        slope=torch.full((b,2),.0003+.00002j,dtype=torch.complex64),
        cap=torch.full((b,2),30.),
        volts=torch.tensor(np.tile([0,50,50,0,80,80,0],(b,1)),
                           dtype=torch.float32))
    return p,torch.rand(b,n,3)*.3,torch.rand(b,4),np.arange(n)*15.


def test_closed_teacher_mix_exact():
    p,src,g,t=batch()
    net=Toy()
    got=closed(net,g,t,src,np.ones(3),np.ones(14),p,mix=0)
    h=net.init_state(len(g));expect=[torch.zeros(len(g),14)]
    for k in range(1,len(t)):
        h,v=net.step(h,src[:,k],15.,g)
        expect.append(v)
    assert torch.allclose(got,torch.stack(expect,dim=1),atol=2e-6)


def test_closed_physical_power_gradient_and_loss():
    p,src,g,t=batch()
    net=Toy()
    y=closed(net,g,t,src,np.ones(3),np.ones(14),p,mix=1.)
    obj=loss(y,torch.zeros_like(y),torch.zeros(5,7,4),
             np.ones(14),torch.ones(5))
    obj.backward()
    assert torch.isfinite(net.a.grad) and net.a.grad.abs()>1e-9
    u=source(zmat(p["cold"]),torch.zeros(5,2),p["volts"][:,1],p)
    assert torch.isfinite(u).all()
    assert (u>=0).all()


def test_equal_geometry_weight_and_invalid_ensemble():
    d=[(dict(sample="M4"),0)]*6+[(dict(sample="P01"),0)]*2
    w=weights(d)
    assert torch.isclose(w[:6].sum(),w[6:].sum())
    assert torch.isclose(w.mean(),torch.tensor(1.))
    try:
        Multi([Toy(),Toy()],[-1,1])
    except ValueError:
        pass
    else:
        raise AssertionError("negative BFZI model weight accepted")


class Const:
    def __init__(self,ng,level):
        self.ng=ng
        self.level=level

    def init_state(self,n):
        return torch.zeros(n,1)

    def step(self,h,u,dt,g):
        assert g.shape[-1]==self.ng
        return h+dt,torch.full((len(u),14),self.level)


def test_mixed_22d_40d_ensemble_one_common_port():
    net=Multi([Const(22,2.),Const(40,5.)],[.75,.25])
    h=net.init_state(3)
    nxt,y=net.step(h,torch.ones(3,3),30.,torch.ones(3,40))
    assert net.ng==40 and len(nxt)==2
    assert torch.allclose(y,torch.full((3,14),2.75))
