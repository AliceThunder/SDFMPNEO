"""Geometry-independent thermal modes and autonomous neural closure."""
import numpy as np
import torch
import pytest
from scipy.sparse import diags

from sdfmpneo_bfzi.geom_canon import basis, mode_scale, project
from sdfmpneo_bfzi.geom_rom import GeomHeatROM, groups
from sdfmpneo_bfzi.geom_neural import GeomFracET, rollout
from examples.bfzi_geom_neural import scales,score
from examples import bfzi_geom_multi as multi


class Head:
    def __init__(self,scale=1.,offset=0.):
        rng=np.random.default_rng(19)
        self.ns=350
        self.npkg=32
        xyz=rng.uniform(-.15,.15,(self.ns,3))
        xyz[:,0]+=offset
        self.xyz=xyz*scale
        x=np.linspace(-1.,1.,4)
        pts=np.stack(np.meshgrid(x,[-.5,.5],x,indexing='ij'),axis=-1).reshape(-1,3)
        self.pkg_xyz=np.stack([pts*.05+np.array([0,j*.08,0]) for j in range(2)])*scale
        self._pkg_axes=np.tile(np.eye(3),(2,1,1))
        self._pkg_half=np.tile(np.array([.05,.04,.06])*scale,(2,1))
        self.cap=np.r_[rng.uniform(5.,12.,self.ns),rng.uniform(2.,6.,64),[8.,9.]]
        self.H=diags(np.ones(len(self.cap))).tocsr()
        self.amb=293.15
        self.pose=np.ones(18)
        self.geom=np.ones(16)*.09
        self.z0=np.array([.2,1.,.05,.2,.2,1.])
        self.ext=np.eye(2)
        self.vol=np.ones(self.ns)*.01
        self.sea_proj=None
        self.cu=slice(len(self.cap)-2,len(self.cap))
        class Port:
            omega=1.
            air=np.zeros((2,2))
            def copper(self,t):
                return np.ones(2)*.01
        self.port=Port()


def test_canonical_mode_mean_and_capacity_orthogonality():
    h=Head()
    p=basis(h)
    assert p.shape==(len(h.cap),32)
    assert np.allclose(p.T@(h.cap[:,None]*p),np.eye(32),atol=1e-9)
    t=np.zeros(len(h.cap))
    for j,sl in enumerate(groups(h)):
        t[sl]=j+2
    a=project(h,t,p)
    assert np.allclose(a[:5],[2,3,4,5,6])
    assert np.max(abs(a[5:]))<1e-9


def test_two_geometries_same_thermal_semantics():
    a,b=Head(1.),Head(1.07,.02)
    x=project(a,np.ones(len(a.cap))*4.,basis(a))
    y=project(b,np.ones(len(b.cap))*4.,basis(b))
    assert np.allclose(x[:5],y[:5])
    assert np.linalg.norm(x[5:])<1e-9
    assert np.linalg.norm(y[5:])<1e-9
    with pytest.raises(ValueError,match='5 or 32'):
        basis(a,24)


def test_canonical_autonomous_five_budget_conservation():
    h=Head()
    rom=GeomHeatROM(h,basis(h))
    def circuit(theta,dsea,voltage):
        z=np.array([[.2+1j,.05+.2j],[.05+.2j,.2+1j]])+dsea
        I=np.linalg.solve(z+rom.ext,np.array([voltage,0.],complex))
        return z,z,I
    def source(theta,I,sea):
        heat=np.zeros(len(rom.cap))
        heat[:rom.ns]=.002*abs(I[0])**2
        heat[rom.cu]=.001*abs(I)**2
        return (heat,np.full(rom.ns,.002*abs(I[0])**2),
                np.zeros((rom.ns,3),complex),float(heat[:rom.ns].sum()),
                heat[rom.cu])
    rom.circuit=circuit
    rom.source=source
    model=GeomFracET(ng=40,rank=32)
    for p in model.net.parameters():
        torch.nn.init.zeros_(p)
    ck=dict(canonical=True,port_sea=True,src_scale=np.ones(3),
            y_scale=np.ones(14),a_scale=np.ones(32))
    v=np.array([0.,40.,0.,0.])
    t=np.array([0.,15.,30.,45.])
    res=rollout(rom,model,ck,v,t)
    assert np.max(abs(res['material_budget_W']))<1e-7
    assert np.allclose(res['temp_K']-rom.amb,
                       (res['heat_state']*mode_scale(rom))@rom.phi.T)
    assert res['heat_implicit_solves_online']==0


def test_multi_geometry_shared_training_gradients(monkeypatch):
    h1,h2=Head(1.),Head(1.09,.02)
    heads={'T01':h1,'T02':h2}
    class MockHeat:
        @staticmethod
        def load(path):
            return heads[path]
    def fake_read(name,rom):
        time=np.array([0.,15.,30.,45.])
        temp=np.ones((len(time),len(rom.cap)))*rom.amb
        for k in range(1,len(time)):
            for j,sl in enumerate(groups(rom)):
                temp[k,sl]+=k*(j+1)*.12
        coef=((temp-rom.amb)*rom.cap)@rom.phi
        return dict(time=time,voltage=np.array([0.,60.,60.,0.]),
                    src=np.array([[0.,0.,0.],[.1,.2,.3],[.2,.15,.1],[0.,0.,0.]]),
                    y=np.zeros((4,14)),a=coef,
                    q=np.ones_like(coef)*.1,
                    temp_K=temp,z6=np.tile(rom.z0,(4,1)),file=name)
    monkeypatch.setattr(multi,'GeomHeat',MockHeat)
    monkeypatch.setattr(multi,'read',fake_read)
    rows=multi.prepare({'anchor':'T01','teacher':'a'}) + multi.prepare(
         {'anchor':'T02','teacher':'b'})
    assert np.allclose(rows[0]['a'][:,:5],rows[1]['a'][:,:5])
    for row in rows:
        rom=row['rom']
        m=mode_scale(rom)
        x=row['a'][2]
        assert np.allclose(row['K']@x,(rom.K@(x*m))/m)
        assert row['basis_max_K']<1e-9
    sx,sy,sa=scales(rows)
    model=GeomFracET(ng=40,rank=32)
    loss=score(model,rows,None,sx,sy,sa,therm_w=.03)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.heat_head[-1].weight.grad.abs().sum()>0
