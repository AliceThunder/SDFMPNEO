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
    with pytest.raises(ValueError,match='5, 32, 48 or 64'):
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


def test_s5_skin_modes_recover_local_hot_water():
    """Explicit TX/RX shell modes prevent smoothing away 2mm skin heating."""
    from examples.bfzi_geom_basis_check import metrics
    h=Head()
    nwet=h.ns
    nskin=1024
    rng=np.random.default_rng(23)
    h.nwet=nwet
    h.nskin=nskin
    h.skin_pkg=np.repeat(np.arange(2),nskin//2)
    mid=h.pkg_xyz.mean(axis=1)
    xyz=np.vstack([
        rng.uniform(-.1,.1,(nskin//2,3))+mid[0],
        rng.uniform(-.1,.1,(nskin//2,3))+mid[1]])
    h.xyz=np.vstack([h.xyz,xyz])
    h.ns=nwet+nskin
    h.cap=np.r_[h.cap[:nwet],np.full(nskin,.12),h.cap[nwet:]]
    h.cu=slice(len(h.cap)-2,len(h.cap))
    h.H=diags(np.ones(len(h.cap))).tocsr()
    h.amb=293.15
    theta=np.zeros(len(h.cap))
    theta[:nwet]=.5
    theta[nwet:nwet+nskin]=12.+3*h.skin_pkg
    theta[h.ns:h.ns+2*h.npkg]=3.
    theta[h.cu]=17.
    temp=h.amb+np.stack([theta,1.1*theta])
    result=metrics(h,temp,rank=32)
    assert result["regions"]["sea_skin"]["max_abs_K"]<1e-7
    assert result["regions"]["sea_bulk"]["max_abs_K"]<1e-7
    assert result["material_mean_max_K"]<1e-8
    phi=basis(h)
    assert np.max(abs(phi.T@(h.cap[:,None]*phi)-np.eye(32)))<1e-9


def test_s5_skin_labels_must_match_two_packages():
    h=Head()
    h.nwet=h.ns
    h.nskin=4
    h.skin_pkg=np.array([0,1,0,3])
    h.xyz=np.r_[h.xyz,h.xyz[:4]]
    h.ns+=4
    h.cap=np.r_[h.cap[:h.nwet],np.ones(4),h.cap[h.nwet:]]
    with pytest.raises(ValueError,match="package labels"):
        basis(h)


def test_positive_spatial_projection_keeps_material_means():
    from sdfmpneo_bfzi.geom_neural import _pos_beta
    h=Head()
    rom=GeomHeatROM(h,basis(h))
    a=(np.ones(len(h.cap))*h.cap)@rom.phi
    a*=.5
    a[5:]=0.
    a[5]=100.
    pre=rom.phi@a
    assert pre.min()<0.
    beta=_pos_beta(rom,a)
    assert 0<=beta<1.
    projected=a.copy()
    projected[5:]*=beta*.999
    temp=rom.phi@projected
    assert temp.min()>=-1e-10
    for sl in groups(rom):
        before=float(pre[sl]@rom.cap[sl]/rom.cap[sl].sum())
        after=float(temp[sl]@rom.cap[sl]/rom.cap[sl].sum())
        assert abs(after-before)<1e-10

def test_64_mode_bulk_skin_geometry_nested():
    from sdfmpneo_bfzi.geom_heat import GeomHeat
    head=GeomHeat.load('runs/bfzi_s5_D02_ultra_v4_anchor.npz')
    p32=basis(head,32)
    p48=basis(head,48)
    p64=basis(head,64)
    assert np.max(abs(p64[:,:32]-p32))<1e-9
    assert np.max(abs(p64[:,:48]-p48))<1e-9
    assert np.max(abs(p64.T@(head.cap[:,None]*p64)-np.eye(64)))<1e-8
    assert np.max(abs(p64[head.ns:,48:]))<1e-12
    # Orthonormalizing new bulk-centered modes against the retained
    # 48 sea modes also induces small components on the seawater skin.
    assert np.max(abs(head.cap[:head.ns] @ p64[:head.ns,48:]))<1e-7
    bulk=float(np.sum(head.cap[:head.nwet,None]*p64[:head.nwet,48:]**2))
    skin=float(np.sum(head.cap[head.nwet:head.ns,None]*
                      p64[head.nwet:head.ns,48:]**2))
    assert bulk>skin
