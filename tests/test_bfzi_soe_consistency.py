"""Fractional SOE quadrature, time partition and closed-loop source tests."""
import numpy as np
import torch

from sdfmpneo_bfzi.frac_memory import FracNet,build_sources,zmat
from sdfmpneo_bfzi.geom_memory import GeomFrac,heat_drive
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from examples.bfzi_geom_neural import _port_src
from test_bfzi_geom_heat import case


def test_log_quadrature_matches_bandlimited_fractional_step():
    from numpy.polynomial.legendre import leggauss
    lo,hi=1/2400.,1/12.
    nodes,weights=leggauss(140)
    xi=np.log(lo)+(.5+.5*nodes)*np.log(hi/lo)
    lam=np.exp(xi)
    quad=weights*.5*np.log(hi/lo)
    times=np.geomspace(15,600,120)
    net=GeomFrac(ng=22,soe_quad="trap")
    old=GeomFrac(ng=22,soe_quad="legacy")
    for alpha in (.24,.36,.536,.719,.903):
        raw=np.log((alpha-.12)/(.95-alpha))
        with torch.no_grad():
            net.raw.fill_(raw)
            old.raw.fill_(raw)
        coeff=quad*lam**(-alpha)
        coeff/=coeff.sum()
        ref=(coeff[:,None]*(-np.expm1(-lam[:,None]*times))).sum(axis=0)
        h=-np.expm1(-net.rates.detach().numpy()[:,None]*times)
        new=net.soe_weights().detach().numpy()[0]@h
        orig=old.soe_weights().detach().numpy()[0]@h
        assert np.max(abs(new/ref-1))<.013
        assert np.max(abs(new/ref-1))<.25*np.max(abs(orig/ref-1))


def test_fractional_kernel_has_finite_positive_gradient():
    net=FracNet(nout=6,mode="hybrid",soe_quad="trap")
    src=torch.rand((2,5,3))
    times=[0.,15.,30.,45.,60.]
    y=net(src,times)
    y.square().sum().backward()
    assert net.raw.grad.abs().sum()>1e-8
    assert net.raw_tau.grad.abs().sum()>1e-8
    assert torch.isfinite(y).all()


def test_neural_heat_integration_composes_across_sampling_steps():
    torch.manual_seed(14)
    net=GeomFracET(ng=22,rank=8,soe_quad="trap",heat_dt=1.875)
    with torch.no_grad():
        net.heat_head[-1].weight.normal_(std=.02)
    u=torch.tensor([[.4,.03,.05]],dtype=torch.float32)
    g=torch.randn(1,22)*.01
    a=torch.randn(1,8)*.03
    ac=torch.ones(8)
    h0=net.init_state(1)
    def integrate(n):
        h=h0.clone()
        st=a.clone()
        for _ in range(n):
            h,y,st=net.step_et(h,u,15./n,g,st,ac)
        return h,y,st
    h1,y1,a1=integrate(1)
    for n in (2,4,8):
        hn,yn,an=integrate(n)
        assert torch.allclose(hn,h1,atol=2e-6,rtol=1e-5)
        assert torch.allclose(yn,y1,atol=2e-6,rtol=1e-5)
        assert torch.allclose(an,a1,atol=2e-6,rtol=1e-5)
    a1.sum().backward()
    assert net.heat_head[-1].weight.grad.abs().sum()>0
    assert net.raw.grad.abs().sum()>0


def test_closed_circuit_training_source_is_physical_at_start():
    full=case()
    rom=GeomHeatROM.build(full,rank=10)
    temp=torch.zeros((1,rom.rank),requires_grad=True)
    y=torch.zeros((1,14),requires_grad=True)
    sy=np.ones(14)
    row={"voltage":np.array([60.,0.])}
    actual=_port_src(row,rom,temp,y,sy,60.)
    expect=heat_drive(60.,zmat(rom.z0),rom.ext,rom.port,
                      np.full(2,rom.amb),rom.cap[rom.cu])
    assert np.allclose(actual.detach().numpy()[0],expect,rtol=2e-6,atol=2e-7)
    (actual.square().sum()).backward()
    assert torch.isfinite(y.grad).all()
    assert torch.isfinite(temp.grad).all()
    assert y.grad.abs().sum()>0


def test_fractional_sources_use_left_endpoint_drive():
    zr=np.array([.15,.002,.02,.003,.14,.005])
    v=np.array([[75.,0.,0.,75.]])
    z=np.broadcast_to(zr,(1,4,6))
    src=build_sources(v,z,np.eye(2)*.5)
    assert np.all(src[0,1]>0)
    assert np.all(src[0,2]==0)
    assert np.all(src[0,3]==0)


def test_budget_correction_matches_physical_solver_and_gradients():
    from examples.bfzi_geom_neural import _heat_balance
    from sdfmpneo_bfzi.geom_neural import _balance
    from sdfmpneo_bfzi.geom_rom import groups
    full=case()
    rom=GeomHeatROM.build(full,rank=10)
    dt=15.
    volts=60.
    old=np.zeros(rom.rank)
    pred=np.zeros(rom.rank)
    cp=rom.cap[:,None]*rom.phi
    hp=rom.H@rom.phi
    B=np.stack([cp[sl].sum(axis=0) for sl in groups(rom)])
    D=np.stack([hp[sl].sum(axis=0) for sl in groups(rom)])
    phys,*_=_balance(rom,old,pred,dt,volts,
                     np.zeros((2,2),complex),B,D)
    row={"voltage":np.array([volts,0.])}
    a=torch.tensor(pred[None],dtype=torch.float32,requires_grad=True)
    y=torch.zeros((1,14),dtype=torch.float32,requires_grad=True)
    corrected=_heat_balance(row,rom,a,a,y,np.ones(14),dt,volts)
    assert np.allclose(corrected.detach().numpy()[0,:5],phys[:5],
                       atol=2e-5,rtol=2e-5)
    corrected.square().sum().backward()
    assert torch.isfinite(a.grad).all()
    assert torch.isfinite(y.grad).all()
