"""C-stage causal rollout, conservation, reload checks."""
import numpy as np
import torch
from sdfmpneo_bfzi.heat_rom import HeatROM
from sdfmpneo_bfzi.frac_memory import FracNet
from sdfmpneo_bfzi.frac_rom import FracROM

def case():
    n=6
    cap=np.arange(2.,8.)
    H=np.diag(np.full(n,2.))-np.eye(n,k=1)-np.eye(n,k=-1)
    rom=HeatROM(cap,H,293.15+np.eye(n),rank=n)
    fr=FracROM(np.array([2.,.2,.12,.01,1.8,.2]),
               np.ones((6,n))*1e-4,np.diag([.1j,.1j]),
               np.diag([4.,4.]),np.array([.12+.01j,.11+.01j]),
               np.array([.0002+.00001j,.0002+.00001j]),
               np.array([.2,.2,.3,.3,0.,0.]),
               np.array([4,5]),293.15)
    return rom,fr

def test_energy_causal_reload(tmp_path):
    rom,fr=case()
    t=np.arange(0.,51.,5.)
    v=np.array([20.,20.,10.,0.,0.,10.,20.,20.,20.,0.,0.])
    w=v.copy();w[6:]=0.
    a=fr.rollout(rom,t,v);b=fr.rollout(rom,t,w)
    np.testing.assert_allclose(a["z6"][:6],b["z6"][:6])
    np.testing.assert_allclose(a["temp_K"][:6],b["temp_K"][:6])
    assert np.max(a["total_budget_W"])<1e-8
    assert a["temp_K"][1].max()>293.15
    assert a["Psea_W"][3]==0.
    fr.save(tmp_path/"fr.npz")
    c=FracROM.load(tmp_path/"fr.npz").rollout(rom,t,v)
    np.testing.assert_allclose(c["temp_K"],a["temp_K"])

def test_neural_fractional_exponential_causality():
    rom,fr=case()
    t=np.arange(0.,31.,5.)
    a=np.array([20.,20.,0.,20.,20.,0.,0.])
    b=a.copy();b[-2:]=20.
    for mode in ("frac","exp"):
        torch.manual_seed(3)
        net=FracNet(nout=6,mode=mode,hidden=12)
        ck=dict(vref=20.,scale=np.full(6,1e-5))
        x=fr.rollout(rom,t,a,net,ck)
        y=fr.rollout(rom,t,b,net,ck)
        np.testing.assert_allclose(x["temp_K"][:5],y["temp_K"][:5])
        assert np.max(x["total_budget_W"])<1e-8

def test_nonlinear_copper_curve_and_conservation(tmp_path):
    rom,fr=case()
    fr.cu_grid=np.array([0.,50.,100.])
    fr.cu_table=np.stack((fr.cu0,fr.cu0+fr.dcu*40.,
                          fr.cu0+fr.dcu*70.))
    s=rom.encode(fr.amb+np.array([0.,0.,0.,0.,50.,40.]))
    z,sea,cu=fr.impedance(rom,s)
    np.testing.assert_allclose(z-sea,fr.air+np.diag(cu),atol=1e-12)
    assert abs(cu[0]-(fr.cu0[0]+50*fr.dcu[0]))>1e-7
    fr.save(tmp_path/"curve.npz")
    z2=FracROM.load(tmp_path/"curve.npz").impedance(rom,s)[0]
    np.testing.assert_allclose(z,z2)
