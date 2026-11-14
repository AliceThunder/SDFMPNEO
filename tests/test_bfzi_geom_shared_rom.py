"""Shared BFZI V150 32-ROM geometry transfer and target-grid mapping."""
import numpy as np
from sdfmpneo_bfzi.geom_shared_rom import SharedROM
from sdfmpneo_bfzi.frac_rom import FracROM
from sdfmpneo_bfzi.heat_rom import HeatROM

def test_shared_cold_causal_field():
    n=2+64+2+1024
    rom=HeatROM.__new__(HeatROM)
    rom.cap=np.ones(n)
    rom.basis=np.zeros((n,2))
    rom.basis[:,:]=0
    rom.basis[:,0]=1/np.sqrt(n)
    rom.basis[66,1]=1.
    rom.m=np.eye(2)
    rom.k=np.diag([.1,1.])
    rom.amb=293.15
    rom.rank=2
    w=np.zeros(n);w[:2]=.5
    fr=FracROM([1.2,.2,.05,.01,1.1,.2],np.zeros((6,2)),
               np.diag([.1j,.1j]),np.diag([4.,4.]),
               [.12+.01j,.11+.01j],[.0002,.0002],
               w,np.array([66,67]),293.15)
    corr=np.array([.005,0.,.002,0.,.004,0.])
    donor=SharedROM(rom,fr,corr,np.array([[0.,0.,0.],[1.,0.,0.]]))
    cold=np.array([1.,.2,.04,.01,1.2,.2])
    new=donor.target(cold)
    np.testing.assert_allclose(new.z0,cold+corr)
    np.testing.assert_allclose(fr.z0,[1.2,.2,.05,.01,1.1,.2])
    v=np.array([10.,10.,0.,0.,10.])
    t=np.arange(len(v))*5.
    a=donor.rollout(cold,t,v,sea_xyz=np.array([[1.,0.,0.],[0.,0.,0.],[.5,0.,0.]]))
    assert a["temp_K"].shape==(5,n+1)
    assert np.max(abs(a["total_budget_W"]))<1e-8
    assert a["Psea_W"][2]==0.
    b=v.copy();b[3:]=0.
    c=donor.rollout(cold,t,b)
    np.testing.assert_allclose(a["z6"][:3],c["z6"][:3])
    np.testing.assert_allclose(a["temp_K"][:3,3:],
                               donor.project(c["temp_K"][:3],
                                             np.array([[1.,0.,0.],[0.,0.,0.],[.5,0.,0.]]))[:,3:])

    from types import SimpleNamespace
    from sdfmpneo_bfzi.end_to_end import complete, metrics
    weak=np.zeros((3,2,2))
    weak[:,0,0]=.03
    weak[:,1,1]=.03
    cold_field=SimpleNamespace(weak_q=weak,
        unit_j=np.ones((3,2,3),complex)*.01)
    coils=[SimpleNamespace(coil=SimpleNamespace(
        center=(0.,s,0.),yaw_deg=0.,pitch_deg=0.,
        roll_deg=90.,plane_angle_deg=90.)) for s in (-.02,.02)]
    em=SimpleNamespace(vol=np.ones(3),points=np.array([
        [1.,0.,0.],[0.,0.,0.],[.5,0.,0.]]),
        sigma0=4.,alpha=.02,ambient=293.15,
        base=dict(sample={},cfg={},families=coils))
    field=complete(em,cold_field,new,a,v)
    np.testing.assert_allclose(field["qsea_W_m3"]@em.vol,
                               a["Psea_W"],atol=1e-10)
    assert metrics(field)["max_rel_port_power"]<1e-10
    assert metrics(field)["max_rel_q_power"]<1e-10
    altered=donor.target(cold,alpha_scale=.8)
    assert np.linalg.norm(altered.jac-fr.jac)>0
