"""Heat-conditioned seawater impedance preserves the V4 checkpoint."""
import torch
from examples.bfzi_geom_neural import load_init
from sdfmpneo_bfzi.geom_neural import GeomFracET


def test_heat_z_warm_start_and_grad():
    torch.manual_seed(7)
    old=GeomFracET(ng=22,rank=10,heat_dt=1.875)
    net=GeomFracET(ng=22,rank=10,heat_dt=1.875,heat_z=True)
    load_init(net,dict(state=old.state_dict(),ng=22,port_sea=True))
    h=old.init_state(2)
    u=torch.tensor([[.2,.3,.4],[.5,.2,.1]])
    g=torch.randn(2,22)
    a=torch.rand(2,10)
    sc=torch.ones(10)
    ho,yo,ao=old.step_et(h,u,15.,g,a,sc)
    hn,yn,an=net.step_et(h,u,15.,g,a,sc)
    assert torch.allclose(hn,ho,atol=0,rtol=0)
    assert torch.allclose(yn,yo,atol=0,rtol=0)
    assert torch.allclose(an,ao,atol=0,rtol=0)
    yn[:,:6].sum().backward()
    assert net.heat_z_head[-1].weight.grad.abs().sum()>0

    with torch.no_grad():
        net.heat_z_head[-1].weight.fill_(.02)
    _,hot,_=net.step_et(h,u,15.,g,a,sc)
    _,cold,_=net.step_et(h,u,15.,g,torch.zeros_like(a),sc)
    assert not torch.allclose(hot[:,:6],yo[:,:6])
    _,old_cold,_=old.step_et(h,u,15.,g,torch.zeros_like(a),sc)
    assert torch.allclose(cold[:,:6],old_cold[:,:6],atol=0,rtol=0)

def test_coupled_tangent_preserves_material_budgets():
    import numpy as np
    from sdfmpneo_bfzi.geom_neural import rollout
    from sdfmpneo_bfzi.geom_rom import GeomHeatROM
    from test_bfzi_geom_heat import case
    full=case()
    rom=GeomHeatROM.build(full,rank=10)
    net=GeomFracET(ng=22,rank=rom.rank)
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14),
            a_scale=np.ones(rom.rank),port_sea=True)
    def tangent(g,a):
        return torch.cat((.0001*a[:,:1],torch.zeros(len(a),5)),dim=1)
    pred=rollout(rom,net,ck,[0.,50.,50.,0.],[0.,15.,30.,45.],
                 tangent=tangent)
    assert np.isfinite(pred['z6']).all()
    assert np.max(abs(pred['material_budget_W']))<1e-6
