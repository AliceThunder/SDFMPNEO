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

def test_rank48_keeps_pretrained_rank32_thermal_dynamics():
    import numpy as np
    from examples.bfzi_geom_neural import load_init
    torch.manual_seed(12)
    old=GeomFracET(ng=45,rank=32,heat_dt=1.875)
    with torch.no_grad():
        old.heat_head[-1].weight.normal_(0,.001)
    ck=dict(state=old.state_dict(),rank=32,ng=45,port_sea=True)
    new=GeomFracET(ng=45,rank=48,heat_dt=1.875)
    load_init(new,ck)
    h=old.init_state(1)
    u=torch.rand(1,3)
    geom=torch.rand(1,45)
    temp=torch.rand(1,32)
    _, y0, a0=old.step_et(h,u,15.,geom,temp,torch.ones(32))
    _, y1, a1=new.step_et(h,u,15.,geom,
                         torch.cat((temp,torch.zeros(1,16)),dim=1),torch.ones(48))
    assert torch.allclose(y1,y0,atol=1e-7)
    assert torch.allclose(a1[:,:32],a0,atol=1e-7)
    assert torch.count_nonzero(a1[:,32:])==0

def test_package_correction_still_enforces_five_budgets():
    import numpy as np
    from sdfmpneo_bfzi.geom_pkg import PkgFix
    from sdfmpneo_bfzi.geom_rom import GeomHeatROM
    from sdfmpneo_bfzi.geom_canon import basis
    from sdfmpneo_bfzi.geom_neural import rollout
    from test_bfzi_geom_canon import Head
    head=Head()
    rom=GeomHeatROM(head,basis(head))
    def circuit(theta,dsea,voltage):
        z=np.array([[.2+1j,.05+.2j],[.05+.2j,.2+1j]])+dsea
        cur=np.linalg.solve(z+rom.ext,np.array([voltage,0.],complex))
        return z,z,cur
    def source(theta,cur,sea):
        heat=np.zeros(len(rom.cap))
        heat[:rom.ns]=.002*abs(cur[0])**2
        heat[rom.cu]=.001*abs(cur)**2
        return heat,np.full(rom.ns,.002*abs(cur[0])**2),np.zeros((rom.ns,3),complex),float(heat[:rom.ns].sum()),heat[rom.cu]
    rom.circuit=circuit
    rom.source=source
    net=GeomFracET(ng=40,rank=32)
    pkg=PkgFix(ng=40)
    with torch.no_grad():
        pkg.net[-1].weight.fill_(.005)
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14),
            a_scale=np.ones(32),port_sea=True)
    result=rollout(rom,net,ck,[0.,50.,50.,0.],
                   [0.,15.,30.,45.],pkg=pkg)
    assert np.max(abs(result["material_budget_W"]))<1e-6
    assert np.isfinite(result["z6"]).all()

def test_feedback_lift_preserves_material_energy():
    import numpy as np
    from sdfmpneo_bfzi.geom_canon import basis
    from sdfmpneo_bfzi.geom_rom import GeomHeatROM,groups
    from sdfmpneo_bfzi.geom_lift import lift_basis
    from sdfmpneo_bfzi.geom_neural import rollout
    # The S5 sea-skin layout is a property of real anchors; using the
    # real Q03 basis exercises the physically relevant local shell.
    from sdfmpneo_bfzi.geom_heat import GeomHeat
    h=GeomHeat.load('runs/bfzi_s5_Q03_ultra_v5_anchor.npz')
    rom=GeomHeatROM(h,basis(h,32))
    w=np.zeros((14,16))
    w[0,0]=.01
    eff=lift_basis(rom,w)
    for sl in groups(rom):
        assert np.max(abs((rom.cap[sl])@ (eff[sl]-rom.phi[sl])))<1e-6
    net=GeomFracET(ng=45,rank=32)
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14),
            a_scale=np.ones(32),port_sea=True,canonical=True)
    base=rollout(rom,net,ck,[0.,50.,0.],[0.,15.,30.])
    new=rollout(rom,net,ck,[0.,50.,0.],[0.,15.,30.],lift=w)
    assert np.max(abs(new['material_budget_W']))<1e-6
    assert np.isfinite(new['temp_K']).all()

def test_v7_skin_zero_correction_preserves_rank32_feedback():
    import numpy as np
    from sdfmpneo_bfzi.geom_heat import GeomHeat
    from sdfmpneo_bfzi.geom_rom import GeomHeatROM
    from sdfmpneo_bfzi.geom_canon import basis
    from sdfmpneo_bfzi.geom_neural import rollout
    from sdfmpneo_bfzi.geom_skin import SkinFix
    h=GeomHeat.load('runs/bfzi_s5_Q03_ultra_v5_anchor.npz')
    rom=GeomHeatROM(h,basis(h,32))
    net=GeomFracET(ng=45,rank=32)
    fix=SkinFix(ng=45)
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14),
            a_scale=np.ones(32),port_sea=True,canonical=True)
    voltage=np.array([0.,45.,0.])
    times=np.array([0.,15.,30.])
    a=rollout(rom,net,ck,voltage,times)
    b=rollout(rom,net,ck,voltage,times,skin=fix)
    assert np.allclose(a['temp_K'],b['temp_K'],atol=1e-7)
    assert np.max(abs(b['material_budget_W']))<1e-7
    assert np.max(abs(b['skin_state']))<1e-10
    assert b['heat_implicit_solves_online']==0

def test_v7_residual_skin_zero_gain_preserves_v6_linear_feedback():
    import numpy as np
    from sdfmpneo_bfzi.geom_heat import GeomHeat
    from sdfmpneo_bfzi.geom_rom import GeomHeatROM
    from sdfmpneo_bfzi.geom_canon import basis
    from sdfmpneo_bfzi.geom_neural import rollout
    from sdfmpneo_bfzi.geom_skin import SkinFix
    h=GeomHeat.load('runs/bfzi_s5_Q03_ultra_v5_anchor.npz')
    rom=GeomHeatROM(h,basis(h,32))
    net=GeomFracET(ng=45,rank=32)
    fix=SkinFix(ng=45)
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(14),
            a_scale=np.ones(32),port_sea=True,canonical=True)
    w=np.zeros((14,16))
    w[1,1]=.03
    v=np.array([0.,45.,0.])
    t=np.array([0.,15.,30.])
    old=rollout(rom,net,ck,v,t,lift=w)
    new=rollout(rom,net,ck,v,t,lift=w,skin=fix,skin_gain=0.)
    assert np.allclose(old['temp_K'],new['temp_K'],atol=1e-7)
    assert np.max(abs(new['material_budget_W']))<1e-7

def test_v7_skin_output_preserves_material_means_and_original_temperatures():
    import numpy as np
    from sdfmpneo_bfzi.geom_heat import GeomHeat
    from sdfmpneo_bfzi.geom_canon import basis
    from sdfmpneo_bfzi.geom_rom import GeomHeatROM,groups
    from sdfmpneo_bfzi.geom_skin import SkinFix,reconstruct
    rom=GeomHeatROM(GeomHeat.load('runs/bfzi_s5_Q03_ultra_v5_anchor.npz'),
                    basis(GeomHeat.load('runs/bfzi_s5_Q03_ultra_v5_anchor.npz'),32))
    net=SkinFix(ng=45)
    n=3
    time=np.array([0.,15.,30.])
    voltage=np.array([0.,45.,45.])
    heat=np.random.default_rng(73).uniform(295,300,(n,len(rom.cap)))
    state=np.random.default_rng(37).normal(size=(n,32))
    out=reconstruct(rom,heat,state,time,voltage,net)
    assert np.allclose(out,heat,atol=1e-9)
    with torch.no_grad():
        net.net[-1].weight.fill_(.01)
    out=reconstruct(rom,heat,state,time,voltage,net)
    for sl in groups(rom):
        assert np.max(abs((out[:,sl]-heat[:,sl])@rom.cap[sl]))<1e-6
    assert np.array_equal(out[:,rom.ns:],heat[:,rom.ns:])
