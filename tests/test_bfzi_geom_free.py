"""Anchor-free electothermal rollout, five-material energy budgets."""
import json
import numpy as np
import torch
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_ztune import restore

def test_free_closed_loop():
    case=json.load(open('runs/geomfrac_s5_v9_plan.json'))['cases']['W01']
    atlas=GeomInit.load('runs/geomfrac_init_v1.npz')
    ck=torch.load('runs/geomfrac_et_v4_8g_quality.pt',
                  map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=32,
        soe_quad=ck.get('soe_quad','trap'),
        heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    tan=restore('runs/geomfrac_et_v9_ztune.pt')
    out=rollout(case,atlas,net,ck,np.array([0.,15.,30.,45.,60.]),
                np.full(5,40.),tan)
    assert out['cold_BFZI_solves']==0
    assert out['heat_basis_builds']==0
    assert out['heat_FV_solves']==0
    assert out['temp_group_K'].shape==(5,5)
    assert out['heat_state'].shape==(5,32)
    assert np.isfinite(out['z6']).all()
    assert np.allclose(out['z6'][0],atlas.predict(case)['z0'],atol=1e-10)
    assert np.max(abs(out['material_budget_W']))<1e-7

def test_cold_source_changes_physical_shape_without_breaking_budget():
    import numpy as np
    import torch
    from sdfmpneo_bfzi.geom_init import GeomInit
    from sdfmpneo_bfzi.geom_neural import GeomFracET
    from sdfmpneo_bfzi.geom_srcfix import SrcFix
    from sdfmpneo_bfzi.geom_free import rollout
    from sdfmpneo_bfzi.geom_ztune import restore

    torch.set_num_threads(2)
    case=json.load(open('runs/geomfrac_s5_v9_plan.json'))['cases']['W01']
    atlas=GeomInit.load('runs/geomfrac_init_v1.npz')
    ck=torch.load('runs/geomfrac_et_v4_8g_quality.pt',
                  map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=32,
        soe_quad=ck.get('soe_quad','trap'),
        heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    net.eval()
    tan=restore('runs/geomfrac_et_v9_ztune.pt')
    init=atlas.predict(case)
    new=SrcFix.load('runs/geomfrac_src_v6.npz').apply(case,init)
    t=np.array([0.,15.,30.,45.,60.])
    v=np.full(len(t),75.)
    old=rollout(case,atlas,net,ck,t,v,tan,init_override=init,
                shape_mix=.5)
    pred=rollout(case,atlas,net,ck,t,v,tan,init_override=new,
                 shape_mix=.5)
    assert np.max(abs(pred['heat_state'][:,5:]-old['heat_state'][:,5:]))>1e-8
    assert np.max(abs(pred['material_budget_W']))<1e-8
    assert np.isfinite(pred['temp_group_K']).all()


def test_shared_implicit_heat_rom_budget_and_determinism():
    from sdfmpneo_bfzi.geom_free import rollout
    from sdfmpneo_bfzi.geom_init import GeomInit
    from sdfmpneo_bfzi.geom_ztune import restore

    torch.set_num_threads(2)
    case=json.load(open('runs/geomfrac_s5_v9_plan.json'))['cases']['W01']
    atlas=GeomInit.load('runs/geomfrac_init_v1.npz')
    ck=torch.load('runs/geomfrac_et_v4_8g_quality.pt',
                  map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=32,
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    tan=restore('runs/geomfrac_et_v9_ztune.pt')
    t=np.array([0.,15.,30.,45.,60.])
    v=np.array([65.,65.,0.,65.,65.])
    a=rollout(case,atlas,net,ck,t,v,tan,heat_mode='rom')
    b=rollout(case,atlas,net,ck,t,v,tan,heat_mode='rom')
    assert a['heat_mode']=='rom'
    assert a['cold_BFZI_solves']==0
    assert a['heat_basis_builds']==0
    assert a['heat_FV_solves']==0
    assert a['heat_state'].shape==(5,32)
    assert np.max(abs(a['material_budget_W']))<1e-8
    assert np.allclose(a['z6'],b['z6'],rtol=0,atol=1e-12)
    assert np.isfinite(a['temp_group_K']).all()


def test_rank48_free_rollout_uses_all_shared_heat_coordinates():
    from sdfmpneo_bfzi.geom_init import GeomInit,encode
    from sdfmpneo_bfzi.geom_neural import GeomFracET
    from sdfmpneo_bfzi.geom_free import rollout

    case=dict(donor='g1',alpha_per_K=.020,
        geometry=dict(alpha1=.8,alpha2=.95,core1_mm=19,core2_mm=15,
            shift_mm=[2,94,0],tilt_deg=10,yaw_deg=2,pitch_deg=3,
            box1_mm=221,box2_mm=230),
        thermal=dict(ins_k=.2,ins_t=.0004,pkg_k=.25,pkg_sea_h=20.))
    x=np.stack([encode(case)]*3)
    y=np.zeros((3,14))
    y[:,0]=.2
    y[:,4]=.2
    y[:,6:9]=[1e-5,1e-6,1e-5]
    y[:,9:14]=np.log([1e5,500,500,20,20])
    k=np.stack([np.eye(48)*.01]*3)
    src=np.ones((3,48,3))
    ext=np.stack([np.eye(2,dtype=complex)]*3)
    atlas=GeomInit.fit(x,y,k,src,ext,ell=1.,lam=.1)
    net=GeomFracET(ng=45,rank=48,hidden=16,nrate=4,
                   soe_quad='trap',heat_dt=1.875)
    net.eval()
    ck=dict(src_scale=np.ones(3),y_scale=np.ones(6),
            a_scale=np.ones(48),canonical=True)
    t=np.array([0.,15.,30.])
    v=np.array([40.,40.,0.])
    pred=rollout(case,atlas,net,ck,t,v,shape_mix=.5)
    rom=rollout(case,atlas,net,ck,t,v,heat_mode='rom')
    assert pred['heat_rank']==48
    assert pred['heat_state'].shape==(3,48)
    assert rom['heat_state'].shape==(3,48)
    assert np.isfinite(pred['z6']).all()
    assert np.isfinite(rom['z6']).all()
    assert np.max(abs(pred['material_budget_W']))<1e-6
    assert np.max(abs(rom['material_budget_W']))<1e-6
