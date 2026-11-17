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
