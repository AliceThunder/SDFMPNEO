"""End-to-end mesh-free BFZI geometry-only 32-mode proxy inference."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit,mix_init
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--case',required=True)
    p.add_argument('--atlas',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    p.add_argument('--cold-atlas',type=Path,default=None)
    p.add_argument('--cold-weight',type=float,default=0.)
    p.add_argument('--model',type=Path,default=Path('runs/geomfrac_et_v4_8g_quality.pt'))
    p.add_argument('--z-fix-model',type=Path,
                   default=Path('runs/geomfrac_et_v9_ztune.pt'))
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_free_infer.npz'))
    p.add_argument('--voltage',type=float,default=60.)
    p.add_argument('--horizon',type=float,default=600.)
    p.add_argument('--step',type=float,default=15.)
    a=p.parse_args()
    torch.set_num_threads(2)
    start=time.perf_counter()
    case=json.loads(a.plan.read_text(encoding='utf-8'))['cases'][a.case]
    atlas=GeomInit.load(a.atlas)
    extra=GeomInit.load(a.cold_atlas) if a.cold_atlas else None
    init=mix_init(case,atlas,extra,a.cold_weight)
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    net.eval()
    tangent=load_z(a.z_fix_model)
    loaded=time.perf_counter()
    times=np.arange(0,a.horizon+a.step*.5,a.step)
    voltage=np.full(len(times),a.voltage)
    result=rollout(case,atlas,net,ck,times,voltage,tangent,
                   init_override=init)
    infer=time.perf_counter()-loaded
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**{k:v for k,v in result.items()
                               if isinstance(v,np.ndarray)})
    whole=time.perf_counter()-start
    print(json.dumps(dict(status='first anchor-free closed-loop prototype',
        case=a.case,out=str(a.out),num_steps=len(times),
        cold_atlas_weight=a.cold_weight,
        load_s=loaded-start,infer_s=infer,total_s=whole,
        cold_BFZI_solves=0,new_geometry_heat_basis_builds=0,
        online_full_FV_solves=0,
        budget_max_W=float(np.max(abs(result['material_budget_W']))),
        final_port_Z6=result['z6'][-1].tolist(),
        final_material_T=result['temp_group_K'][-1].tolist()),indent=2))


if __name__=='__main__':
    main()
