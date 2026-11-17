"""Select cold-circuit blending on original developer histories only."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit,mix_init
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_ztune import restore


def main():
    torch.set_num_threads(2)
    a1=GeomInit.load('runs/geomfrac_init_v1.npz')
    a2=GeomInit.load('runs/geomfrac_init_v2.npz')
    ck=torch.load('runs/geomfrac_et_v4_8g_quality.pt',
                  map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=32,
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    tan=restore('runs/geomfrac_et_v9_ztune.pt')
    plan=json.loads(Path('runs/geomfrac_s5_v4_plan.json').read_text())
    mf=json.loads(Path('runs/geomfrac_s5_v4_30g_manifest.json').read_text())
    data=[]
    for spec in mf['val']:
        name=Path(spec['anchor']).stem.split('_')[2]
        case=plan['cases'][name]
        data.extend((case,r) for r in prepare(spec,32))
    results=[]
    for lam in (0.,.2,.4,.6,.8,1.):
        ms=[]
        for case,row in data:
            init=mix_init(case,a1,a2,lam)
            pred=rollout(case,a1,net,ck,row['time'],row['voltage'],
                         tan,init_override=init)
            ms.append(metric(pred,row))
        res=dict(weight=lam,
                 cold=float(np.mean([r['cold_z_rel_L2'] for r in ms])),
                 dyn=float(np.mean([r['dynamic_z_rel_L2'] for r in ms])),
                 temp_K=float(np.mean([r['material_mean_max_K'] for r in ms])))
        results.append(res)
        print('FREE_BLEND_DEV',lam,round(100*res['dyn'],3),
              round(100*res['cold'],3),round(res['temp_K'],3),flush=True)
    best=min(results,key=lambda x:x['dyn'])
    out=dict(status='dev-selected conservative combination of v1/v2 cold port',
             train_split='original 30 and +24 cold-only',
             dev_histories=len(data),candidates=results,best=best)
    Path('runs/geomfrac_free_blend_dev.json').write_text(
        json.dumps(out,indent=2))
    print('FREE_BLEND_BEST',best,flush=True)


if __name__=='__main__':
    main()
