"""Diagnose cold and shared thermal surrogate errors, using physics only as labels."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from examples.bfzi_geom_init import records
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_rom import groups
from sdfmpneo_bfzi.geom_ztune import restore


def main():
    torch.set_num_threads(2)
    ck=torch.load('runs/geomfrac_et_v4_8g_quality.pt',map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=32,
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    tan=restore('runs/geomfrac_et_v9_ztune.pt')
    atlas=GeomInit.load('runs/geomfrac_init_v1.npz')
    pl=json.loads(Path('runs/geomfrac_s5_v4_plan.json').read_text(encoding='utf-8'))
    mf=json.loads(Path('runs/geomfrac_s5_v4_30g_manifest.json').read_text(encoding='utf-8'))
    results=[]
    for spec in mf['val']:
        name=Path(spec['anchor']).stem.split('_')[2]
        case=pl['cases'][name]
        rr=records(pl,dict(val=[spec]),'val')[0]
        guard=atlas.predict(case)
        true=dict(z0=rr['y'][:6],air=rr['y'][6:9],cap=rr['cap'],
                  k=rr['k'],src=rr['src'],ext=rr['ext'],
                  guard=guard['guard'],gmat=guard['gmat'])
        base=atlas.predict(case)
        for row in prepare(spec,32):
            d={}
            for key,init in (
                ('all_pred',base),
                ('cold_true',dict(base,z0=true['z0'],air=true['air'],cap=true['cap'],ext=true['ext'])),
                ('heat_true',dict(base,k=true['k'],src=true['src'])),
                ('all_true',true)):
                pred=rollout(case,atlas,net,ck,row['time'],row['voltage'],
                             tan,init_override=init)
                d[key]=metric(pred,row)
            results.append(dict(case=name,file=row['file'],by_model=d))
        print('GFREE_DIAG',name,[(k,round(np.mean([x['by_model'][k]['dynamic_z_rel_L2']
                            for x in results if x['case']==name])*100,2))
                            for k in d],flush=True)
    summary={}
    for key in ('all_pred','cold_true','heat_true','all_true'):
        summary[key]={metric+'_mean':float(np.mean([v['by_model'][key][metric]
                           for v in results]))
                           for metric in ('dynamic_z_rel_L2','material_mean_max_K',
                                          'copper_max_K','energy_budget_W')}
    Path('runs/geomfrac_free_dev_diag.json').write_text(
        json.dumps(dict(results=results,summary=summary),indent=2),encoding='utf-8')
    print('DIAGNOSTIC_MEANS',[(k,round(100*v['dynamic_z_rel_L2_mean'],2),
                             round(v['material_mean_max_K_mean'],3))
                             for k,v in summary.items()],flush=True)


if __name__=='__main__':
    main()
