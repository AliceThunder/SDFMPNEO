"""Check native BFZI M4 terminal-lead clearance before cold reference training.

The original 12 mm X-lane offset can bring the two insulated +Z leads
closer than the BFZI insulated-wire diameter for some rounded-square coils.
Keep every raw NPZ; exclude physically overlapping routes from model splits.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from examples.bfzi_param import originals,create


def gap(bf,geom):
    d,_=create(bf,None,'M4',**geom,coils_only=True,
               physical_box_only=True,sea_void='oriented')
    cfg=d['cfg']
    diameter=float(cfg['BSR_INSULATED_WIRE_D_M'])
    res=[]
    for coil in d['coils']:
        lead,meta=bf.zlead_line_geom_for_coil(coil,cfg)
        cut=int(meta['panels_outer'])
        x=np.concatenate((lead.pos[:cut]-.5*lead.dl[:cut],
                          lead.pos[:cut],lead.pos[:cut]+.5*lead.dl[:cut]))
        y=np.concatenate((lead.pos[cut:]-.5*lead.dl[cut:],
                          lead.pos[cut:],lead.pos[cut:]+.5*lead.dl[cut:]))
        terminal=float(np.linalg.norm(
            np.asarray(meta['outer_terminal_world_m'])-
            np.asarray(meta['inner_terminal_world_m'])))
        dmin=min(terminal,float(cdist(x,y).min()))
        res.append(dict(gap_mm=1000*dmin,
                        terminal_mm=1000*terminal,
                        wire_diameter_mm=1000*diameter,
                        clear=bool(dmin+1e-12>=diameter)))
    return res


def scan(plan,bf):
    result={}
    for name in plan['train']+plan['val']+plan['test']:
        try:
            ports=gap(bf,plan['cases'][name]['geometry']
                      if 'geometry' in plan['cases'][name]
                      else plan['cases'][name])
            result[name]=dict(ports=ports,
                              clear=all(p['clear'] for p in ports))
        except Exception as exc:
            result[name]=dict(clear=False,
                              error=f'{type(exc).__name__}: {exc}')
    return result


def apply(plan,results):
    ids=set()
    by={}
    for split in ('train','val','test'):
        bad=[key for key in plan[split] if not results[key]['clear']]
        plan[split]=[key for key in plan[split] if key not in bad]
        by[split]=bad
        ids.update(bad)
    prev={r['name'] for r in plan.get('excluded_leads',[])}
    plan.setdefault('excluded_leads',[]).extend(
        dict(name=name,geometry=plan['cases'][name],result=results[name])
        for name in sorted(ids-prev))
    return by


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bfzi-root',type=Path,required=True)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_cold500_plan.json'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold500_lead_guard.json'))
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    bf,_=originals(a.bfzi_root)
    rows=scan(plan,bf)
    summary=dict(status='scanned',threshold='BFZI insulated wire diameter',
                 checked=len(rows),bad=sum(not x['clear'] for x in rows.values()),
                 bad_per_split={k:sum(not rows[n]['clear'] for n in plan[k])
                                for k in ('train','val','test')},
                 examples={k:v for k,v in rows.items() if not v['clear']})
    if a.apply:
        summary['excluded']=apply(plan,rows)
        summary['status']='applied'
        a.plan.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    a.out.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('LEAD_GUARD',json.dumps({k:v for k,v in summary.items()
        if k not in ('examples','excluded')}),flush=True)
    for key in ('C0459','C0171','C0013','C0510'):
        if key in rows:
            print('LEAD_CASE',key,rows[key],flush=True)


if __name__=='__main__':
    main()
