"""Summarize six-case cold surrogate dynamics and BFZI physical closure."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=Path('runs'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_cold_dyn6_acceptance.json'))
    a=p.parse_args()
    mf=json.loads((a.runs/'geomfrac_s5_cold_dyn6_manifest.json').read_text())
    br=json.loads((a.runs/'geomfrac_s5_cold_dyn6_build_report.json').read_text())
    ev=json.loads((a.runs/'geomfrac_cold_dyn6_eval.json').read_text())
    cases=ev['selected_geometry']
    expected=set(cases)
    entries={Path(x['anchor'].replace('\\','/')).stem.split('_')[2]:x
             for x in mf['test']}
    missing=sorted(expected-set(entries))
    physics={}
    for name in cases:
        item=br['cases'].get(name,{})
        traces=item.get('traces',[])
        physics[name]=dict(
            status=item.get('status'),
            histories=len(traces),
            physics_status=[x.get('physics',{}).get('status') for x in traces],
            duration_s=[x.get('physics',{}).get('completed_s') for x in traces],
            max_thermal_residual_W=max(
                [x.get('physics',{}).get('peak_thermal_residual_W',0)
                 for x in traces] or [0]),
            max_global_budget_W=max(
                [x.get('physics',{}).get('peak_global_budget_W',0)
                 for x in traces] or [0]),
            max_port_power_residual_W=max(
                [x.get('physics',{}).get('max_port_power_residual_W',0)
                 for x in traces] or [0]))
    valid=not missing and all(
        physics[k]['status']=='complete' and
        physics[k]['histories']==2 and
        physics[k]['physics_status']==['candidate','candidate'] and
        sorted(physics[k]['duration_s'])==[60.,300.]
        for k in cases)
    scores={k:float(ev['results'][k]['mean_dynamic_Z_pct'])
            for k in ('v5','coldnet','coldres')}
    per={}
    for name in cases:
        per[name]={k:float(ev['results'][k]['per_geometry'][name]
                           ['mean_dynamic_Z_pct'])
                   for k in ('v5','coldnet','coldres')}
    wins={k:sum(per[name][k]<per[name]['v5'] for name in cases)
          for k in ('coldnet','coldres')}
    improvement={k:100*(scores['v5']-scores[k])/scores['v5']
                 for k in ('coldnet','coldres')}
    out=dict(status='complete' if valid else 'incomplete',
             geometry_count=len(cases),
             history_count=sum(physics[k]['histories'] for k in cases),
             cases=cases,missing=missing,physical=physics,
             dynamic_Z_pct=scores,improvement_vs_v5_pct=improvement,
             geometries_better_than_v5=wins,per_geometry=per,
             decision='review dynamic gains and thermal transfer; do not auto-promote',
             sources=['geomfrac_s5_cold_dyn6_manifest.json',
                      'geomfrac_s5_cold_dyn6_build_report.json',
                      'geomfrac_cold_dyn6_eval.json'])
    a.out.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    print('DYN6_ACCEPT',json.dumps({
        'status':out['status'],'histories':out['history_count'],
        'dynamic_Z_pct':scores,'improvement_pct':improvement,
        'wins':wins,'max_thermal_residual_W':max(
          x['max_thermal_residual_W'] for x in physics.values())}),flush=True)


if __name__=='__main__':
    main()
