"""Frozen cold-only geometry test after terminal-model checkpoint selection.

No fitting, hyperparameter selection or new BFZI dynamic teacher is performed.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sdfmpneo_bfzi.frac_memory import zmat
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_portself import PortSelf


def main():
    base=GeomInit.load('runs/geomfrac_init_v1.npz')
    models={
        'baseline_v3':None,
        'dual_port_30':PortSelf.load('runs/geomfrac_portself_v5.npz'),
        'dual_port_54':PortSelf.load('runs/geomfrac_portself_v5_54.npz')}
    plan=json.loads(Path('runs/geomfrac_portself_v5_holdout_plan.json').read_text())
    names=plan['test']
    rows={k:[] for k in models}
    for name in names:
        case=plan['cases'][name]
        with np.load(f'runs/cold_port_holdout/cold_{name}_32.npz') as d:
            truth=np.asarray(d['y'][:6],float)
            ext=np.asarray(d['ext'],complex)
        cur=np.linalg.solve(zmat(truth)+ext,[1.,0.])
        for label,model in models.items():
            pred=base.predict(case)
            if model is not None:
                pred=model.apply(case,pred)
            z=np.asarray(pred['z0'],float)
            amp=np.linalg.solve(zmat(z)+pred['ext'],[1.,0.])
            rows[label].append(dict(case=name,
                Z_rel_L2=float(np.linalg.norm(z-truth)/np.linalg.norm(truth)),
                current_rel_L2=float(np.linalg.norm(amp-cur)/np.linalg.norm(cur)),
                Z_max_abs_ohm=float(np.max(abs(z-truth)))))
    result={'status':'frozen cold-only Q geometry independent test',
            'plan':'runs/geomfrac_portself_v5_holdout_plan.json',
            'train_geometries':{'baseline_v3':30,'dual_port_30':30,'dual_port_54':54},
            'geometries':names,'models':{}}
    for key,measure in rows.items():
        result['models'][key]={'results':measure,
           'mean_Z_rel_L2':float(np.mean([r['Z_rel_L2'] for r in measure])),
           'mean_current_rel_L2':float(np.mean([r['current_rel_L2'] for r in measure])),
           'worst_current_rel_L2':float(max(r['current_rel_L2'] for r in measure))}
        print('FROZEN_Q',key,'Z%',
             round(100*result['models'][key]['mean_Z_rel_L2'],3),
             'current%',round(100*result['models'][key]['mean_current_rel_L2'],3),
             'worstCurrent%',round(100*result['models'][key]['worst_current_rel_L2'],3),
             flush=True)
    Path('runs/geomfrac_portself_v5_holdout.json').write_text(
        json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
