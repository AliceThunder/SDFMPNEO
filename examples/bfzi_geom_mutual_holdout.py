"""Independent frozen R geometry test of cold mutual Z and sea source shape."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sdfmpneo_bfzi.frac_memory import zmat
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix


def main():
    atlas=GeomInit.load('runs/geomfrac_init_v1.npz')
    snet=PortSelf.load('runs/geomfrac_portself_v5.npz')
    mnet=MutFix.load('runs/geomfrac_mutual_v6.npz')
    plan=json.loads(Path('runs/geomfrac_mutual_v6_holdout_plan.json').read_text())
    cases=plan['test']
    out={key:[] for key in ('base','dual','mutual')}
    for name in cases:
        with np.load(f'runs/mutual_R_cold/cold_{name}_32.npz') as f:
            target=f['y'].copy()
            ext=f['ext'].copy()
            source=f['src'].copy()
        case=plan['cases'][name]
        pred=atlas.predict(case)
        preds={'base':pred,'dual':snet.apply(case,pred)}
        preds['mutual']=mnet.apply(case,preds['dual'])
        ct=np.linalg.solve(zmat(target[:6])+ext,[1,0])
        for key,guess in preds.items():
            z=guess['z0']
            cp=np.linalg.solve(zmat(z)+guess['ext'],[1,0])
            err=dict(case=name,
                z=float(np.linalg.norm(z-target[:6])/np.linalg.norm(target[:6])),
                current=float(np.linalg.norm(cp-ct)/np.linalg.norm(ct)),
                mutual=float(np.linalg.norm(z[[2,3]]-target[[2,3]])/
                             np.linalg.norm(target[[2,3]])),
                air12=float(abs(guess['air'][1]-target[7])*2*np.pi*200000),
                src=float(np.linalg.norm(guess['src']-source)/
                          max(np.linalg.norm(source),1e-15)))
            out[key].append(err)
    result=dict(status='Independent frozen six-R cold-only physical validation',
                plan='runs/geomfrac_mutual_v6_holdout_plan.json',
                geometries=cases,models={})
    for key,rows in out.items():
        stat={k+'_mean':float(np.mean([r[k] for r in rows]))
              for k in ('z','current','mutual','air12','src')}
        result['models'][key]=dict(rows=rows,summary=stat)
        print('MUTUAL_INDEPENDENT',key,
              'Z%',round(100*stat['z_mean'],3),
              'current%',round(100*stat['current_mean'],3),
              'mutual%',round(100*stat['mutual_mean'],3),
              'src%',round(100*stat['src_mean'],3),
              flush=True)
    Path('runs/geomfrac_mutual_v6_holdout.json').write_text(
       json.dumps(result,indent=2))


if __name__=='__main__':
    main()
