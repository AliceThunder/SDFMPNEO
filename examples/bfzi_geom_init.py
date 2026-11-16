"""Train geometry-only BFZI cold impedance + shared thermal atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.geom_init import GeomInit,encode
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_canon import make_rom
from sdfmpneo_bfzi.geom_rom import groups


def records(plan,manifest,split,rank=32):
    out=[]
    for spec in manifest[split]:
        ident=Path(spec['anchor']).stem.split('_')[2]
        case=plan['cases'][ident]
        head=GeomHeat.load(spec['anchor'])
        rom=make_rom(head,rank)
        y=np.r_[head.z0,head.geom[11:14],
                np.log([head.cap[span].sum() for span in groups(head)])]
        src=np.zeros((rank,3))
        for j,(a,b) in enumerate(((0,0),(0,1),(1,1))):
            force=np.zeros(len(head.cap))
            force[:head.ns]=head.weak[:,a,b]*head.vol
            src[:,j]=rom.phi.T@force
        out.append(dict(name=ident,x=encode(case),y=y,
                        k=rom.K,src=src,ext=head.ext,case=case,
                        cap=np.exp(y[9:])))
    return out


def report(model,rows):
    out=[]
    for r in rows:
        y=model.predict(r['case'])
        out.append(dict(case=r['name'],
            cold_Z_rel_L2=float(np.linalg.norm(y['z0']-r['y'][:6])/
                                     max(np.linalg.norm(r['y'][:6]),1e-15)),
            cold_Z_max_abs_ohm=float(np.max(abs(y['z0']-r['y'][:6]))),
            air_rel_L2=float(np.linalg.norm(y['air']-r['y'][6:9])/
                                  max(np.linalg.norm(r['y'][6:9]),1e-15)),
            cap_rel_L2=float(np.linalg.norm(y['cap']-r['cap'])/
                                  max(np.linalg.norm(r['cap']),1e-15)),
            heat_K_rel_F=float(np.linalg.norm(y['k']-r['k'])/
                                    max(np.linalg.norm(r['k']),1e-15)),
            heat_K_min_eig=float(np.linalg.eigvalsh(y['k'])[0]),
            source_rel_F=float(np.linalg.norm(y['src']-r['src'])/
                                    max(np.linalg.norm(r['src']),1e-15))))
    return dict(rows=out,**{k+'_mean':float(np.mean([q[k] for q in out]))
                       for k in out[0] if k!='case'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('runs/geomfrac_s5_v4_plan.json'))
    p.add_argument('--manifest',type=Path,
                   default=Path('runs/geomfrac_s5_v4_30g_manifest.json'))
    p.add_argument('--rank',type=int,choices=(32,48,64),default=32)
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_init_v1_dev.json'))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    tr=records(plan,mf,'train',a.rank)
    dev=records(plan,mf,'val',a.rank)
    x=np.stack([r['x'] for r in tr])
    y=np.stack([r['y'] for r in tr])
    k=np.stack([r['k'] for r in tr])
    src=np.stack([r['src'] for r in tr])
    ext=np.stack([r['ext'] for r in tr])
    best=None
    for ell in (.6,1.,1.7,2.8,4.,6.):
        for lam in (.003,.03,.2):
            model=GeomInit.fit(x,y,k,src,ext,ell=ell,lam=lam)
            d=report(model,dev)
            score=(d['cold_Z_rel_L2_mean']+
                   .02*d['heat_K_rel_F_mean']+
                   .002*d['cap_rel_L2_mean'])
            if best is None or score<best[0]:
                best=(score,ell,lam,d)
    _,ell,lam,_=best
    model=GeomInit.fit(x,y,k,src,ext,ell=ell,lam=lam)
    raw=report(model,dev)
    best_self=(raw['cold_Z_rel_L2_mean'],None,None)
    for sl in (1.2,1.8,2.5,3.5,5.):
        for sr in (.001,.01,.05,.1):
            candidate=GeomInit.fit(x,y,k,src,ext,ell=ell,lam=lam)
            candidate.fit_self(tr,sl,sr)
            devs=report(candidate,dev)
            if devs['cold_Z_rel_L2_mean']<best_self[0]:
                best_self=(devs['cold_Z_rel_L2_mean'],sl,sr)
    if best_self[1] is not None:
        model.fit_self(tr,best_self[1],best_self[2])
    devres=report(model,dev)
    trainres=report(model,tr)
    data=dict(status='first geometry-only cold and shared-operator emulator',
              train_geometries=len(tr),dev_geometries=len(dev),
              rank=a.rank,
              ell=ell,lam=lam,self_ell=best_self[1],
              self_lambda=best_self[2],baseline_dev_Z_rel_L2=raw['cold_Z_rel_L2_mean'],
              train=trainres,dev=devres,
              requires_BFZI_at_query=False,
              requires_FV_operator_at_query=False,
              outputs=['cold Z0','air-inductance tuple',
                       'five material heat capacities',
                       '32D thermal K','32x3 cold weak-source coefficients',
                       'external circuit'])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    model.save(a.out)
    a.report.write_text(json.dumps(data,indent=2),encoding='utf-8')
    for metric in ('cold_Z_rel_L2','cold_Z_max_abs_ohm',
                   'heat_K_rel_F','cap_rel_L2','source_rel_F'):
        print('INIT_DEV',metric,round(devres[metric+'_mean'],8),flush=True)
    print('INIT_ATLAS',a.out,'train',len(tr),'dev',len(dev),
          'ell',ell,'lambda',lam,flush=True)


if __name__=='__main__':
    main()
