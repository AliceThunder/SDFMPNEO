"""Train geometry-only BFZI cold impedance + shared thermal atlas."""
from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.geom_init import GeomInit,encode
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_canon import make_rom,labels
from sdfmpneo_bfzi.geom_rom import groups


def pack(head,case,name,rank=32):
    rom=make_rom(head,rank)
    y=np.r_[head.z0,head.geom[11:14],
            np.log([head.cap[span].sum() for span in groups(head)])]
    src=np.zeros((rank,3))
    for j,(a,b) in enumerate(((0,0),(0,1),(1,1))):
        force=np.zeros(len(head.cap))
        force[:head.ns]=head.weak[:,a,b]*head.vol
        src[:,j]=rom.phi.T@force
    lab=labels(rank)
    phi=rom.phi*np.sqrt(np.exp(y[9:])[lab])[None,:]
    sl=groups(head)
    guard=[]
    for j in range(5,rank):
        a,b=sl[lab[j]].start,sl[lab[j]].stop
        q=phi[a:b,j]
        guard.extend([phi[a+np.argmax(q),5:],
                      phi[a+np.argmin(q),5:]])
    return dict(name=name,x=encode(case),y=y,
                k=rom.K,src=src,ext=head.ext,case=case,
                guard=np.asarray(guard),cap=np.exp(y[9:]))


def records(plan,manifest,split,rank=32):
    out=[]
    for spec in manifest[split]:
        ident=Path(spec['anchor']).stem.split('_')[2]
        case=plan['cases'][ident]
        head=GeomHeat.load(spec['anchor'])
        out.append(pack(head,case,ident,rank))
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
    p.add_argument('--cold-plan',type=Path,default=None,
                   help="optional geometry-disjoint cold-only extra training plan")
    p.add_argument('--cold-dir',type=Path,default=None,
                   help="directory of compact cold-only anchors")
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_init_v1_dev.json'))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding='utf-8'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    tr=records(plan,mf,'train',a.rank)
    dev=records(plan,mf,'val',a.rank)
    if a.cold_plan and a.cold_dir:
        extra=json.loads(a.cold_plan.read_text(encoding='utf-8'))
        for name in extra['train']:
            f=a.cold_dir/f'cold_{name}_{a.rank}.npz'
            with np.load(f) as data:
                item={key:data[key] for key in ('x','y','k','src','ext','guard')}
            item['name']=name
            item['case']=extra['cases'][name]
            item['cap']=np.exp(item['y'][9:])
            tr.append(item)
        print('COLD_ONLY_EXTRA',len(extra['train']),flush=True)
    x=np.stack([r['x'] for r in tr])
    y=np.stack([r['y'] for r in tr])
    k=np.stack([r['k'] for r in tr])
    src=np.stack([r['src'] for r in tr])
    ext=np.stack([r['ext'] for r in tr])
    guard=np.stack([r['guard'] for r in tr])
    best=None
    for ell in (.6,1.,1.7,2.8,4.,6.):
        for lam in (.003,.03,.2):
            model=GeomInit.fit(x,y,k,src,ext,ell=ell,lam=lam)
            d=report(model,dev)
            score=(d['cold_Z_rel_L2_mean']+
                   .25*d['air_rel_L2_mean']+
                   .02*d['source_rel_F_mean']+
                   .002*d['cap_rel_L2_mean'])
            if best is None or score<best[0]:
                best=(score,ell,lam,d)
    _,ell,lam,_=best
    model=GeomInit.fit(x,y,k,src,ext,ell=ell,lam=lam)
    raw=report(model,dev)
    from sdfmpneo_bfzi.frac_memory import zmat
    def current_err(candidate):
        errs=[]
        for row in dev:
            guess=candidate.predict(row['case'])
            target=np.linalg.solve(zmat(row['y'][:6])+row['ext'],[1.,0.])
            got=np.linalg.solve(zmat(guess['z0'])+guess['ext'],[1.,0.])
            errs.append(np.linalg.norm(got-target)/max(np.linalg.norm(target),1e-12))
        return float(np.mean(errs))
    baseline_current=current_err(model)
    best_self=(baseline_current,None,None,None)
    for dim in (2,3):
        for sl in (.75,1.,1.5,2.,3.,5.):
            for sr in (.01,.05,.1,.3):
                candidate=GeomInit.fit(x,y,k,src,ext,ell=ell,lam=lam)
                candidate.fit_self(tr,sl,sr,dims=dim)
                cur=current_err(candidate)
                if cur<best_self[0]:
                    best_self=(cur,dim,sl,sr)
    if best_self[1] is not None:
        model.fit_self(tr,best_self[2],best_self[3],dims=best_self[1])
    model.set_guard(guard)
    best_k=(float('inf'),None,None)
    for kell in (.6,1.,1.7,2.8,4.,6.):
        model.set_kell(kell)
        for exp in itertools.product((0.,.25,.5,1.),repeat=3):
            model.set_kscale(exp)
            err=np.mean([np.linalg.norm(model.predict(r['case'])['k']-r['k'])/
                         np.linalg.norm(r['k']) for r in dev])
            if err<best_k[0]:
                best_k=(err,kell,exp)
    model.set_kell(best_k[1]).set_kscale(best_k[2])
    devres=report(model,dev)
    trainres=report(model,tr)
    data=dict(status='first geometry-only cold and shared-operator emulator',
              train_geometries=len(tr),dev_geometries=len(dev),
              rank=a.rank,
              ell=ell,lam=lam,thermal_kernel_ell=best_k[1],
              thermal_scale_exp=list(best_k[2]),
              self_dims=best_self[1],self_ell=best_self[2],
              self_lambda=best_self[3],
              baseline_dev_current_rel_L2=baseline_current,
              selected_dev_current_rel_L2=best_self[0],
              baseline_dev_Z_rel_L2=raw['cold_Z_rel_L2_mean'],
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
          'ell',ell,'lambda',lam,'kell',best_k[1],flush=True)


if __name__=='__main__':
    main()
