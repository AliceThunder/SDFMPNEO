"""Re-fit cold TX/RX self-port models on 42 geometries; hold out dev/BB.

The cold model is geometry-only at inference. Selection uses the frozen
five-geometry dev split; BB references are evaluated after selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.frac_memory import zmat
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def load_rows(plan,mf,part):
    rows=[]
    for spec in mf[part]:
        name=Path(spec['anchor'].replace('\\','/')).stem.split('_')[2]
        head=GeomHeat.load(spec['anchor'])
        rows.append(dict(name=name,case=plan['cases'][name],
                         y=np.r_[head.z0,head.geom[11:14]],
                         ext=head.ext))
    return rows


def cold_score(rows,atlas,ps,mu):
    zerrs=[];curr=[];each=[]
    for r in rows:
        case=r['case']
        ini=atlas.predict(case)
        ini=ps.apply(case,ini)
        ini=mu.apply(case,ini)
        ref=zmat(r['y'][:6])
        pred=zmat(ini['z0'])
        true=np.linalg.solve(ref+r['ext'],np.array([1.,0.],complex))
        got=np.linalg.solve(pred+ini['ext'],np.array([1.,0.],complex))
        zerr=float(np.linalg.norm(pred-ref)/max(np.linalg.norm(ref),1e-12))
        ierr=float(np.linalg.norm(got-true)/max(np.linalg.norm(true),1e-12))
        zerrs.append(zerr)
        curr.append(ierr)
        each.append(dict(name=r['name'],cold_Z_pct=100*zerr,
                         cold_current_pct=100*ierr,
                         X11_ohm=float(pred[0,0].imag),
                         X11_ref_ohm=float(ref[0,0].imag),
                         X22_ohm=float(pred[1,1].imag),
                         X22_ref_ohm=float(ref[1,1].imag)))
    return dict(cold_Z_pct=100*float(np.mean(zerrs)),
                cold_current_pct=100*float(np.mean(curr)),
                per_geometry=each)


def dyn_score(rows,plan,atlas,ps,mu,src,tan,net,ck,mix):
    """Original deployed autonomous closed loop; no BFZI operator at query."""
    from examples.bfzi_geom_multi import free_metric,prepare
    data=[r for spec in rows for r in prepare(spec,32)]
    free=(plan,atlas,ps,mu,src,tan,mix)
    eck=dict(src_scale=ck['src_scale'],y_scale=ck['y_scale'],
             a_scale=ck['a_scale'],canonical=True,port_sea=True)
    total=100*free_metric(data,free,net,eck)
    by={}
    for r in data:
        name=Path(r['file'].split('#')[0].replace('\\','/')).stem.split('_')[2]
        by.setdefault(name,[]).append(r)
    each={key:100*free_metric(group,free,net,eck)
          for key,group in by.items()}
    return dict(dynamic_Z_pct=float(total),per_geometry_pct=each)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,
                   default=Path('runs/geomfrac_pose42_plan.json'))
    p.add_argument('--manifest',type=Path,
                   default=Path('runs/geomfrac_s5_v6_42g_manifest.json'))
    p.add_argument('--bb-plan',type=Path,
                   default=Path('runs/geomfrac_s5_free11_plan.json'))
    p.add_argument('--bb-manifest',type=Path,
                   default=Path('runs/geomfrac_s5_free11_manifest.json'))
    p.add_argument('--config',type=Path,
                   default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--model',type=Path,
                   default=Path('runs/geomfrac_et_v5_closed.pt'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_portself_v6.npz'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_portself_v6.json'))
    a=p.parse_args()
    torch.set_num_threads(2)
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    bp=json.loads(a.bb_plan.read_text(encoding='utf-8-sig'))
    bm=json.loads(a.bb_manifest.read_text(encoding='utf-8-sig'))
    train=load_rows(plan,mf,'train')
    dev=load_rows(plan,mf,'val')
    bb=load_rows(bp,bm,'test')
    if ({r['name'] for r in train}&{r['name'] for r in dev+bb}
            or {r['name'] for r in dev}&{r['name'] for r in bb}):
        raise ValueError('cold training/dev/BB must be geometry-disjoint')
    atlas=GeomInit.load(cfg['atlas'])
    mu=MutFix.load(cfg['mutual_model'])
    src=SrcFix.load(cfg['source_model'])
    tan=load_z(cfg['z_fix_model'])
    base=PortSelf.load(cfg['port_self_model'])
    initial=cold_score(dev,atlas,base,mu)
    best=None
    for dims in (2,3):
        for ell in (.5,1.,1.5,2.5,3.5,5.,7.,10.):
            for lam in (.001,.01,.05,.1,.3):
                ps=PortSelf.fit(train,dims,ell,lam)
                score=cold_score(dev,atlas,ps,mu)
                objective=score['cold_current_pct']+.3*score['cold_Z_pct']
                if best is None or objective<best[0]:
                    best=(objective,ps,dims,ell,lam,score)
    _,candidate,dims,ell,lam,selected=best
    report=dict(status='candidate_diagnostic',
                train_geometries=len(train),dev_geometries=len(dev),
                BB_geometries=len(bb),train_source='42G cold S5 anchors',
                dims=dims,ell=ell,lam=lam,score_rule='dev current + 0.3 cold Z',
                baseline_dev=initial,selected_dev=selected,
                baseline_BB=cold_score(bb,atlas,base,mu),
                candidate_BB=cold_score(bb,atlas,candidate,mu))
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    for name,subset,pn in [('dev',mf['val'],plan),
                           ('BB',bm['test'],bp)]:
        print('DYNAMIC_BEGIN',name,flush=True)
        report['dynamic_'+name]={}
        for label,ps in [('baseline',base),('candidate',candidate)]:
            res=dyn_score(subset,pn,atlas,ps,mu,src,tan,net,ck,
                          float(cfg['shape_mix']))
            report['dynamic_'+name][label]=res
            print('DYNAMIC',name,label,
                  round(res['dynamic_Z_pct'],5),flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    candidate.save(a.out)
    report['model']=str(a.out)
    report['decision']='diagnostic_only; independent new geometry acceptance required'
    a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('RESULT',json.dumps({key:report[key] for key in
        ('dims','ell','lam','baseline_dev','selected_dev')},
        default=float),flush=True)


if __name__=='__main__':
    main()
