"""Select shared thermal atlas kernels using the complete BFZI-free closed loop.

Cold port, mutual and source corrections are fixed to their deployed models.
The candidate atlas therefore changes the shared K/capacity interpolation and
is selected only on the original development geometries plus six internal
training-geometry holdouts. The external BB set is never used here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_init import records
from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from examples.bfzi_geom_ztune import restore as load_tan
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET, rollout


def make_rows(specs, plan):
    out=[]
    for spec in specs:
        name=Path(spec['anchor']).stem.split('_')[2]
        case=plan['cases'][name]
        for row in prepare(spec,32):
            out.append(dict(case=case,row=row))
    return out


def score(rows, atlas, net, ck, tan, ps, mu, src, shape_mix):
    vals=[]
    with torch.no_grad():
        for item in rows:
            case=item['case']
            init=atlas.predict(case)
            init=ps.apply(case,init)
            init=mu.apply(case,init)
            init=src.apply(case,init)
            pred=rollout(case,atlas,net,ck,item['row']['time'],
                         item['row']['voltage'],tan,init_override=init,
                         shape_mix=shape_mix)
            vals.append(metric(pred,item['row']))
    return dict(
        dynamic_Z_pct=float(np.mean([x['dynamic_z_rel_L2'] for x in vals])*100),
        current_pct=float(np.mean([x['current_rel_L2'] for x in vals])*100),
        temp_group_K=float(np.mean([x['material_mean_max_K'] for x in vals])),
        energy_budget_W=float(max(x['energy_budget_W'] for x in vals)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--atlas',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--z-fix-model',type=Path,required=True)
    p.add_argument('--port-self-model',type=Path,required=True)
    p.add_argument('--mutual-model',type=Path,required=True)
    p.add_argument('--source-model',type=Path,required=True)
    p.add_argument('--shape-mix',type=float,default=.5)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--holdout',nargs='*',
                   default=['N05','N10','N15','N20','N25','N27'])
    args=p.parse_args()
    torch.set_num_threads(2)
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    mf=json.loads(args.manifest.read_text(encoding='utf-8'))
    names={Path(x['anchor']).stem.split('_')[2] for x in mf['train']}
    held=set(args.holdout)&names
    train_specs=[x for x in mf['train']
                 if Path(x['anchor']).stem.split('_')[2] not in held]
    hold_specs=[x for x in mf['train']
                if Path(x['anchor']).stem.split('_')[2] in held]
    if len(held)!=len(args.holdout):
        raise ValueError('one or more internal holdout geometries are missing')
    train=records(plan,dict(mf,train=train_specs),'train')
    guard=np.stack([x['guard'] for x in train])
    x=np.stack([r['x'] for r in train])
    y=np.stack([r['y'] for r in train])
    k=np.stack([r['k'] for r in train])
    src=np.stack([r['src'] for r in train])
    ext=np.stack([r['ext'] for r in train])
    dev_rows=make_rows(mf['val'],plan)
    hold_rows=make_rows(hold_specs,plan)
    ck=torch.load(args.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    tan=load_tan(args.z_fix_model)
    from sdfmpneo_bfzi.geom_portself import PortSelf
    from sdfmpneo_bfzi.geom_mutual import MutFix
    from sdfmpneo_bfzi.geom_srcfix import SrcFix
    ps=PortSelf.load(args.port_self_model)
    mu=MutFix.load(args.mutual_model)
    sf=SrcFix.load(args.source_model)
    old=GeomInit.load(args.atlas)
    base_dev=score(dev_rows,old,net,ck,tan,ps,mu,sf,args.shape_mix)
    base_hold=score(hold_rows,old,net,ck,tan,ps,mu,sf,args.shape_mix)
    base_select=.5*(base_dev['dynamic_Z_pct']+base_hold['dynamic_Z_pct'])
    results=[dict(label='baseline',dev=base_dev,hold=base_hold,
                  selection_pct=base_select)]
    best=(base_select,'baseline',None)
    for ell in (1.5,2.8,4.0,6.0):
        for lam in (.003,.03,.2):
            atlas=GeomInit.fit(x,y,k,src,ext,ell=ell,lam=lam)
            atlas.set_guard(guard)
            atlas.set_kscale(old.kexp)
            dv=score(dev_rows,atlas,net,ck,tan,ps,mu,sf,args.shape_mix)
            hv=score(hold_rows,atlas,net,ck,tan,ps,mu,sf,args.shape_mix)
            sel=.5*(dv['dynamic_Z_pct']+hv['dynamic_Z_pct'])
            label=f'ell{ell:g}_lam{lam:g}'
            results.append(dict(label=label,ell=ell,lam=lam,
                                dev=dv,hold=hv,selection_pct=sel))
            print('CLOSED_ATLAS',label,'dev',round(dv['dynamic_Z_pct'],4),
                  'hold',round(hv['dynamic_Z_pct'],4),
                  'selection',round(sel,4),flush=True)
            if sel<best[0]:
                best=(sel,label,atlas)
    promoted=best[2] is not None and best[0]<base_select-1e-4
    if promoted:
        best[2].save(args.out)
    report=dict(status='closed-loop K/capacity atlas kernel selection',
        train_geometries=len(train_specs),hold_geometries=len(hold_specs),
        dev_geometries=len(mf['val']),train_histories=sum(
            len(json.loads(Path(x['teacher'][0]).read_text())['time_s'])
            if False else 1 for x in train_specs),
        base_atlas=str(args.atlas),shape_mix=args.shape_mix,
        baseline_selection_pct=base_select,best_label=best[1],
        best_selection_pct=best[0],promoted=promoted,
        output=str(args.out) if promoted else None,
        candidates=results)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CLOSED_ATLAS_FINAL',best[1],round(best[0],4),
          'baseline',round(base_select,4),'promoted',promoted,flush=True)


if __name__=='__main__':
    main()
