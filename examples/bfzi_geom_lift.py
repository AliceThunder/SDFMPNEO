"""Fit a low-cost neural-state-aware 48-mode sea-skin temperature decoder."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_z_tangent import restore
from sdfmpneo_bfzi.geom_canon import basis,mode_scale
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_lift import IDX,fit,reconstruct
from sdfmpneo_bfzi.geom_neural import GeomFracET,rollout


def run(spec,net,ck,tangent):
    head=GeomHeat.load(spec['anchor'])
    phi=basis(head,48)
    sc=mode_scale(head,48)
    out=[]
    for row in prepare(spec,32):
        with torch.no_grad():
            pred=rollout(row['rom'],net,ck,row['voltage'],row['time'],
                         tangent=tangent)
        target=((row['temp_K']-head.amb)*head.cap)@phi/sc
        out.append(dict(file=row['file'],head=head,
                        x=pred['heat_state'][:,IDX],
                        y=target[:,32:],temp=pred['temp_K'],
                        state=pred['heat_state'],true=row['temp_K']))
    return out


def score(rows,w):
    results=[]
    for r in rows:
        enhanced=reconstruct(r['head'],r['temp'],r['state'],w)
        results.append(dict(file=r['file'],
            base_max_K=float(np.max(abs(r['temp']-r['true']))),
            lift_max_K=float(np.max(abs(enhanced-r['true'])))))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tangent-model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--reg',type=float,default=1e-5)
    a=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    if int(ck['rank'])!=32:
        p.error('a rank-32 neural heat model is required')
    net=GeomFracET(ng=int(ck['ng']),rank=32,
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    tan=restore(a.tangent_model)
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    xs,ys=[],[]
    for spec in mf['train']:
        for r in run(spec,net,ck,tan):
            xs.append(r['x']);ys.append(r['y'])
    w=fit(np.concatenate(xs),np.concatenate(ys),a.reg)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,w=w,idx=np.array(IDX),rank=48,
                        reg=np.array(a.reg))
    report=dict(reg=a.reg,train_geometries=len(mf['train']),model=str(a.model),
                tangent_model=str(a.tangent_model))
    for split in ('val','test'):
        if mf.get(split):
            rows=[r for spec in mf[split] for r in run(spec,net,ck,tan)]
            result=score(rows,w)
            report[split]=dict(results=result,
                mean_base_max_K=float(np.mean([x['base_max_K'] for x in result])),
                mean_lift_max_K=float(np.mean([x['lift_max_K'] for x in result])))
            print(split,'baseline',round(report[split]['mean_base_max_K'],3),
                  'lift',round(report[split]['mean_lift_max_K'],3),flush=True)
    a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
