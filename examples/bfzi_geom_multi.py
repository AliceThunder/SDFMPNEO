"""Train one GeomFracET across independent BFZI geometries in shared 32D heat space.

Manifest JSON:
{"train":[{"anchor":"runs/T01_anchor.npz","teacher":["runs/T01_a.npz"]}],
 "val":[{"anchor":"runs/T02_anchor.npz","teacher":["runs/T02_b.npz"]}]}

Every geometry receives the same five material-mean and 27 spatial mode
definitions. The FV operators and true spatial reconstructions remain local.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_neural import read,scales,score,eval_rollout,load_init
from sdfmpneo_bfzi.geom_canon import make_rom,mode_scale
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.s5_bridge import feature


def prepare(spec,rank=32):
    """Load a cold anchor and project its histories into common K-scale modes."""
    head=GeomHeat.load(spec['anchor'])
    rom=make_rom(head,rank)
    ng=45 if hasattr(head,'thermal') else (
        40 if getattr(head,'pose',None) is not None else 22)
    geom=feature(rom,ng)
    m=mode_scale(rom,rank)
    K=rom.K*m[None,:]/m[:,None]
    files=spec['teacher']
    if isinstance(files,str):
        files=[files]
    result=[]
    for file in files:
        row=read(file,rom)
        row['a']/=m
        row.update(geom=geom,K=K,rom=rom)
        reconstructed=rom.amb+(row['a']*m)@rom.phi.T
        row['basis_max_K']=float(np.max(abs(reconstructed-row['temp_K'])))
        result.append(row)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--init-model',type=Path,default=None)
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_et_multi.pt'))
    p.add_argument('--report',type=Path,default=Path('runs/geomfrac_et_multi.json'))
    p.add_argument('--rank',type=int,default=32,choices=(5,32))
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--patience',type=int,default=90)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--closed-weight',type=float,default=.2)
    p.add_argument('--therm-weight',type=float,default=.03)
    a=p.parse_args()
    config=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    trspec,vaspec=config['train'],config['val']
    trgeo={str(item['anchor']) for item in trspec}
    vageo={str(item['anchor']) for item in vaspec}
    if not trgeo or not vageo or trgeo & vageo:
        p.error('train and val require disjoint nonempty geometries')
    if any(re.search(r'(^|_)U(?:0[1-9]|1[0-2])(?:_|$)',Path(g).stem)
           for g in trgeo):
        p.error('U01-U12 are frozen independent physical-validation cases')
    torch.manual_seed(41)
    torch.set_num_threads(2)
    train=[r for spec in trspec for r in prepare(spec,a.rank)]
    val=[r for spec in vaspec for r in prepare(spec,a.rank)]
    if {r['file'] for r in train} & {r['file'] for r in val}:
        p.error('train and val teacher histories must be disjoint')
    dims={len(r['geom']) for r in train+val}
    if len(dims)!=1:
        p.error('geometry and thermal feature dimensions must match')
    sx,sy,sa=scales(train)
    net=GeomFracET(ng=dims.pop(),rank=a.rank)
    if a.init_model:
        old=torch.load(a.init_model,map_location='cpu',weights_only=True)
        load_init(net,old)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    best=float('inf')
    best_epoch=0
    saved=None
    for ep in range(a.epochs):
        net.train()
        opt.zero_grad(set_to_none=True)
        loss=score(net,train,None,sx,sy,sa,therm_w=a.therm_weight)
        if a.closed_weight and ep>=20:
            loss+=a.closed_weight*score(net,train,None,sx,sy,sa,
                        closed=True,therm_w=a.therm_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
        opt.step()
        net.eval()
        with torch.no_grad():
            v=float(score(net,val,None,sx,sy,sa,therm_w=a.therm_weight))
            if a.closed_weight:
                v+=a.closed_weight*float(score(net,val,None,sx,sy,sa,
                        closed=True,therm_w=a.therm_weight))
        if v<best-1e-8:
            best,best_epoch=v,ep+1
            saved={k:vv.detach().clone() for k,vv in net.state_dict().items()}
        if ep+1-best_epoch>=a.patience:
            break
    net.load_state_dict(saved)
    ck=dict(state=net.state_dict(),ng=net.ng,rank=net.rank,
            src_scale=torch.as_tensor(sx,dtype=torch.float32),
            y_scale=torch.as_tensor(sy,dtype=torch.float32),
            a_scale=torch.as_tensor(sa,dtype=torch.float32),
            canonical=True,port_sea=True,architecture='GeomFracET',
            train_files=[r['file'] for r in train],
            val_files=[r['file'] for r in val])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck,a.out)
    autonomous=eval_rollout(None,net,ck,val)
    metrics=dict(architecture='GeomFracET-cross-geometry',
                 mode='canonical-material-32',rank=net.rank,
                 ntrain_geometries=len(trgeo),nval_geometries=len(vageo),
                 train_histories=len(train),val_histories=len(val),
                 best_epoch=best_epoch,val_loss=best,
                 train_projection_max_K=max(r['basis_max_K'] for r in train),
                 val_projection_max_K=max(r['basis_max_K'] for r in val),
                 autonomous_validation=autonomous)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
