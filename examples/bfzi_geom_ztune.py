"""Train a nonlinear seawater impedance tangent correction on V6 autonomous states."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_neural import eval_rollout
from examples.bfzi_z_tangent import restore as load_tan
from sdfmpneo_bfzi.geom_neural import GeomFracET,rollout
from sdfmpneo_bfzi.geom_pkg import restore as load_pkg
from sdfmpneo_bfzi.geom_ztune import ZFix


def get_data(specs,net,ck,tan,pkg,lift):
    out=[]
    for spec in specs:
        for r in prepare(spec,32):
            with torch.no_grad():
                p=rollout(r['rom'],net,ck,r['voltage'],r['time'],
                          tangent=tan,pkg=pkg,lift=lift)
            out.append(dict(row=r,state=p['heat_state'],
                            den=max(float(np.linalg.norm(r['z6']-r['z6'][0])),1e-11)))
        print('ZT_PREP',spec['anchor'],flush=True)
    return out


def pack(data,base,sy):
    groups=[]
    for r in data:
        row=r['row'];a=torch.as_tensor(r['state'][1:],dtype=torch.float32)
        g=torch.as_tensor(np.broadcast_to(row['geom'],(len(a),len(row['geom']))).copy(),
                          dtype=torch.float32)
        with torch.no_grad():b=base(g,a)/sy
        y=torch.as_tensor(row['y'][1:,:6],dtype=torch.float32)/sy
        groups.append(dict(g=g,a=a,b=b,y=y,den=r['den'],
                           file=row['file']))
    return groups


def loss_fn(model,rows,sy,reg=0.):
    vals=[]
    mags=[]
    for r in rows:
        corr=model(r['g'],r['a'])
        diff=(r['b']+corr-r['y'])*sy
        vals.append(diff.square().sum()/r['den']**2)
        mags.append(corr.square().mean())
    return torch.stack(vals).mean()+reg*torch.stack(mags).mean()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tangent-model',type=Path,required=True)
    p.add_argument('--pkg-model',type=Path,required=True)
    p.add_argument('--lift-model',type=Path,required=True)
    p.add_argument('--lift-gain',type=float,default=.2)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=240)
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--lr',type=float,default=.0005)
    a=p.parse_args()
    torch.manual_seed(67)
    torch.set_num_threads(2)
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
       soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
       heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    base=load_tan(a.tangent_model)
    pkg=load_pkg(a.pkg_model)
    with np.load(a.lift_model) as z:lift=z['w']*a.lift_gain
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    train=get_data(mf['train'],net,ck,base,pkg,lift)
    dev=get_data(mf['val'],net,ck,base,pkg,lift)
    sy=torch.as_tensor(np.asarray(ck['y_scale'])[:6],dtype=torch.float32)
    tr=pack(train,base,sy)
    model=ZFix(ng=int(ck['ng']),sa=np.asarray(ck['a_scale']))
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.002)
    best=float('inf')
    saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    best_ep=0
    history=[]
    def current():
        model.eval()
        with torch.no_grad():
            def tangent(g,aa):return base(g,aa)+model(g,aa)*sy
            scores=eval_rollout(None,net,ck,[v['row'] for v in dev],
                                tangent=tangent,pkg=pkg,lift=lift)
        return (float(np.mean([x['dynamic_Z_rel_L2'] for x in scores])),
                float(np.mean([x['temperature_max_K'] for x in scores])))
    b0,t0=current()
    best=b0
    print('ZT_INIT','Zpct',round(b0*100,4),'TempK',round(t0,4),flush=True)
    history.append(dict(epoch=0,z_pct=b0*100,temp_max_K=t0))
    for ep in range(1,a.epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=loss_fn(model,tr,sy,reg=.002)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
        if ep%a.eval_every:continue
        score,temp=current()
        history.append(dict(epoch=ep,z_pct=100*score,temp_max_K=temp))
        if score<best-1e-5:
            best=score
            best_ep=ep
            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
        print('ZT_DEV',ep,'train_loss',round(float(loss.detach()),5),
              'Zpct',round(score*100,4),'TempK',round(temp,4),
              'best_Zpct',round(best*100,4),'best_ep',best_ep,flush=True)
        if ep-best_ep>=120:break
    model.load_state_dict(saved)
    result=dict(architecture='nonlinear geometry-conditioned sea-Z residual tangent',
        best_epoch=best_ep,epochs_trained=ep,
        baseline_dev_Z_pct=100*b0,best_dev_Z_pct=100*best,
        baseline_dev_T_K=t0,
        train_geometries=len(mf['train']),dev_geometries=len(mf['val']),
        train_histories=len(train),dev_histories=len(dev),history=history,
        base_tangent=str(a.tangent_model),pkg_model=str(a.pkg_model),
        lift_model=str(a.lift_model),lift_gain=a.lift_gain)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state_dict(),ng=model.ng,
                    sa=torch.as_tensor(np.asarray(ck['a_scale'])),sy=sy,
                    base=str(a.tangent_model),
                    train_files=[x['row']['file'] for x in train]),a.out)
    a.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('ZT_FINAL',best_ep,'dynamicZpct',round(100*best,4),flush=True)


if __name__=='__main__':
    main()
