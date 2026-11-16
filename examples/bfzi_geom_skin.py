"""Fit V7 seawater-skin modes on frozen V5 neural-state trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_z_tangent import restore as load_tan
from sdfmpneo_bfzi.geom_canon import basis,mode_scale
from sdfmpneo_bfzi.geom_neural import GeomFracET,rollout
from sdfmpneo_bfzi.geom_skin import SkinFix, reconstruct
from sdfmpneo_bfzi.geom_pkg import restore as load_pkg


def gather(specs,net,ck,tan,pkg=None,lift=None,rank=48,base_skin=None):
    data=[]
    for spec in specs:
        for row in prepare(spec,32):
            rom=row['rom']
            with torch.no_grad():
                pred=rollout(rom,net,ck,row['voltage'],row['time'],
                             tangent=tan,pkg=pkg,lift=lift)
            n=len(row['time'])
            phi=basis(rom,rank)
            sc=mode_scale(rom,rank)
            base=pred['temp_K']
            if base_skin is not None:
                base=reconstruct(rom,base,pred['heat_state'],row['time'],
                                 row['voltage'],base_skin,rank=48)
            y=((row['temp_K']-base)*rom.cap)@phi[:,rank-16:]/sc[rank-16:]
            t=np.asarray(row['time'],float).reshape(-1,1)
            vin=np.r_[0.,row['voltage'][:-1]].reshape(-1,1)
            data.append(dict(file=row['file'],row=row,rom=rom,
                geom=torch.as_tensor(np.broadcast_to(row['geom'],(n,len(row['geom']))).copy(),
                                     dtype=torch.float32),
                state=torch.as_tensor(pred['heat_state'],dtype=torch.float32),
                time=torch.as_tensor(t,dtype=torch.float32),
                voltage=torch.as_tensor(vin,dtype=torch.float32),
                y=torch.as_tensor(y,dtype=torch.float32),
                base=base,true=row['temp_K'],
                extra=phi[:,rank-16:]*sc[rank-16:]))
        print('SKIN_PREP',spec['anchor'],flush=True)
    return data


def metric(model,rows):
    sea=[]
    whole=[]
    with torch.no_grad():
        for r in rows:
            extra=model(r['geom'],r['state'],r['time'],r['voltage']).numpy()
            nwet=int(r['rom'].nwet)
            nskin=int(r['rom'].nskin)
            sl=slice(nwet,nwet+nskin)
            pred=r['base']+extra@r['extra'].T
            sea.append(float(np.max(abs(pred[:,sl]-r['true'][:,sl]))))
            whole.append(float(np.max(abs(pred-r['true']))))
    return float(np.mean(whole)),float(np.mean(sea)),whole


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tangent-model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=350)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--patience',type=int,default=80)
    p.add_argument('--pkg-model',type=Path,default=None,
                   help="train residual on autonomous V6 package rollout")
    p.add_argument('--lift-model',type=Path,default=None,
                   help="train residual on V6 linear seawater feedback")
    p.add_argument('--lift-gain',type=float,default=.2)
    p.add_argument('--rank',type=int,choices=(48,64),default=48)
    p.add_argument('--base-skin-model',type=Path,default=None,
                   help="frozen V7 decoder before rank-64 bulk residual")
    a=p.parse_args()
    torch.manual_seed(59)
    torch.set_num_threads(2)
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state']);net.eval()
    tan=load_tan(a.tangent_model)
    pkg=load_pkg(a.pkg_model) if a.pkg_model else None
    lift=None
    if a.lift_model:
        with np.load(a.lift_model) as data:
            lift=data['w']*a.lift_gain
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    base_skin=None
    if a.base_skin_model:
        from sdfmpneo_bfzi.geom_skin import restore
        base_skin=restore(a.base_skin_model)
    train=gather(mf['train'],net,ck,tan,pkg,lift,a.rank,base_skin)
    val=gather(mf['val'],net,ck,tan,pkg,lift,a.rank,base_skin)
    fix=SkinFix(ng=int(ck['ng']),scale=np.asarray(ck['a_scale']))
    X=torch.cat([r['state'] for r in train])
    G=torch.cat([r['geom'] for r in train])
    T=torch.cat([r['time'] for r in train])
    V=torch.cat([r['voltage'] for r in train])
    Y=torch.cat([r['y'] for r in train])
    sy=Y.std(0).clamp_min(.05)
    opt=torch.optim.AdamW(fix.parameters(),lr=a.lr,weight_decay=.001)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.65,patience=18)
    best,init_sea,_=metric(fix,val)
    initial=best
    best_ep=0
    saved={k:v.detach().clone() for k,v in fix.state_dict().items()}
    print('SKIN_INIT',round(best,5),flush=True)
    for ep in range(1,a.epochs+1):
        fix.train()
        opt.zero_grad(set_to_none=True)
        pred=fix(G,X,T,V)
        loss=(((pred-Y)/sy)**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fix.parameters(),5.)
        opt.step()
        if ep%5:continue
        fix.eval()
        value,sea_val,_=metric(fix,val)
        sched.step(value)
        if value<best-1e-5:
            best,best_ep=value,ep
            saved={k:v.detach().clone() for k,v in fix.state_dict().items()}
        if ep%50==0:
            print('SKIN_TRAIN',ep,'mse',round(float(loss.detach()),5),
                  'dev_all_max_K',round(value,4),
                  'dev_sea_max_K',round(sea_val,4),'best',round(best,4),
                  'epoch',best_ep,flush=True)
        if ep-best_ep>=a.patience:break
    fix.load_state_dict(saved)
    offline,final_sea,val_errors=metric(fix,val)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(ng=fix.ng,scale=fix.scale,state=fix.state_dict(),
             train_files=[r['file'] for r in train],
             val_files=[r['file'] for r in val],
             src_model=str(a.model),tangent_model=str(a.tangent_model),
             pkg_model=str(a.pkg_model) if a.pkg_model else None,
             lift_model=str(a.lift_model) if a.lift_model else None,
             lift_gain=a.lift_gain,rank=a.rank,
             base_skin_model=str(a.base_skin_model) if a.base_skin_model else None),a.out)
    report=dict(best_epoch=best_ep,epochs_trained=ep,train_geometries=len(mf['train']),
       val_geometries=len(mf['val']),train_histories=len(train),val_histories=len(val),
       dev_global_baseline_max_K=initial,dev_global_predicted_max_K=offline,
       dev_skin_baseline_max_K=init_sea,dev_skin_predicted_max_K=final_sea,
       pkg_model=str(a.pkg_model) if a.pkg_model else None,
       lift_model=str(a.lift_model) if a.lift_model else None,
       rank=a.rank,
       base_skin_model=str(a.base_skin_model) if a.base_skin_model else None,
       val_history_max_K=[dict(file=r['file'],temperature_max_K=e)
           for r,e in zip(val,val_errors)])
    a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('SKIN_FINAL',best_ep,'global_base',round(initial,3),
          'global_predict',round(offline,3),'sea_predict',round(final_sea,3),flush=True)


if __name__=='__main__':
    main()
