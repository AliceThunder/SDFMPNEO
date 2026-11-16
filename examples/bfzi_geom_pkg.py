"""Fit package spatial-state correction from autonomous BFZI neural trajectories."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_z_tangent import restore as restore_tan
from sdfmpneo_bfzi.geom_canon import mode_scale
from sdfmpneo_bfzi.geom_neural import GeomFracET,rollout
from sdfmpneo_bfzi.geom_pkg import PkgFix,PKG


def gather(specs,net,ck,tan):
    data=[]
    for spec in specs:
        for row in prepare(spec,32):
            rom=row['rom']
            with torch.no_grad():
                pred=rollout(rom,net,ck,row['voltage'],row['time'],tangent=tan)
            state=pred['heat_state']
            x=torch.as_tensor(state,dtype=torch.float32)
            geom=torch.as_tensor(np.repeat(row['geom'][None],len(x),axis=0),dtype=torch.float32)
            target=torch.as_tensor(row['a'][:,PKG]-state[:,PKG],dtype=torch.float32)
            v=mode_scale(rom,32)[PKG]
            pm=rom.phi[rom.ns:rom.cu.start,PKG]*v
            data.append(dict(file=row['file'],geom=geom,x=x,target=target,
                delta=torch.as_tensor(pm.T,dtype=torch.float32),
                base=torch.as_tensor(pred['temp_K'][:,rom.ns:rom.cu.start],dtype=torch.float32),
                true=torch.as_tensor(row['temp_K'][:,rom.ns:rom.cu.start],dtype=torch.float32),
                rom=rom,row=row))
        print('PKG_PREP',spec['anchor'],flush=True)
    return data


def metric(fix,rows):
    vals=[]
    with torch.no_grad():
        for r in rows:
            field=r['base']+fix(r['geom'],r['x'])@r['delta']
            vals.append(float(torch.max(abs(field-r['true']))))
    return float(np.mean(vals)),vals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--tangent-model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=350)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--patience',type=int,default=70)
    a=p.parse_args()
    torch.manual_seed(47)
    torch.set_num_threads(2)
    ck=torch.load(a.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
        soe_quad=ck.get('soe_quad','trap'),heat_dt=float(ck.get('heat_dt',1.875)),
        heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    net.eval()
    tan=restore_tan(a.tangent_model)
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    train=gather(mf['train'],net,ck,tan)
    val=gather(mf['val'],net,ck,tan)
    fix=PkgFix(ng=int(ck['ng']),scale=np.asarray(ck['a_scale']))
    X=torch.cat([r['x'] for r in train])
    G=torch.cat([r['geom'] for r in train])
    Y=torch.cat([r['target'] for r in train])
    ys=Y.std(0).clamp_min(.2)
    opt=torch.optim.AdamW(fix.parameters(),lr=a.lr,weight_decay=.002)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,factor=.65,patience=15)
    best,_=metric(fix,val)
    init=best
    best_ep=0
    saved={k:v.detach().clone() for k,v in fix.state_dict().items()}
    print('PKG_INIT',round(best,5),flush=True)
    for ep in range(1,a.epochs+1):
        opt.zero_grad(set_to_none=True)
        pred=fix(G,X)
        loss=(((pred-Y)/ys)**2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fix.parameters(),5.)
        opt.step()
        if ep%5:
            continue
        v,_=metric(fix,val)
        sched.step(v)
        if v<best-1e-5:
            best,best_ep=v,ep
            saved={k:x.detach().clone() for k,x in fix.state_dict().items()}
        if ep%50==0:
            print('PKG_TRAIN',ep,'loss',round(float(loss.detach()),5),
                  'devK',round(v,4),'best',round(best,4),'epoch',best_ep,flush=True)
        if ep-best_ep>=a.patience:
            break
    fix.load_state_dict(saved)
    offline,vals=metric(fix,val)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(ng=fix.ng,scale=fix.scale,state=fix.state_dict(),
                    train_files=[r['file'] for r in train],
                    val_files=[r['file'] for r in val],
                    src_model=str(a.model),tangent_model=str(a.tangent_model)),a.out)
    auto=[]
    for r in val:
        with torch.no_grad():
            out=rollout(r['rom'],net,ck,r['row']['voltage'],r['row']['time'],
                        tangent=tan,pkg=fix)
        s=r['rom'].ns;e=r['rom'].cu.start
        auto.append(dict(file=r['file'],
                         pkg_max_K=float(np.max(abs(out['temp_K'][:,s:e]-
                             r['row']['temp_K'][:,s:e]))),
                         temp_max_K=float(np.max(abs(out['temp_K']-
                             r['row']['temp_K']))),
                         material_budget_max_W=float(np.max(abs(out['material_budget_W'])))))
    report=dict(best_epoch=best_ep,epochs_trained=ep,
                offline_val_pkg_base_max_K=init,
                offline_val_pkg_corrected_max_K=offline,
                autonomous_pkg_mean_max_K=float(np.mean([v['pkg_max_K'] for v in auto])),
                autonomous_temp_mean_max_K=float(np.mean([v['temp_max_K'] for v in auto])),
                dev_histories=auto)
    a.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('PKG_FINAL',best_ep,'offline',round(init,3),round(offline,3),
          'autonomous',round(report['autonomous_pkg_mean_max_K'],3),
          round(report['autonomous_temp_mean_max_K'],3),flush=True)


if __name__=='__main__':
    main()
