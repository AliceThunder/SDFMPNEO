"""Fit cold TX/RX residuals from V150 labels while preserving V5 mutual Z."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.geom_coldres import ColdRes,_input
from sdfmpneo_bfzi.geom_coldres_pose import ColdResPose,_input_pose
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_portself import PortSelf


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _mat(z):
    a,b,c,d,e,f=z
    return np.array([[complex(a,b),complex(c,d)],
                     [complex(c,d),complex(e,f)]])


def score(rows,port,mut,exts):
    if not rows:return None
    zr=[];ar=[];ir=[];per={}
    for row in rows:
        case={'geometry':row['geometry']}
        p=mut.apply(case,port.apply(case,dict(z0=np.zeros(6),
                                            air=np.zeros(3))))
        z=np.asarray(row['z0'],float)
        zz=np.asarray(p['z0'],float)
        val=float(np.linalg.norm(zz-z)/max(np.linalg.norm(z),1e-12))
        zr.append(val)
        per[row['name']]=100*val
        air=np.asarray(row['air'],float)
        if air.shape==(2,2):air=air[[0,0,1],[0,1,1]]
        ar.append(float(np.linalg.norm(p['air']-air)/
                        max(np.linalg.norm(air),1e-12)))
        for ext in exts:
            try:
                ref=np.linalg.solve(_mat(z)+ext,[1.,0.])
                pred=np.linalg.solve(_mat(zz)+ext,[1.,0.])
                ir.append(float(np.linalg.norm(ref-pred)/
                                max(np.linalg.norm(ref),1e-12)))
            except np.linalg.LinAlgError:
                ir.append(float('inf'))
    return dict(geometries=len(rows),
                cold_Z_pct=100*float(np.mean(zr)),
                air_L_pct=100*float(np.mean(ar)),
                cold_current_pct=(100*float(np.mean(ir)) if ir else None),
                per_geometry=per)


def fit(new,old,cfg,*,out,epochs=400,lr=3e-4,
        batch=64,patience=85,seed=41,current_weight=.5,
        init_model=None,rx_only=False,pose_context=False):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    tr=list(new.get('train',[]))+list(old.get('train',[]))
    old_dev=list(old.get('dev',[]))
    dev=list(new.get('val',[]))
    test=list(new.get('test',[]))
    ids=[{r['name'] for r in part} for part in (tr,old_dev,dev,test)]
    for i in range(len(ids)):
        for j in range(i+1,len(ids)):
            if ids[i]&ids[j]:raise ValueError('cold geometry split overlap')
    if len(tr)<30 or not old_dev:raise ValueError('cold training set too small')
    base_path=str(cfg['port_self_model'])
    frozen=PortSelf.load(base_path)
    mut=MutFix.load(cfg['mutual_model'])
    exts={}
    for part in ('train','dev'):
        for row in old.get(part,[]):
            donor=row.get('donor')
            if donor and 'ext_re' in row and donor not in exts:
                exts[donor]=np.asarray(row['ext_re'])+1j*np.asarray(row['ext_im'])
    if not exts:
        raise ValueError('cold current loss needs known external network families')
    ext_list=np.stack([exts[k] for k in sorted(exts)])
    input_fn=_input_pose if pose_context else _input
    inp=np.stack([input_fn(r['geometry']) for r in tr])
    yy=np.stack([np.stack((np.asarray(r['z0'])[[0,1]],
                           np.asarray(r['z0'])[[4,5]])) for r in tr])
    oldval=np.stack([frozen.predict({'geometry':r['geometry']}) for r in tr])
    yy=(yy-oldval).astype(np.float32)
    xm=inp.mean(0)
    xs=np.maximum(inp.std(0),np.array([.02,1,2,.01,.1],
                                     dtype=np.float32))
    ys=np.maximum(yy.std(0),np.array([.01,.1],
                                    dtype=np.float32))
    if pose_context:
        if init_model:
            old_net=ColdRes.load(init_model)
            xm[:,:5]=old_net.xm.cpu().numpy()
            xs[:,:5]=old_net.xs.cpu().numpy()
            ys=old_net.ys.cpu().numpy()
            net=ColdResPose.from_old(old_net,xm,xs)
        else:
            net=ColdResPose(base_path,xm,xs,ys)
    else:
        net=ColdRes.load(init_model) if init_model else ColdRes(base_path,xm,xs,ys)
    if net.base_path!=base_path:
        raise ValueError('warm-start model uses a different PortSelf base')
    if rx_only:
        for param in net.head[0].parameters():
            param.requires_grad_(False)
    t=torch.as_tensor(inp,dtype=torch.float32)
    y=torch.as_tensor(yy,dtype=torch.float32)
    base_z=[]
    current_ref=[]
    for row in tr:
        case={'geometry':row['geometry']}
        init=mut.apply(case,frozen.apply(case,dict(z0=np.zeros(6),air=np.zeros(3))))
        base_z.append(init['z0'])
        current_ref.append([np.linalg.solve(_mat(row['z0'])+ext,
                            np.array([1.,0.],complex)) for ext in ext_list])
    base_z=torch.as_tensor(np.asarray(base_z),dtype=torch.float32)
    current_ref=torch.as_tensor(np.asarray(current_ref),dtype=torch.complex64)
    ext_t=torch.as_tensor(ext_list,dtype=torch.complex64)
    opt=torch.optim.AdamW((p for p in net.parameters() if p.requires_grad),
                          lr=lr,weight_decay=5e-4)
    ref=score(old_dev,frozen,mut,ext_list)
    def metric():
        net.eval()
        o=score(old_dev,net,mut,ext_list)
        n=score(dev,net,mut,ext_list)
        key=o['cold_current_pct']+.5*n['cold_current_pct']+.4*o['cold_Z_pct']+.2*n['cold_Z_pct']
        return key,o,n
    ref_new=score(dev,frozen,mut,ext_list) if dev else None
    init_old=score(old_dev,net,mut,ext_list) if init_model else ref
    init_new=score(dev,net,mut,ext_list) if init_model and dev else ref_new
    best=init_old['cold_current_pct']+.5*(init_new['cold_current_pct'] if init_new else 0.)
    best+=.4*init_old['cold_Z_pct']+.2*(init_new['cold_Z_pct'] if init_new else 0.)
    best_ep=0
    weights=copy.deepcopy(net.state_dict())
    hist=[]
    rng=np.random.default_rng(seed)
    for ep in range(1,epochs+1):
        net.train()
        for ids in np.array_split(rng.permutation(len(tr)),
                                  max(1,int(np.ceil(len(tr)/batch)))):
            pred=net(t[ids])
            delta=(pred-y[ids])/net.ys
            loss_z=delta.square().mean()+.005*(pred/net.ys).square().mean()
            zv=base_z[ids].clone()
            zv[:,:2]+=pred[:,0,:]
            zv[:,4:6]+=pred[:,1,:]
            a=torch.complex(zv[:,0],zv[:,1])
            b=torch.complex(zv[:,2],zv[:,3])
            c=torch.complex(zv[:,4],zv[:,5])
            zm=torch.stack((torch.stack((a,b),dim=-1),
                            torch.stack((b,c),dim=-1)),dim=-2)
            rhs=torch.zeros((len(ids),len(ext_list),2),dtype=torch.complex64)
            rhs[...,0]=1.
            ip=torch.linalg.solve(zm[:,None]+ext_t[None],rhs)
            ir=current_ref[ids]
            loss_i=((ip-ir).abs().square().sum(-1)/
                    ir.abs().square().sum(-1).clamp_min(1e-12)).mean()
            loss=loss_z+current_weight*loss_i
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep%5:continue
        key,o,n=metric()
        hist.append(dict(epoch=ep,train_loss=float(loss.detach()),
                         train_Z_loss=float(loss_z.detach()),
                         train_current_loss=float(loss_i.detach()),
                         olddev_Z_pct=o['cold_Z_pct'],
                         olddev_current_pct=o['cold_current_pct'],
                         newdev_Z_pct=None if n is None else n['cold_Z_pct'],
                         newdev_current_pct=None if n is None else n['cold_current_pct']))
        if key<best:
            best=key;best_ep=ep
            weights=copy.deepcopy(net.state_dict())
        if ep%25==0:
            print('RES_TRAIN',ep,'old_Z',round(o['cold_Z_pct'],4),
                  'old_I',round(o['cold_current_pct'],4),'best',best_ep,
                  flush=True)
        if ep-best_ep>=patience:break
    net.load_state_dict(weights)
    net.eval()
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    net.save(out)
    return dict(status='candidate' if best_ep else 'baseline',
                checkpoint=str(out),best_epoch=best_ep,epochs_trained=ep,
                train_geometries=len(tr),new_train_geometries=len(new.get('train',[])),
                baseline_old_dev=ref,residual_old_dev=score(old_dev,net,mut,ext_list),
                baseline_new_dev=ref_new,residual_new_dev=score(dev,net,mut,ext_list),
                baseline_new_test=score(test,frozen,mut,ext_list),
                residual_new_test=score(test,net,mut,ext_list),
                baseline_BB_diagnostic=score(old.get('bb_diagnostic',[]),
                                             frozen,mut,ext_list),
                residual_BB_diagnostic=score(old.get('bb_diagnostic',[]),
                                             net,mut,ext_list),
                current_loss_weight=float(current_weight),
                external_network_families=len(ext_list),
                warm_start=init_model,rx_only=bool(rx_only),
                pose_context=bool(pose_context),input_dim=int(inp.shape[-1]),
                initial_candidate_old_dev=init_old if init_model else None,
                initial_candidate_new_dev=init_new if init_model else None,
                history=hist)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,
                   default=Path('runs/geomfrac_cold500/data.json'))
    p.add_argument('--old',type=Path,
                   default=Path('runs/geomfrac_cold_ports_v1.json'))
    p.add_argument('--config',type=Path,
                   default=Path('runs/geomfrac_free_pose42_kell_v1.json'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_coldres_v1.pt'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_coldres_v1.json'))
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--current-weight',type=float,default=.5)
    p.add_argument('--init-model',type=Path,default=None)
    p.add_argument('--rx-only',action='store_true')
    p.add_argument('--pose-context',action='store_true')
    a=p.parse_args()
    res=fit(_read(a.data),_read(a.old),_read(a.config),
            out=str(a.out),epochs=a.epochs,batch=a.batch,
            current_weight=a.current_weight,
            init_model=str(a.init_model) if a.init_model else None,
            rx_only=a.rx_only,pose_context=a.pose_context)
    a.report.write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    o=res['residual_old_dev']
    print('RES_RESULT','best',res['best_epoch'],'train',res['train_geometries'],
          'oldZ',round(o['cold_Z_pct'],5),'oldI',
          round(o['cold_current_pct'],5),flush=True)


if __name__=='__main__':
    main()
