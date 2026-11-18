"""Train a geometry-only cold two-port model on disjoint BFZI V150 samples."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.geom_coldnet import ColdNet,feat,target,unpack


def _read(path):
    if path is None:return dict(train=[],val=[],test=[])
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _rows(new,old,part):
    rows=list(new.get(part,[]))
    if part=='val':
        rows+=old.get('dev',[])
    elif part=='train':
        rows+=old.get('train',[])
    return rows


def _mat(z):
    a,b,c,d,e,f=z
    return np.array([[a+1j*b,c+1j*d],[c+1j*d,e+1j*f]])


def eval_rows(net,rows):
    if not rows:return None
    zerr=[];aerr=[];curr=[];names=[]
    for r in rows:
        pred,air=net.predict(r['geometry'])
        z=np.asarray(r['z0'],float)
        aa=np.asarray(r['air'],float)
        if aa.shape==(2,2):aa=aa[[0,0,1],[0,1,1]]
        zerr.append(float(np.linalg.norm(pred-z)/max(np.linalg.norm(z),1e-12)))
        aerr.append(float(np.linalg.norm(air-aa)/max(np.linalg.norm(aa),1e-12)))
        if 'ext_re' in r:
            ext=np.asarray(r['ext_re'])+1j*np.asarray(r['ext_im'])
            ref=np.linalg.solve(_mat(z)+ext,[1.,0.])
            got=np.linalg.solve(_mat(pred)+ext,[1.,0.])
            curr.append(float(np.linalg.norm(got-ref)/max(np.linalg.norm(ref),1e-12)))
        names.append(r['name'])
    return dict(geometries=len(rows),
                cold_Z_pct=100*float(np.mean(zerr)),
                air_L_pct=100*float(np.mean(aerr)),
                current_pct=100*float(np.mean(curr)) if curr else None,
                per_geometry={n:100*v for n,v in zip(names,zerr)})


def train(new,old,*,epochs=400,patience=70,lr=.0006,seed=41,
          batch=64,out='runs/geomfrac_coldnet_v1.pt'):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    tr=_rows(new,old,'train')
    dev=_rows(new,old,'val')
    test=list(new.get('test',[]))
    if len(tr)<20 or len(dev)<3:
        raise ValueError('need at least 20 training and 3 validation geometries')
    ids={part:{r['name'] for r in rows} for part,rows in
         [('train',tr),('val',dev),('test',test)]}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if ids[a]&ids[b]: raise ValueError('geometry split overlap')
    x=np.stack([feat(r['geometry']) for r in tr])
    y=np.stack([target(r) for r in tr])
    xm=x.mean(0)
    xs=np.maximum(x.std(0),np.array([.02,.02,1,1,2,2,2,2,2,2,2,2,
                                    .01,.1,.01,.1],dtype=np.float32))
    ym=y.mean(0)
    ys=np.maximum(y.std(0),np.array([.01,.003,.003,.003,.01,.003,
                                    .25,.025,.25],dtype=np.float32))
    net=ColdNet(xm,xs,ym,ys)
    inp=torch.as_tensor(x)
    tgt=torch.as_tensor((y-ym)/ys)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    best=float('inf');best_ep=0
    best_state=copy.deepcopy(net.state_dict())
    hist=[]
    rng=np.random.default_rng(seed)
    for ep in range(1,epochs+1):
        net.train()
        order=rng.permutation(len(tr))
        for start in range(0,len(tr),batch):
            ix=order[start:start+batch]
            output=net(inp[ix])
            loss=((output-torch.as_tensor(y[ix]))/net.ys).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep%5:continue
        net.eval()
        d=eval_rows(net,dev)
        metric=d['cold_Z_pct']+(.1*d['current_pct']
                              if d['current_pct'] is not None else 0)
        hist.append(dict(epoch=ep,train_loss=float(loss.detach()),
                         dev_Z_pct=d['cold_Z_pct'],dev_current_pct=d['current_pct']))
        if metric<best:
            best=metric;best_ep=ep
            best_state=copy.deepcopy(net.state_dict())
        if ep%25==0:
            print('COLD_TRAIN',ep,'dev_Z',round(d['cold_Z_pct'],4),
                  'best',best_ep,flush=True)
        if ep-best_ep>=patience:break
    net.load_state_dict(best_state);net.eval()
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    net.save(out)
    return dict(status='trained_candidate',checkpoint=str(out),
        best_epoch=best_ep,epochs_trained=ep,
        train_geometries=len(tr),val_geometries=len(dev),
        independent_test_geometries=len(test),
        val=eval_rows(net,dev),test=eval_rows(net,test),
        history=hist,model='ColdNet air+sea/copper-decomposed z6')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,
                   default=Path('runs/geomfrac_cold500/data.json'))
    p.add_argument('--old',type=Path,
                   default=Path('runs/geomfrac_cold_ports_v1.json'))
    p.add_argument('--out',type=Path,
                   default=Path('runs/geomfrac_coldnet_v1.pt'))
    p.add_argument('--report',type=Path,
                   default=Path('runs/geomfrac_coldnet_v1.json'))
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--batch',type=int,default=64)
    a=p.parse_args()
    res=train(_read(a.data),_read(a.old),epochs=a.epochs,batch=a.batch,out=a.out)
    a.report.write_text(json.dumps(res,indent=2)+'\n',encoding='utf-8')
    print('COLD_RESULT',json.dumps({k:res[k] for k in
          ('best_epoch','train_geometries','val_geometries',
           'independent_test_geometries','val','test')}),flush=True)


if __name__=='__main__':
    main()
