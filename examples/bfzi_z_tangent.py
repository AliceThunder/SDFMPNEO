"""Learn a geometry-dependent seawater impedance tangent from physical temperatures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from examples.bfzi_geom_multi import prepare


IDX=[0,*range(5,18)]  # sea mean plus 13 sea spatial modes


class Tangent(nn.Module):
    """Geometry-conditioned linear seawater thermal susceptibility."""

    def __init__(self,ng=45,nmode=len(IDX),hidden=64):
        super().__init__()
        self.nmode=nmode
        self.net=nn.Sequential(nn.Linear(ng,hidden),nn.SiLU(),
                               nn.Linear(hidden,6*nmode))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self,g,a):
        w=self.net(g).reshape(-1,6,self.nmode)
        return torch.einsum('bij,bj->bi',w,a)


def restore(path):
    """Load the learned geometry-conditioned tangent for autonomous coupling."""
    ck=torch.load(path,map_location='cpu',weights_only=True)
    net=Tangent(ng=int(ck['ng']),nmode=len(ck['idx']))
    net.load_state_dict(ck['state'])
    net.eval()
    for v in net.parameters():
        v.requires_grad_(False)
    ids=list(ck['idx'])
    sc=torch.as_tensor(np.asarray(ck['sa'])[ids],dtype=torch.float32)
    sy=torch.as_tensor(np.asarray(ck['sy'])[:6],dtype=torch.float32)

    def predict(g,a):
        return net(g,a[:,ids]/sc)*sy
    return predict


def load(specs,sa,sy,tau=0,state_net=None,state_ck=None,
         base_tangent=None,mix=0.):
    groups=[]
    for spec in specs:
        for r in prepare(spec,32):
            den=max(float(np.linalg.norm(r['z6']-r['z6'][0])),1e-12)
            weight=1/(1+(r['basis_max_K']/tau)**2) if tau else 1.
            true_a=r['a']
            if state_net is not None:
                from sdfmpneo_bfzi.geom_neural import rollout
                pred=rollout(r['rom'],state_net,state_ck,
                             r['voltage'],r['time'],tangent=base_tangent)
                input_a=pred['heat_state']
            else:
                input_a=true_a
            for arr,factor in ([(input_a,1.),(true_a,mix)]
                                if state_net is not None and mix else
                                [(input_a,1.)]):
                groups.append(dict(
                    g=np.repeat(r['geom'][None],len(r['time'])-1,axis=0),
                    a=arr[1:,IDX]/sa[IDX],
                    y=r['y'][1:,:6]/sy[:6],
                    den=den,
                    weight=weight*factor,
                    file=r['file']))
        print('LOAD_TANGENT',spec['anchor'],flush=True)
    n=len(groups)
    arrays={}
    for k in ('g','a','y'):
        arrays[k]=torch.as_tensor(np.concatenate([v[k] for v in groups]),
                                  dtype=torch.float32)
    arrays['row']=groups
    arrays['lens']=[len(v['a']) for v in groups]
    return arrays


def score(net,d,sy):
    pred=net(d['g'],d['a'])
    ref=d['y']
    diff=(pred-ref)*torch.as_tensor(sy[:6],dtype=pred.dtype)
    err=[]
    start=0
    for r,n in zip(d['row'],d['lens']):
        val=diff[start:start+n].square().sum()/(r['den']**2)
        err.append(val)
        start+=n
    return torch.stack(err)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--init-model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=700)
    p.add_argument('--tau',type=float,default=5.)
    p.add_argument('--state-model',type=Path,default=None)
    p.add_argument('--base-tangent',type=Path,default=None)
    p.add_argument('--tangent-init',type=Path,default=None)
    p.add_argument('--mix-true',type=float,default=0.)
    a=p.parse_args()
    torch.manual_seed(41)
    torch.set_num_threads(2)
    ck=torch.load(a.init_model,map_location='cpu',weights_only=True)
    sa=np.asarray(ck['a_scale'])
    sy=np.asarray(ck['y_scale'])
    mf=json.loads(a.manifest.read_text(encoding='utf-8'))
    state_net=None
    base_tangent=None
    if a.state_model:
        from sdfmpneo_bfzi.geom_neural import GeomFracET
        state_ck=torch.load(a.state_model,map_location='cpu',weights_only=True)
        state_net=GeomFracET(ng=int(state_ck['ng']),rank=int(state_ck['rank']),
                            soe_quad=state_ck.get('soe_quad','trap'),
                            heat_dt=float(state_ck.get('heat_dt',1.875)),
                            heat_z=bool(state_ck.get('heat_z',False)))
        state_net.load_state_dict(state_ck['state'])
        state_net.eval()
        if a.base_tangent:
            base_tangent=restore(a.base_tangent)
    else:
        state_ck=None
    if a.mix_true<0 or a.mix_true>1:
        p.error('mix-true must be between 0 and 1')
    tr=load(mf['train'],sa,sy,a.tau,state_net,state_ck,base_tangent,a.mix_true)
    val=load(mf['val'],sa,sy,0,state_net,state_ck,base_tangent)
    model=Tangent(ng=tr['g'].shape[1])
    if a.tangent_init:
        init=torch.load(a.tangent_init,map_location='cpu',weights_only=True)
        model.load_state_dict(init['state'])
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode='min',factor=.6,patience=10)
    best=float('inf')
    saved=None
    best_ep=0
    if a.tangent_init:
        with torch.no_grad():
            best=float(score(model,val,sy).sqrt().mean())
        saved={k:v.detach().clone() for k,v in model.state_dict().items()}
        print('TANGENT_INIT',round(best,6),flush=True)
    train_w=torch.tensor([v['weight'] for v in tr['row']],dtype=torch.float32)
    for ep in range(1,a.epochs+1):
        opt.zero_grad(set_to_none=True)
        loss=(score(model,tr,sy)*train_w).sum()/train_w.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),3.)
        opt.step()
        if ep%5:
            continue
        model.eval()
        with torch.no_grad():
            e=score(model,val,sy)
            metric=float(torch.sqrt(e).mean())
        sched.step(metric)
        if metric<best-1e-7:
            best=metric
            best_ep=ep
            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
        if ep%50==0:
            print('TANGENT',ep,'train',round(float(loss),5),
                  'dev',round(metric,5),'best',best_ep,flush=True)
        if ep-best_ep>=120:
            break
    model.load_state_dict(saved)
    with torch.no_grad():
        vals=score(model,val,sy).sqrt().tolist()
    out=dict(architecture='geometry-conditioned-linear-sea-tangent',
             best_epoch=best_ep,epochs_trained=ep,
             dev_sea_dynamic_mean_rel_L2=float(np.mean(vals)),
             dev_results=[dict(file=x['file'],dynamic_Z_rel_L2=float(y))
                          for x,y in zip(val['row'],vals)],
             training_geometries=len(mf['train']),
             dev_geometries=len(mf['val']),tau=a.tau,
             state_model=str(a.state_model) if a.state_model else None,
             base_tangent=str(a.base_tangent) if a.base_tangent else None,
             mix_true=a.mix_true)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state_dict(),ng=tr['g'].shape[1],
                    sa=torch.as_tensor(sa),sy=torch.as_tensor(sy),idx=IDX,
                    train_files=[r['file'] for r in tr['row']]),
               a.out)
    a.report.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('TANGENT_FINAL',round(100*out['dev_sea_dynamic_mean_rel_L2'],3),
          'epoch',best_ep,flush=True)


if __name__ == '__main__':
    main()
