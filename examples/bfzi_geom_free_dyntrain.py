"""Train geometry-only closed-loop dynamic-Z correction from full S5 histories."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_dyn import DynFix
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit,encode
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def records(mf,pl,atlas,net,ck,tan,split,report=True,
            port_self=None,mutual=None,source=None,shape_mix=0.):
    rows=[]
    for i,spec in enumerate(mf[split]):
        name=Path(spec['anchor']).stem.split('_')[2]
        case=pl['cases'][name]
        init=atlas.predict(case)
        if port_self is not None:
            init=port_self.apply(case,init)
        if mutual is not None:
            init=mutual.apply(case,init)
        if source is not None:
            init=source.apply(case,init)
        for row in prepare(spec,32):
            with torch.no_grad():
                pred=rollout(case,atlas,net,ck,row['time'],
                             row['voltage'],tan,init_override=init,shape_mix=shape_mix)
            truth=np.asarray(row['z6'])
            base=np.asarray(pred['z6'])
            target=(truth-truth[0])-(base-base[0])
            denom=max(float(np.linalg.norm(truth-truth[0])),1e-11)
            rows.append(dict(name=name,case=case,row=row,init=init,
                g=torch.as_tensor(np.repeat(encode(case)[None],len(truth),axis=0),
                                  dtype=torch.float32),
                a=torch.as_tensor(pred['heat_state'],dtype=torch.float32),
                t=torch.as_tensor(row['time'][:,None],dtype=torch.float32),
                v=torch.as_tensor(row['voltage'][:,None],dtype=torch.float32),
                target=torch.as_tensor(target,dtype=torch.float32),
                den=denom))
        if report and ((i+1)%5==0 or i+1==len(mf[split])):
            print('DYN_PREP',split,i+1,len(mf[split]),flush=True)
    return rows


def objective(model,rows,weights):
    errs=[];reg=[]
    w=torch.as_tensor(weights,dtype=torch.float32)
    for r in rows:
        z=model(r['g'],r['a'],r['t'],r['v'])
        errs.append((z.sub(r['target']).square()*w).sum()/r['den']**2)
        reg.append((z.square()*w).sum()/r['den']**2)
    return torch.stack(errs).mean()+.01*torch.stack(reg).mean()


def closed_score(model,rows,atlas,net,ck,tan,field=None,shape_mix=0.):
    from examples.bfzi_geom_free_eval import metric
    model.eval()
    out=[]
    with torch.no_grad():
        for r in rows:
            case=r['case']
            row=r['row']
            pred=rollout(case,atlas,net,ck,row['time'],
                         row['voltage'],tan,init_override=r['init'],extra=model,shape_mix=shape_mix)
            out.append(metric(pred,row))
    return dict(dynamic_Z_pct=float(np.mean([e['dynamic_z_rel_L2']
                                             for e in out])*100),
                cold_Z_pct=float(np.mean([e['cold_z_rel_L2']
                                           for e in out])*100),
                temp_group_K=float(np.mean([e['material_mean_max_K'] for e in out])),
                budget_W=float(max(e['energy_budget_W'] for e in out)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=Path('runs/geomfrac_s5_v4_plan.json'))
    parser.add_argument('--manifest',type=Path,
            default=Path('runs/geomfrac_s5_v4_30g_manifest.json'))
    parser.add_argument('--atlas',type=Path,default=Path('runs/geomfrac_init_v1.npz'))
    parser.add_argument('--model',type=Path,default=Path('runs/geomfrac_et_v4_8g_quality.pt'))
    parser.add_argument('--z-fix-model',type=Path,
                        default=Path('runs/geomfrac_et_v9_ztune.pt'))
    parser.add_argument('--port-self-model',type=Path,default=None)
    parser.add_argument('--mutual-model',type=Path,default=None)
    parser.add_argument('--source-model',type=Path,default=None)
    parser.add_argument('--out',type=Path,
                        default=Path('runs/geomfrac_free_dyn_v4.pt'))
    parser.add_argument('--report',type=Path,
                        default=Path('runs/geomfrac_free_dyn_v4_dev.json'))
    parser.add_argument('--geom-strength',type=float,default=1.)
    parser.add_argument('--shape-mix',type=float,default=0.5)
    parser.add_argument('--holdout',action='store_true')
    parser.add_argument('--epochs',type=int,default=240)
    parser.add_argument('--eval-every',type=int,default=40)
    parser.add_argument('--lr',type=float,default=.0006)
    parser.add_argument('--rx-r-weight',type=float,default=1.)
    parser.add_argument('--rx-x-weight',type=float,default=1.)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(91)
    ck=torch.load(args.model,map_location='cpu',weights_only=True)
    net=GeomFracET(ng=int(ck['ng']),rank=32,
            soe_quad=ck.get('soe_quad','trap'),
            heat_dt=float(ck.get('heat_dt',1.875)),
            heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    net.eval()
    atlas=GeomInit.load(args.atlas)
    tan=load_z(args.z_fix_model)
    port_self=None
    mutual=None
    source=None
    if args.port_self_model:
        from sdfmpneo_bfzi.geom_portself import PortSelf
        port_self=PortSelf.load(args.port_self_model)
    if args.mutual_model:
        from sdfmpneo_bfzi.geom_mutual import MutFix
        mutual=MutFix.load(args.mutual_model)
    if args.source_model:
        from sdfmpneo_bfzi.geom_srcfix import SrcFix
        source=SrcFix.load(args.source_model)
    pl=json.loads(args.plan.read_text(encoding='utf-8'))
    mf=json.loads(args.manifest.read_text(encoding='utf-8'))
    if args.holdout:
        hold_names={'N05','N10','N15','N20','N25','N27'}
        training=dict(mf,train=[s for s in mf['train']
            if Path(s['anchor']).stem.split('_')[2] not in hold_names])
        held=dict(mf,val=[s for s in mf['train']
            if Path(s['anchor']).stem.split('_')[2] in hold_names])
        hold=records(held,pl,atlas,net,ck,tan,'val',
                      port_self=port_self,mutual=mutual,source=source,shape_mix=args.shape_mix)
    else:
        training=mf
        hold=[]
    train=records(training,pl,atlas,net,ck,tan,'train',
                  port_self=port_self,mutual=mutual,source=source,
                  shape_mix=args.shape_mix)
    dev=records(mf,pl,atlas,net,ck,tan,'val',
                port_self=port_self,mutual=mutual,source=source,
                shape_mix=args.shape_mix)
    model=DynFix(atlas.mu,atlas.std,
        np.asarray(ck['a_scale']),np.asarray(ck['y_scale'])[:6],
        geom_strength=args.geom_strength)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.002)
    if args.rx_r_weight<=0 or args.rx_x_weight<=0:
        raise ValueError("RX loss weights must be positive")
    weights=[1.,1.,1.,1.,args.rx_r_weight,args.rx_x_weight]
    best_score=float('inf');best_ep=0
    saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    history=[]
    def measure(ep):
        nonlocal best_score,best_ep,saved
        score=closed_score(model,dev,atlas,net,ck,tan,shape_mix=args.shape_mix)
        hold_score=(closed_score(model,hold,atlas,net,ck,tan,shape_mix=args.shape_mix)
                    if hold else None)
        selection=(.5*score['dynamic_Z_pct']+
                   .5*hold_score['dynamic_Z_pct'] if hold_score else
                   score['dynamic_Z_pct'])
        history.append(dict(epoch=ep,**score,
                            hold_Z_pct=(hold_score['dynamic_Z_pct']
                                if hold_score else None),
                            selection_Z_pct=selection))
        if selection<best_score:
            best_score=selection
            best_ep=ep
            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
        print('DYN_DEV',ep,
          'dynamic_pct',round(score['dynamic_Z_pct'],4),
          'temp_K',round(score['temp_group_K'],4),
          'budget_W',round(score['budget_W'],10),
          'hold_pct',round(hold_score['dynamic_Z_pct'],3) if hold_score else None,
          'selected_pct',round(selection,3),
          'best_epoch',best_ep,flush=True)
    measure(0)
    for epoch in range(1,args.epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=objective(model,train,weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
        if epoch%args.eval_every==0:
            measure(epoch)
    model.load_state_dict(saved)
    result=dict(
        status='development-selected geometry-only causal dynamic Z head',
        train_geometries=len(training['train']),dev_geometries=len(mf['val']),
        hold_geometries=len(hold)//3,geom_strength=args.geom_strength,
        train_histories=len(train),dev_histories=len(dev),rx_r_weight=args.rx_r_weight,rx_x_weight=args.rx_x_weight,
        best_epoch=best_ep,baseline_dev_dynamic_Z_pct=history[0]['dynamic_Z_pct'],
        best_dev_dynamic_Z_pct=next(x['dynamic_Z_pct']
                    for x in history if x['epoch']==best_ep),
        best_selection_Z_pct=best_score,
        baseline_dev_temp_group_K=history[0]['temp_group_K'],
        history=history,
        atlas=str(args.atlas),head=str(args.out),
        port_self_model=str(args.port_self_model) if args.port_self_model else None,
        mutual_model=str(args.mutual_model) if args.mutual_model else None,
        source_model=str(args.source_model) if args.source_model else None,
        shape_mix=args.shape_mix)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state_dict(),mu=model.mu,std=model.std,
                    scale=model.scale,ys=model.ys,
                    train_names=[r['name'] for r in train],
                    best_epoch=best_ep,
                    geom_strength=args.geom_strength),args.out)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('DYN_FINAL',best_ep,best_score,flush=True)


if __name__=='__main__':
    main()

