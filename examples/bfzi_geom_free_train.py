"""Train the GeomFracET heat head through the deployed BFZI-free rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import free_metric, prepare
from sdfmpneo_bfzi.geom_free import rollout as deployed
from sdfmpneo_bfzi.geom_free_train import loss, rollout, setup
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def _name(row):
    return Path(row['file'].split('#')[0].replace('\\','/')).stem.split('_')[2]


def _net(ck):
    net=GeomFracET(ng=int(ck['ng']),rank=int(ck['rank']),
                   soe_quad=ck.get('soe_quad','trap'),
                   heat_dt=float(ck.get('heat_dt',1.875)),
                   heat_z=bool(ck.get('heat_z',False)))
    net.load_state_dict(ck['state'])
    for key,param in net.named_parameters():
        param.requires_grad_(key.startswith('heat_head.'))
    return net


def _data(mf,rank):
    tr=[r for spec in mf['train'] for r in prepare(spec,rank)]
    va=[r for spec in mf['val'] for r in prepare(spec,rank)]
    return tr,va


def _free(plan,cfg):
    atlas=GeomInit.load(cfg['atlas'])
    ps=PortSelf.load(cfg['port_self_model'])
    mu=MutFix.load(cfg['mutual_model'])
    src=SrcFix.load(cfg['source_model'])
    tan=load_z(cfg['z_fix_model'])
    return atlas,ps,mu,src,tan


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--init-model',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_et_free_v1.pt'))
    p.add_argument('--report',type=Path,default=Path('runs/geomfrac_et_free_v1.json'))
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-hist',type=int,default=2)
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--val-every',type=int,default=5)
    p.add_argument('--patience',type=int,default=25)
    p.add_argument('--temp-weight',type=float,default=.01)
    p.add_argument('--replay-tol',type=float,default=.005)
    p.add_argument('--min-improve',type=float,default=.001,
                   help='minimum absolute unseen dynamic-Z error reduction for promotion')
    a=p.parse_args()
    torch.manual_seed(41)
    torch.set_num_threads(2)
    mf=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    plan=json.loads(a.plan.read_text(encoding='utf-8-sig'))
    cfg=json.loads(a.config.read_text(encoding='utf-8-sig'))
    ck=torch.load(a.init_model,map_location='cpu',weights_only=True)
    net=_net(ck)
    tr,va=_data(mf,net.rank)
    seen=set(_name(r) for r in tr)
    dev=[r for r in va if _name(r) not in seen
         and _name(r).startswith(('D','T'))]
    dev_names={_name(r) for r in dev}
    replay=[r for r in va if _name(r) not in dev_names]
    if not tr or not dev or not replay:
        p.error('need nonempty training, unseen development and replay data')
    atlas,ps,mu,src,tan=_free(plan,cfg)
    cases={}
    names={_name(r) for r in tr+va}
    for name in names:
        if name not in plan['cases']:
            p.error('geometry missing from frozen case plan: '+name)
        case=plan['cases'][name]
        ini=atlas.predict(case)
        for model in (ps,mu,src):
            ini=model.apply(case,ini)
        cases[name]=ini
    states={name:setup(plan['cases'][name],cases[name],tan)
            for name in names}
    free=(plan,atlas,ps,mu,src,tan,float(cfg['shape_mix']))
    eck=dict(src_scale=ck['src_scale'],y_scale=ck['y_scale'],
             a_scale=ck['a_scale'],canonical=True,port_sea=True)
    def metrics():
        net.eval()
        with torch.no_grad():
            return (free_metric(dev,free,net,eck),
                    free_metric(replay,free,net,eck))
    base_dev,base_rep=metrics()
    print('BASE',round(base_dev*100,5),round(base_rep*100,5),
          'dev',len(dev),'replay',len(replay),flush=True)
    row=dev[0]
    name=_name(row)
    with torch.no_grad():
        got=rollout(states[name],net,eck,row['time'],row['voltage'],
                    float(cfg['shape_mix']))
        ref=deployed(plan['cases'][name],atlas,net,eck,row['time'],
                     row['voltage'],tan,init_override=cases[name],
                     shape_mix=float(cfg['shape_mix']))
    zerr=float(np.max(np.abs(got['z6'].numpy()-ref['z6'])))
    terr=float(np.max(np.abs(got['temp_group_K'].numpy()
                              -ref['temp_group_K'])))
    print('PARITY max_abs_Z',zerr,'max_abs_T_K',terr,flush=True)
    if zerr>1e-5 or terr>2e-3:
        raise RuntimeError('differentiable rollout differs from deployment')
    if a.check_only:
        row0=dict(row)
        for k in ('time','voltage','z6','y'):
            row0[k]=row[k][:min(4,len(row['time']))]
        net.train()
        grad_loss=loss([row0],states,net,eck,float(cfg['shape_mix']),
                       a.temp_weight)
        grad_loss.backward()
        grad={n:float(t.grad.norm()) if t.grad is not None else 0.
              for n,t in net.named_parameters() if t.requires_grad}
        print('GRAD',json.dumps(grad),flush=True)
        if not all(np.isfinite(v) and v>0 for v in grad.values()):
            raise RuntimeError('thermal head gradient missing')
        return
    opt=torch.optim.AdamW((t for t in net.parameters() if t.requires_grad),
                          lr=a.lr,weight_decay=1e-4)
    best_dev=base_dev
    best_rep=base_rep
    best_ep=0
    saved={k:t.detach().clone() for k,t in net.state_dict().items()}
    history=[]
    rng=np.random.default_rng(41)
    for ep in range(1,a.epochs+1):
        ids=rng.choice(len(tr),size=min(a.batch_hist,len(tr)),replace=False)
        batch=[tr[int(i)] for i in ids]
        net.train()
        opt.zero_grad(set_to_none=True)
        err=loss(batch,states,net,eck,float(cfg['shape_mix']),
                 a.temp_weight)
        if not torch.isfinite(err):
            raise RuntimeError(f'nonfinite BFZI-free loss at epoch {ep}')
        err.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
        opt.step()
        if ep%a.val_every:
            continue
        dev_err,rep_err=metrics()
        history.append(dict(epoch=ep,train_loss=float(err.detach()),
                            unseen_Z_pct=100*dev_err,replay_Z_pct=100*rep_err))
        if dev_err<best_dev-1e-6 and rep_err<=base_rep+a.replay_tol:
            best_dev,best_rep,best_ep=dev_err,rep_err,ep
            saved={k:t.detach().clone() for k,t in net.state_dict().items()}
        print('FREE_TRAIN',ep,'loss',round(float(err.detach()),6),
              'dev',round(dev_err*100,5),'replay',round(rep_err*100,5),
              'best',best_ep,flush=True)
        if ep-best_ep>=a.patience and ep>=a.patience:
            break
    net.load_state_dict(saved)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(ck,state=net.state_dict()),a.out)
    promote=best_ep>0 and base_dev-best_dev>=a.min_improve
    report=dict(status='accepted' if promote else 'rejected',
                best_epoch=best_ep,epochs_trained=ep,
                train_geometries=len(mf['train']),
                unseen_geometries=len({_name(r) for r in dev}),
                replay_geometries=len({_name(r) for r in replay}),
                train_histories=len(tr),unseen_histories=len(dev),
                replay_histories=len(replay),
                baseline_unseen_Z_pct=100*base_dev,
                best_unseen_Z_pct=100*best_dev,
                baseline_replay_Z_pct=100*base_rep,
                best_replay_Z_pct=100*best_rep,
                dev_improvement_pp=100*(base_dev-best_dev),
                promotion_min_improvement_pp=100*a.min_improve,
                forward_parity_Z=zerr,forward_parity_T_K=terr,
                shape_mix=float(cfg['shape_mix']),history=history,
                data_manifest=str(a.manifest),base_model=str(a.init_model))
    a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('FINAL',report['status'],best_ep,
          round(100*best_dev,5),round(100*best_rep,5),flush=True)


if __name__=='__main__':
    main()
