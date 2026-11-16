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
    head=GeomHeat.load(str(spec['anchor']).replace('\\', '/'))
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
    p.add_argument('--reuse-scales',action='store_true',
                   help='keep source/output/state scales when fine-tuning')
    p.add_argument('--heat-z',action='store_true',
                   help='condition seawater impedance directly on thermal ROM state')
    p.add_argument('--head-only',action='store_true',
                   help='only train the new heat-conditioned impedance head')
    p.add_argument('--heat-only',action='store_true',
                   help='freeze SOE impedance and train thermal head only')
    p.add_argument('--tangent-model',type=Path,default=None,
                   help='evaluate autonomous Z with pretrained geometry tangent')
    p.add_argument('--tangent-loss-weight',type=float,default=0.,
                   help='supervise thermal state via tangent-projected impedance')
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_et_multi.pt'))
    p.add_argument('--report',type=Path,default=Path('runs/geomfrac_et_multi.json'))
    p.add_argument('--rank',type=int,default=32,choices=(5,32))
    p.add_argument('--epochs',type=int,default=500)
    p.add_argument('--patience',type=int,default=90)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--lr-patience',type=int,default=20)
    p.add_argument('--lr-factor',type=float,default=.5)
    p.add_argument('--soe-quad',choices=('legacy','trap'),default='trap')
    p.add_argument('--heat-dt',type=float,default=1.875)
    p.add_argument('--closed-weight',type=float,default=.2)
    p.add_argument('--closed-warmup',type=int,default=20)
    p.add_argument('--closed-ramp',type=int,default=60)
    p.add_argument('--therm-weight',type=float,default=.03)
    p.add_argument('--zrel-weight',type=float,default=0.,
                   help='full dynamic port-Z relative squared loss weight')
    p.add_argument('--history-weight',choices=('equal','step'),default='step')
    p.add_argument('--batch-hist',type=int,default=0,
                   help='histories per stochastic step; 0 uses full training set')
    p.add_argument('--val-every',type=int,default=1,
                   help='validate every N training epochs')
    p.add_argument('--select',choices=('loss','dynamic_z'),default='loss',
                   help='best-checkpoint criterion for validation')
    p.add_argument('--proj-tau',type=float,default=0.,
                   help='projection quality scale in K for history sampling; 0=uniform')
    a=p.parse_args()
    config=json.loads(a.manifest.read_text(encoding='utf-8-sig'))
    trspec,vaspec=config['train'],config['val']
    trgeo={str(item['anchor']).replace('\\','/') for item in trspec}
    vageo={str(item['anchor']).replace('\\','/') for item in vaspec}
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
    net=GeomFracET(ng=dims.pop(),rank=a.rank,
                   soe_quad=a.soe_quad,heat_dt=a.heat_dt,heat_z=a.heat_z)
    if a.init_model:
        old=torch.load(a.init_model,map_location='cpu',weights_only=True)
        load_init(net,old)
        if a.reuse_scales:
            sx=np.asarray(old['src_scale'],dtype=float)
            sy=np.asarray(old['y_scale'],dtype=float)
            sa=np.asarray(old['a_scale'],dtype=float)
    elif a.reuse_scales:
        p.error('--reuse-scales requires --init-model')
    if a.head_only and a.heat_only:
        p.error('--head-only and --heat-only are exclusive')
    if a.head_only:
        if not a.heat_z or not a.init_model:
            p.error('--head-only requires --heat-z and --init-model')
        for name,param in net.named_parameters():
            param.requires_grad_(name.startswith('heat_z_head.'))
    if a.heat_only:
        for name,param in net.named_parameters():
            param.requires_grad_(name.startswith('heat_head.'))
    tangent=None
    if a.tangent_model:
        from examples.bfzi_z_tangent import restore
        tangent=restore(a.tangent_model)
    if a.tangent_loss_weight<0 or (a.tangent_loss_weight and tangent is None):
        p.error('nonnegative tangent loss requires --tangent-model')
    opt=torch.optim.AdamW((v for v in net.parameters() if v.requires_grad),lr=a.lr,
                          weight_decay=a.weight_decay)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode='min',factor=a.lr_factor,
        patience=max(1,(a.lr_patience+a.val_every-1)//a.val_every))
    best=float('inf')
    best_epoch=0
    saved=None
    best_train=None
    best_loss=None
    eck=dict(src_scale=sx,y_scale=sy,a_scale=sa,
             canonical=True,port_sea=True)
    if a.select=='dynamic_z' and a.init_model:
        net.eval()
        with torch.no_grad():
            init_z=eval_rollout(None,net,eck,val,tangent=tangent)
        best=float(np.mean([x['dynamic_Z_rel_L2'] for x in init_z]))
        saved={k:v.detach().clone() for k,v in net.state_dict().items()}
        print('INIT dynamic_z',round(best,6),flush=True)
    if a.val_every<1 or a.batch_hist<0 or a.proj_tau<0:
        p.error('invalid validation interval, batch size or projection scale')
    if a.proj_tau:
        prob=torch.as_tensor([
            1./(1.+(r['basis_max_K']/a.proj_tau)**2)
            for r in train],dtype=torch.float32)
        print('PROJ_SAMPLING',a.proj_tau,
              'min',round(float(prob.min()),5),
              'max',round(float(prob.max()),5),flush=True)
    if a.closed_warmup<0 or a.closed_ramp<=0:
        p.error('closed warmup/ramp must be nonnegative/positive')
    warmup=(a.closed_warmup+a.closed_ramp) if a.closed_weight else 0
    for ep in range(a.epochs):
        if a.batch_hist and a.batch_hist<len(train):
            if a.proj_tau:
                ids=torch.multinomial(prob,a.batch_hist,replacement=False).tolist()
            else:
                ids=torch.randperm(len(train))[:a.batch_hist].tolist()
            batch=[train[i] for i in ids]
        else:
            batch=train
        net.train()
        opt.zero_grad(set_to_none=True)
        loss=score(net,batch,None,sx,sy,sa,therm_w=a.therm_weight,
                   zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight)
        if a.closed_weight and ep>=a.closed_warmup:
            mix=min(1.,(ep+1-a.closed_warmup)/a.closed_ramp)
            loss+=a.closed_weight*score(net,batch,None,sx,sy,sa,
                        closed=True,therm_w=a.therm_weight,src_mix=mix,
                        zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"training diverged at epoch {ep+1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),3.)
        opt.step()
        if (ep+1)%a.val_every and ep+1!=a.epochs:
            continue
        net.eval()
        with torch.no_grad():
            v=float(score(net,val,None,sx,sy,sa,therm_w=a.therm_weight,
                          zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight))
            if a.closed_weight:
                v+=a.closed_weight*float(score(net,val,None,sx,sy,sa,
                        closed=True,therm_w=a.therm_weight,src_mix=1.,
                        zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight))
        if not np.isfinite(v):
            raise RuntimeError(f"validation diverged at epoch {ep+1}")
        if a.select=='dynamic_z':
            with torch.no_grad():
                zval=eval_rollout(None,net,eck,val,tangent=tangent)
            metric=float(np.mean([x['dynamic_Z_rel_L2'] for x in zval]))
        else:
            metric=v
        if not np.isfinite(metric):
            raise RuntimeError('nonfinite validation selection metric')
        if ep+1>=warmup:
            sched.step(metric)
        if ep+1>=warmup and metric<best-1e-8:
            best,best_epoch=metric,ep+1
            best_loss=v
            best_train=float(loss.detach())
            saved={k:vv.detach().clone() for k,vv in net.state_dict().items()}
        if (ep+1)%50==0:
            print("TRAIN",ep+1,"train",round(float(loss.detach()),6),
                  "val",round(v,6),"select",round(metric,6),
                  "best",best_epoch,flush=True)
        if saved is not None and ep+1-best_epoch>=a.patience:
            break
    if saved is None:
        raise RuntimeError("training requires enough epochs for closed-loop source warmup")
    net.load_state_dict(saved)
    if best_loss is None:
        net.eval()
        with torch.no_grad():
            best_loss=float(score(net,val,None,sx,sy,sa,
                therm_w=a.therm_weight,zrel_w=a.zrel_weight,
                hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight))
            if a.closed_weight:
                best_loss+=a.closed_weight*float(score(net,val,None,
                    sx,sy,sa,closed=True,therm_w=a.therm_weight,
                    src_mix=1.,zrel_w=a.zrel_weight,
                    hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight))
    ck=dict(state=net.state_dict(),ng=net.ng,rank=net.rank,
            src_scale=torch.as_tensor(sx,dtype=torch.float32),
            y_scale=torch.as_tensor(sy,dtype=torch.float32),
            a_scale=torch.as_tensor(sa,dtype=torch.float32),
            canonical=True,port_sea=True,architecture='GeomFracET',
            heat_z=a.heat_z,soe_quad=a.soe_quad,heat_dt=a.heat_dt,
            drive_rule="left",
            train_files=[r['file'] for r in train],
            val_files=[r['file'] for r in val])
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck,a.out)
    autonomous=eval_rollout(None,net,ck,val,tangent=tangent)
    metrics=dict(architecture='GeomFracET-cross-geometry',
                 mode='canonical-material-32',rank=net.rank,
                 soe_quad=a.soe_quad,heat_dt=a.heat_dt,
                 autonomous_src=True,zrel_weight=a.zrel_weight,
                 reuse_scales=a.reuse_scales,proj_tau=a.proj_tau,
                 heat_z=a.heat_z,head_only=a.head_only,
                 heat_only=a.heat_only,
                 tangent_model=str(a.tangent_model) if a.tangent_model else None,
                 tangent_loss_weight=a.tangent_loss_weight,
                 history_weight=a.history_weight,
                 optimizer='AdamW',weight_decay=a.weight_decay,
                 batch_hist=a.batch_hist,val_every=a.val_every,
                 lr_factor=a.lr_factor,lr_patience=a.lr_patience,
                 final_lr=opt.param_groups[0]['lr'],
                 ntrain_geometries=len(trgeo),nval_geometries=len(vageo),
                 train_histories=len(train),val_histories=len(val),
                 best_epoch=best_epoch,val_loss=best_loss if a.select=='dynamic_z' else best,
                 select=a.select,best_score=best,
                 val_dynamic_Z_mean_rel_L2=float(np.mean(
                     [v['dynamic_Z_rel_L2'] for v in autonomous])),
                 train_loss_at_best=best_train,epochs_trained=ep+1,
                 closed_source_warmup=warmup,
                 closed_warmup=a.closed_warmup,
                 closed_ramp=a.closed_ramp,
                 train_projection_max_K=max(r['basis_max_K'] for r in train),
                 val_projection_max_K=max(r['basis_max_K'] for r in val),
                 autonomous_validation=autonomous)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__':
    main()
