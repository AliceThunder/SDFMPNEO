"""Train one GeomFracET across independent BFZI geometries in shared canonical 32/48-mode heat space.

Manifest JSON:
{"train":[{"anchor":"runs/T01_anchor.npz","teacher":["runs/T01_a.npz"]}],
 "val":[{"anchor":"runs/T02_anchor.npz","teacher":["runs/T02_b.npz"]}]}

Every geometry receives the same five material-mean coordinates and rank-5
spatial coordinates. The FV operators and true spatial reconstructions remain
local.
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


def free_metric(rows,free,net,ck,heat_op="atlas",cu_link=None):
    """Select with the same BFZI-free closed loop used for deployment."""
    from sdfmpneo_bfzi.geom_free import rollout
    plan,atlas,ps,mu,src,tan,mix=free
    cache={}
    errs=[]
    for r in rows:
        name=Path(r['file'].split('#')[0]).stem.split('_')[2]
        case=plan['cases'][name]
        if name not in cache:
            init=atlas.predict(case)
            for model in (ps,mu,src):
                if model is not None:
                    init=model.apply(case,init)
            if heat_op=="conductance":
                from sdfmpneo_bfzi.geom_init import conductance_mix
                init=conductance_mix(atlas,init)
            if heat_op=="contact":
                from sdfmpneo_bfzi.geom_cu import apply as cu_apply
                init=cu_apply(case,init,exchange=True,
                              profile=cu_link.predict(case))
            cache[name]=init
        pred=rollout(case,atlas,net,ck,r['time'],r['voltage'],tan,
                     init_override=cache[name],shape_mix=mix)['z6']
        dz=pred-pred[0]
        true=r['z6']-r['z6'][0]
        errs.append(np.linalg.norm(dz-true)/max(np.linalg.norm(true),1e-12))
    return float(np.mean(errs))


def rank48_extra(net):
    """Train only the appended 16 skin-mode outputs; preserve the original 32."""
    for param in net.parameters():
        param.requires_grad_(False)
    weight=net.heat_head[2].weight
    bias=net.heat_head[2].bias
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    wm=torch.zeros_like(weight)
    bm=torch.zeros_like(bias)
    wm[32:]=1.
    bm[32:]=1.
    weight.register_hook(lambda grad,mask=wm:grad*mask)
    bias.register_hook(lambda grad,mask=bm:grad*mask)


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
    p.add_argument('--rank48-extra-only',action='store_true',help='train only appended 16 rank48 skin-mode outputs')
    p.add_argument('--tangent-model',type=Path,default=None,
                   help='evaluate autonomous Z with pretrained geometry tangent')
    p.add_argument('--tangent-loss-weight',type=float,default=0.,
                   help='supervise thermal state via tangent-projected impedance')
    p.add_argument('--out',type=Path,default=Path('runs/geomfrac_et_multi.pt'))
    p.add_argument('--report',type=Path,default=Path('runs/geomfrac_et_multi.json'))
    p.add_argument('--rank',type=int,default=32,choices=(5,32,48))
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
    p.add_argument('--cu-z-weight',type=float,default=0.,help='copper-only physically weighted dynamic impedance loss')
    p.add_argument('--zrel-weight',type=float,default=0.,
                   help='full dynamic port-Z relative squared loss weight')
    p.add_argument('--history-weight',choices=('equal','step'),default='step')
    p.add_argument('--batch-hist',type=int,default=0,
                   help='histories per stochastic step; 0 uses full training set')
    p.add_argument('--val-every',type=int,default=1,
                   help='validate every N training epochs')
    p.add_argument('--select',choices=('loss','dynamic_z','thermal','free_dynamic_z'),default='loss',
                   help='best-checkpoint criterion for validation')
    p.add_argument('--proj-tau',type=float,default=0.,
                   help='projection quality scale in K for history sampling; 0=uniform')
    p.add_argument('--free-plan',type=Path,default=None,
                   help='geometry case plan for BFZI-free development selection')
    p.add_argument('--free-config',type=Path,default=Path('runs/geomfrac_free_v7_selection.json'),
                   help='frozen geometry-free port/thermal model configuration')
    p.add_argument('--free-heat-op',choices=('atlas','conductance','contact'),default='atlas',help='geometry-only heat operator used for closed-loop checkpoint selection')
    p.add_argument('--cu-link-model',type=Path,default=None,help='frozen geometry-conditioned package contact-mode coefficients')
    a=p.parse_args()
    if a.cu_z_weight<0:
        p.error('copper impedance weight must be nonnegative')
    if a.free_heat_op=='contact' and not a.cu_link_model:
        p.error('contact-based heat operator requires --cu-link-model')
    cu_link=None
    if a.free_heat_op=='contact':
        from sdfmpneo_bfzi.geom_culink import CuLink
        cu_link=CuLink.load(a.cu_link_model)
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
            sa_old=np.asarray(old['a_scale'],dtype=float)
            if len(sa_old)<a.rank:
                sa=np.r_[sa_old,sa[len(sa_old):]]
            else:
                sa=sa_old
    elif a.reuse_scales:
        p.error('--reuse-scales requires --init-model')
    if a.rank48_extra_only and (a.rank!=48 or not a.init_model or a.heat_only or a.head_only):
        p.error('--rank48-extra-only requires rank48 and an init model, without other freeze modes')
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
        tck=torch.load(a.tangent_model,map_location='cpu',weights_only=True)
        if 'base' in tck and 'sy' in tck:
            from sdfmpneo_bfzi.geom_ztune import restore
        else:
            from examples.bfzi_z_tangent import restore
        tangent=restore(a.tangent_model)
    if a.tangent_loss_weight<0 or (a.tangent_loss_weight and tangent is None):
        p.error('nonnegative tangent loss requires --tangent-model')
    free=None
    if a.select=='free_dynamic_z':
        if not a.free_plan:
            p.error('--select free_dynamic_z requires --free-plan')
        plan=json.loads(a.free_plan.read_text(encoding='utf-8-sig'))
        cfg=json.loads(a.free_config.read_text(encoding='utf-8-sig'))
        if plan.get('invalid_for_physics'):
            p.error('invalid BFZI-free selection plan')
        from sdfmpneo_bfzi.geom_init import GeomInit
        from sdfmpneo_bfzi.geom_portself import PortSelf
        from sdfmpneo_bfzi.geom_mutual import MutFix
        from sdfmpneo_bfzi.geom_srcfix import SrcFix
        from sdfmpneo_bfzi.geom_ztune import restore as load_z
        if any(Path(r['file'].split('#')[0]).stem.split('_')[2]
               not in plan['cases'] for r in val):
            p.error('development geometry missing from BFZI-free selection plan')
        free=(plan,GeomInit.load(cfg['atlas']),
              PortSelf.load(cfg['port_self_model']),
              MutFix.load(cfg['mutual_model']),
              SrcFix.load(cfg['source_model']),
              load_z(cfg['z_fix_model']),float(cfg['shape_mix']))
    opt=torch.optim.AdamW((v for v in net.parameters() if v.requires_grad),lr=a.lr,
                          weight_decay=(0. if a.rank48_extra_only else a.weight_decay))
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode='min',factor=a.lr_factor,
        patience=max(1,(a.lr_patience+a.val_every-1)//a.val_every))
    best=float('inf')
    best_epoch=0
    initial_score=None
    saved=None
    best_train=None
    best_loss=None
    eck=dict(src_scale=sx,y_scale=sy,a_scale=sa,
             canonical=True,port_sea=True)
    if a.select in ('dynamic_z','thermal','free_dynamic_z') and a.init_model:
        net.eval()
        with torch.no_grad():
            if a.select=='free_dynamic_z':
                best=free_metric(val,free,net,eck,a.free_heat_op,cu_link)
            else:
                init_z=eval_rollout(None,net,eck,val,tangent=tangent)
                key='dynamic_Z_rel_L2' if a.select=='dynamic_z' else 'temperature_max_K'
                best=float(np.mean([x[key] for x in init_z]))
        initial_score=float(best)
        saved={k:v.detach().clone() for k,v in net.state_dict().items()}
        print('INIT',a.select,round(best,6),flush=True)
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
                   zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight, cu_z_w=a.cu_z_weight)
        if a.closed_weight and ep>=a.closed_warmup:
            mix=min(1.,(ep+1-a.closed_warmup)/a.closed_ramp)
            loss+=a.closed_weight*score(net,batch,None,sx,sy,sa,
                        closed=True,therm_w=a.therm_weight,src_mix=mix,
                        zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight, cu_z_w=a.cu_z_weight)
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
                          zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight, cu_z_w=a.cu_z_weight))
            if a.closed_weight:
                v+=a.closed_weight*float(score(net,val,None,sx,sy,sa,
                        closed=True,therm_w=a.therm_weight,src_mix=1.,
                        zrel_w=a.zrel_weight,hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight, cu_z_w=a.cu_z_weight))
        if not np.isfinite(v):
            raise RuntimeError(f"validation diverged at epoch {ep+1}")
        if a.select=='free_dynamic_z':
            with torch.no_grad():
                metric=free_metric(val,free,net,eck,a.free_heat_op,cu_link)
        elif a.select in ('dynamic_z','thermal'):
            with torch.no_grad():
                zval=eval_rollout(None,net,eck,val,tangent=tangent)
            key='dynamic_Z_rel_L2' if a.select=='dynamic_z' else 'temperature_max_K'
            metric=float(np.mean([x[key] for x in zval]))
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
        if saved is not None and ep+1>=warmup and ep+1-max(best_epoch,warmup)>=a.patience:
            break
    if saved is None:
        raise RuntimeError("training requires enough epochs for closed-loop source warmup")
    net.load_state_dict(saved)
    if best_loss is None:
        net.eval()
        with torch.no_grad():
            best_loss=float(score(net,val,None,sx,sy,sa,
                therm_w=a.therm_weight,zrel_w=a.zrel_weight,
                hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight, cu_z_w=a.cu_z_weight))
            if a.closed_weight:
                best_loss+=a.closed_weight*float(score(net,val,None,
                    sx,sy,sa,closed=True,therm_w=a.therm_weight,
                    src_mix=1.,zrel_w=a.zrel_weight,
                    hist_weight=a.history_weight, tangent=tangent, tan_w=a.tangent_loss_weight, cu_z_w=a.cu_z_weight))
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
                 mode=f'canonical-material-{net.rank}',rank=net.rank,
                 soe_quad=a.soe_quad,heat_dt=a.heat_dt,
                 autonomous_src=True,zrel_weight=a.zrel_weight,
                 reuse_scales=a.reuse_scales,proj_tau=a.proj_tau,
                 heat_z=a.heat_z,head_only=a.head_only,
                 heat_only=a.heat_only,rank48_extra_only=a.rank48_extra_only,
                 copper_z_loss_weight=a.cu_z_weight,
                 free_heat_op=a.free_heat_op,
                 tangent_model=str(a.tangent_model) if a.tangent_model else None,
                 tangent_loss_weight=a.tangent_loss_weight,
                 history_weight=a.history_weight,
                 optimizer='AdamW',weight_decay=a.weight_decay,
                 batch_hist=a.batch_hist,val_every=a.val_every,
                 lr_factor=a.lr_factor,lr_patience=a.lr_patience,
                 final_lr=opt.param_groups[0]['lr'],
                 ntrain_geometries=len(trgeo),nval_geometries=len(vageo),
                 train_histories=len(train),val_histories=len(val),
                 best_epoch=best_epoch,val_loss=best_loss if a.select!='loss' else best,
                 select=a.select,best_score=best,
                 initial_selection_score=initial_score,
                 selection_improvement_pct=(100.*(initial_score-best)/max(abs(initial_score),1e-12) if initial_score is not None else None),
                 free_plan=str(a.free_plan) if a.free_plan else None,
                 free_config=str(a.free_config) if free is not None else None,
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
