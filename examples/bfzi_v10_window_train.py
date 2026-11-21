"""V10 physically anchored, exposure-trained short-window SOE heat tuning.

Use complete BFZI histories; restore SOE memory from the true prefix, then
unroll 2/4/8/16 autonomous steps with differentiable physical circuit and
five-material heat balance. The frozen EM map is never trained against a
mismatched heat-state label.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_window import HeatWindow,zmat,pack


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_heat(path):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=48,
        soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    for name,p in net.named_parameters():
        p.requires_grad_(name.startswith("heat_head."))
    ec=dict(src_scale=ck["src_scale"],y_scale=ck["y_scale"],
            a_scale=ck["a_scale"],canonical=True,port_sea=True)
    return net,ck,ec


def loss_window(item,span,start,exposure,weights):
    """Only this window retains gradients; teacher prefix warms the SOE."""
    row=item["row"]
    win=item["win"]
    t=np.asarray(row["time"],float)
    v=np.asarray(row["voltage"],float)
    truth=torch.as_tensor(row["z6"],dtype=torch.float32)
    atrue=torch.as_tensor(row["a"],dtype=torch.float32)
    pred=torch.as_tensor(item["pred"],dtype=torch.float32)
    n=min(len(t),start+span+1)
    if n<=start+1:
        raise ValueError("short-window history requires at least one step")
    h=win.warm(row,start,item["pred_src"],item["pred"],exposure)
    with torch.no_grad():
        a0=atrue[start]+exposure*(pred[start]-atrue[start])
        a0=torch.cat((a0[:5].clamp_min(0),a0[5:]))
        old=a0*win.scale
        win.anchor(old)
        initial=old.detach()
    dtrue=truth-truth[0]
    scale_z=dtrue.square().sum(dim=1).mean().clamp_min(1e-10)
    current_scale=torch.as_tensor(item["i_scale"],dtype=torch.float32)
    power_scale=torch.as_tensor(item["p_scale"],dtype=torch.float32)
    sums=[]
    for k in range(start+1,n):
        dt=float(t[k]-t[k-1])
        h,old=win.step(h,old,dt,float(v[k-1]),float(t[k]))
        z,I,ps,pc=win.circuit(old,float(v[k]),float(t[k]))
        # At fixed time and drive, subtract response at the window's initial
        # state. This prevents thermal training from compensating cold-Z error.
        with torch.no_grad():
            base_z,base_i,base_ps,base_pc=win.circuit(
                initial,float(v[k]),float(t[k]))
        zp=pack(z)
        zb=pack(base_z)
        state=((old/win.scale-atrue[k])/win.sa).square().mean()
        skin=((old[32:]/win.scale[32:]-atrue[k,32:])/
              win.sa[32:]).square().mean()
        zd=((zp-zb)-(truth[k]-truth[start])).square().sum()/scale_z
        drive=torch.tensor([float(v[k]),0.],dtype=torch.complex64)
        ti=torch.linalg.solve(zmat(truth[k])+item["ref_ext"],drive)
        ti0=torch.linalg.solve(zmat(truth[start])+item["ref_ext"],drive)
        curr=((I-base_i)-(ti-ti0)).abs().square().sum()/current_scale
        # Match differential physical loss rather than the absolute cold bias.
        real_pow=item["power_coef"][k]*float(v[k])**2
        base_pow=item["power_coef"][start]*float(v[k])**2
        pow_err=(((ps-base_ps)-(real_pow[0]-base_pow[0]))/
                 power_scale[0]).square()
        pow_err+=(((pc[0]-base_pc[0])-(real_pow[1]-base_pow[1]))/
                  power_scale[1]).square()
        pow_err+=(((pc[1]-base_pc[1])-(real_pow[2]-base_pow[2]))/
                  power_scale[2]).square()
        sums.append(state+weights["skin"]*skin+weights["z"]*zd+
                    weights["current"]*curr+weights["power"]*pow_err)
    return torch.stack(sums).mean()


def evaluate(rows,atlas,net,ec,tan,par,mix):
    keys=("dynamic_z_rel_L2","R11_dynamic_rel_L2_self",
          "R22_dynamic_rel_L2_self","X11_dynamic_rel_L2_self",
          "X22_dynamic_rel_L2_self","current_rel_L2",
          "material_mean_rmse_K","sea_power_rel_L2",
          "copper_power_rel_L2")
    score=[]
    with torch.no_grad():
        for item in rows:
            r=item["row"]
            pred=rollout(item["case"],atlas,net,ec,r["time"],r["voltage"],
                         tan,init_override=item["init"],extra=par,
                         shape_mix=mix)
            score.append(metric(pred,r))
    return {k:float(np.mean([s[k] for s in score])) for k in keys}


def refresh(item,atlas,net,ec,tan,par,mix):
    """Current deployment trajectory for state and SOE source exposure."""
    row=item["row"]
    init=item["init"]
    with torch.no_grad():
        pred=rollout(item["case"],atlas,net,ec,row["time"],row["voltage"],
                    tan,init_override=init,extra=par,shape_mix=mix)
    item["pred"]=np.asarray(pred["heat_state"],np.float32)
    src=np.zeros((len(row["time"]),3),dtype=np.float32)
    src[1:,0]=np.asarray(pred["Psea_W"][:-1],np.float32)
    src[1:,1:]=(np.asarray(pred["Pcu_W"][:-1],np.float32)/
                 np.asarray(init["cap"][3:5],np.float32)[None])
    item["pred_src"]=src


def create_rows(specs,plan,atlas,cold,src,net,ec,tan,par,mix,train):
    out=[];names=set()
    for spec in specs:
        name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
        if name not in plan["cases"] or name in names:
            raise ValueError("missing or duplicate physical geometry "+name)
        names.add(name)
        case=plan["cases"][name]
        init=thermal_mix(atlas,src.apply(case,cold.apply(
            case,atlas.predict(case))),.25)
        win=HeatWindow(case,init,net,ec,tan,par,mix)
        for row in prepare(spec,48):
            item=dict(case=case,init=init,row=row,win=win)
            if train:
                refresh(item,atlas,net,ec,tan,par,mix)
                # The physical reference includes its own exact port/circuit.
                ztrue=np.asarray(row["z6"],float)
                vt=np.asarray(row["voltage"],float)
                port=row["rom"].port
                ext=row["rom"].ext
                currents=np.asarray([np.linalg.solve(
                    np.array([[z[0]+1j*z[1],z[2]+1j*z[3]],
                              [z[2]+1j*z[3],z[4]+1j*z[5]]])+ext,
                    np.array([v,0],complex))
                    for z,v in zip(ztrue,vt)])
                item["ref_ext"]=torch.as_tensor(ext,dtype=torch.complex64)
                item["i_scale"]=max(float(np.mean(abs(currents)**2)),1e-8)
                powers=[]
                coeff=[]
                for k,(z,i) in enumerate(zip(ztrue,currents)):
                    mat=np.array([[z[0]+1j*z[1],z[2]+1j*z[3]],
                                  [z[2]+1j*z[3],z[4]+1j*z[5]]])
                    cu=port.copper(row["temp_K"][k,row["rom"].cu])
                    sea=mat-1j*port.omega*port.air-np.diag(cu)
                    psea=max(float(.5*np.vdot(i,sea.real@i).real),0.)
                    pcu=.5*cu.real*abs(i)**2
                    powers.append([psea,*pcu])
                    iv=np.linalg.solve(mat+ext,np.array([1.,0.],complex))
                    upsea=max(float(.5*np.vdot(iv,sea.real@iv).real),0.)
                    upcu=.5*cu.real*abs(iv)**2
                    coeff.append([upsea,*upcu])
                item["power_coef"]=torch.as_tensor(coeff,dtype=torch.float32)
                item["p_scale"]=np.maximum(np.mean(powers,axis=0),1.)
            out.append(item)
    return out,names


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8_full_manifest.json"))
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_full_plan.json"))
    p.add_argument("--init-model",type=Path,default=Path("runs/geomfrac_et_v8_heat_tune.pt"))
    p.add_argument("--cold-model",type=Path,default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--refresh-every",type=int,default=4)
    p.add_argument("--epochs",type=int,default=16)
    p.add_argument("--updates",type=int,default=8)
    p.add_argument("--batch",type=int,default=3)
    p.add_argument("--val-every",type=int,default=4)
    p.add_argument("--lr",type=float,default=2e-5)
    p.add_argument("--skin-weight",type=float,default=.35)
    p.add_argument("--z-weight",type=float,default=.25)
    p.add_argument("--current-weight",type=float,default=.15)
    p.add_argument("--power-weight",type=float,default=.015)
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v10_window_heat.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v10_window_heat.json"))
    a=p.parse_args()
    if min(a.epochs,a.updates,a.batch,a.val_every,a.refresh_every)<=0:
        p.error("positive epochs, batch, updates and validation interval required")
    torch.manual_seed(19437)
    torch.set_num_threads(2)
    rng=np.random.default_rng(19437)
    plan=read(a.plan)
    mf=read(a.manifest)
    cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
    net,ck,ec=load_heat(a.init_model)
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load(a.cold_model)
    src=SrcFix.load(cfg["source_model"])
    tan=restore_rank48("runs/geomfrac_rank48_skin_tan_v3.pt",
                       "runs/geomfrac_et_v9_ztune.pt")
    par=ParZ48.load("runs/geomfrac_v8_par48.pt")
    for param in par.parameters():
        param.requires_grad_(False)
    mix=float(cfg["shape_mix"])
    net.eval()
    tr,tn=create_rows(mf["train"],plan,atlas,cold,src,net,ec,tan,par,mix,True)
    va,vn=create_rows(mf["val"],plan,atlas,cold,src,net,ec,tan,par,mix,False)
    if (tn & vn or len(tn)!=48 or len(vn)!=15 or len(tr)!=132 or len(va)!=41):
        raise ValueError("require frozen geometry-disjoint 48/15 and 132/41 histories")
    baseline=evaluate(va,atlas,net,ec,tan,par,mix)
    print("V10_BASE",json.dumps(baseline),flush=True)
    baseline_state=copy.deepcopy(net.state_dict())
    best_state=baseline_state
    best=baseline
    best_epoch=0
    weights=dict(skin=a.skin_weight,z=a.z_weight,
                 current=a.current_weight,power=a.power_weight)
    opt=torch.optim.AdamW((x for x in net.parameters() if x.requires_grad),
                          lr=a.lr,weight_decay=1e-4)
    history=[]
    for ep in range(1,a.epochs+1):
        if ep>1 and (ep-1)%a.refresh_every==0:
            net.eval()
            for item in tr:
                refresh(item,atlas,net,ec,tan,par,mix)
            print("V10_REFRESH",ep,len(tr),flush=True)
        span=(2,4,8,16)[min(3,(ep-1)*4//a.epochs)]
        exposure=min(.7,.15+.5*(ep-1)/max(1,a.epochs-1))
        net.train()
        losses=[]
        for _ in range(a.updates):
            ix=rng.choice(len(tr),size=a.batch,replace=False)
            sample=[]
            for j in ix:
                item=tr[j]
                n=len(item["row"]["time"])-1
                steps=min(span,n)
                start=int(rng.integers(0,n-steps+1))
                sample.append(loss_window(item,steps,start,exposure,weights))
            loss=torch.stack(sample).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite differentiable rollout loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (x for x in net.parameters() if x.requires_grad),1.)
            opt.step()
            losses.append(float(loss.detach()))
        if ep%a.val_every and ep!=a.epochs:
            continue
        net.eval()
        scores=evaluate(va,atlas,net,ec,tan,par,mix)
        row=dict(epoch=ep,span=span,exposure=exposure,
                 training_loss=float(np.mean(losses)),**scores)
        history.append(row)
        print("V10_VAL",json.dumps(row),flush=True)
        improve=(scores["dynamic_z_rel_L2"]<best["dynamic_z_rel_L2"]-.001
                 and scores["R11_dynamic_rel_L2_self"]<
                     baseline["R11_dynamic_rel_L2_self"]*1.02
                 and scores["R22_dynamic_rel_L2_self"]<
                     baseline["R22_dynamic_rel_L2_self"]*1.02
                 and scores["current_rel_L2"]<
                     baseline["current_rel_L2"]*1.02
                 and scores["material_mean_rmse_K"]<
                     baseline["material_mean_rmse_K"]*1.03)
        if improve:
            best=scores
            best_epoch=ep
            best_state=copy.deepcopy(net.state_dict())
    if best_epoch:
        saved=dict(ck)
        saved["state"]=best_state
        a.out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(saved,a.out)
    result=dict(status="development_candidate" if best_epoch else "not_promoted",
        baseline=baseline,best=best,best_epoch=best_epoch,
        checkpoint=str(a.out) if best_epoch else None,
        history=history,windows=[2,4,8,16],
        train_geometries=len(tn),train_histories=len(tr),
        val_geometries=len(vn),val_histories=len(va),
        training="frozen ColdNet/physical EM; teacher SOE prefix and differentiable window",
        weights=weights,init_model=str(a.init_model),
        cold_model=str(a.cold_model),refresh_every=a.refresh_every,
        blind_not_used=True)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V10_RESULT",json.dumps({k:result[k] for k in
          ("status","best_epoch","baseline","best")}),flush=True)


if __name__=="__main__":
    main()
