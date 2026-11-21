"""Train V9 EM temperature map from complete BFZI histories and rollout states."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_canon import labels
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import geometry,rollout
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_skintan import SkinTanV3
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_base


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def avg(rows,key):
    return float(np.mean([r[key] for r in rows]))


def phys_delta(item,truth,pred,phy):
    """BFZI-constrained local impedance change when thermal input changes."""
    init=item["init"]
    src=np.asarray(init["src"],float)
    scale=np.sqrt(np.asarray(init["cap"],float)[labels(48)])
    gain=np.asarray(phy["gain"],float)
    react=np.asarray(phy["react_gain"],float)
    def calc(a):
        x=a*scale[None]
        parts=np.stack((x[:,:1]@src[:1],
                        x[:,1:32]@src[1:32],
                        x[:,32:48]@src[32:48]),axis=1)
        real=(parts*gain[None]).sum(axis=1)
        imag=(parts*react[None]).sum(axis=1)
        z=np.empty((len(a),6),float)
        z[:,0::2]=real
        z[:,1::2]=imag
        return float(item["case"]["alpha_per_K"])*z
    return calc(pred)-calc(truth)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v8_full_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v8_full_manifest.json"))
    p.add_argument("--heat-model",type=Path,default=Path("runs/geomfrac_et_v8_heat_tune.pt"))
    p.add_argument("--skin-model",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v3.pt"))
    p.add_argument("--skin-base",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--parallel",type=Path,default=Path("runs/geomfrac_v8_par48.pt"))
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--lr",type=float,default=0.00006)
    p.add_argument("--pred-weight",type=float,default=0.5)
    p.add_argument("--prior",type=float,default=.15)
    p.add_argument("--val-every",type=int,default=5)
    p.add_argument("--phys-gain",type=Path,
                   default=Path("runs/geomfrac_v9_phys_region.json"))
    p.add_argument("--phys-weight",type=float,default=.5,
                   help="physically consistent label shift for predicted states")
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v9_full_skin.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v9_full_skin.json"))
    args=p.parse_args()
    if not 0<=args.phys_weight<=1:
        p.error("physical label shift weight must be in [0,1]")
    phy=read(args.phys_gain) if args.phys_weight else None
    if phy is not None and ("gain" not in phy or "react_gain" not in phy):
        p.error("physical calibration must provide real/imaginary susceptibility")

    torch.manual_seed(9227)
    np.random.seed(9227)
    torch.set_num_threads(2)
    cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
    plan=read(args.plan)
    mf=read(args.manifest)
    if len(mf["train"])!=48 or len(mf["val"])!=15:
        raise ValueError("V9 expects 48/15 geometry split")

    ck=torch.load(args.heat_model,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=48,soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    net.eval()
    for par in net.parameters():
        par.requires_grad_(False)
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
    src=SrcFix.load(cfg["source_model"])
    par=ParZ48.load(args.parallel)
    base=load_base(args.skin_base)
    sc=torch.load(args.skin_model,map_location="cpu",weights_only=True)
    skin=SkinTanV3(ng=int(sc["ng"]),sa=np.asarray(sc["sa"]),ss=np.asarray(sc["ss"]))
    skin.load_state_dict(sc["state"])
    original=copy.deepcopy(skin).eval()
    sy=torch.tensor(np.asarray(sc["sy"])[:6],dtype=torch.float32)
    mix=float(cfg["shape_mix"])
    norm=np.asarray(ck["a_scale"],float)

    def mk_tan(s):
        def predict(g,a):
            return base(g,a[:,:32])+s(g,a[:,:32],a[:,32:])*sy
        predict.time_conditioned=False
        return predict

    old_tan=mk_tan(original)
    def collect(split):
        out=[];keys=set()
        for spec in mf[split]:
            name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
            if name in keys or name not in plan["cases"]:
                raise ValueError("duplicate or missing geometry "+name)
            keys.add(name)
            case=plan["cases"][name]
            init=thermal_mix(atlas,src.apply(case,cold.apply(
                case,atlas.predict(case))),.25)
            g=geometry(case,init)[1].astype(np.float32)
            for row in prepare(spec,48):
                out.append(dict(name=name,case=case,init=init,g=g,row=row))
        return out,keys

    train,train_names=collect("train")
    val,val_names=collect("val")
    if train_names & val_names:
        raise ValueError("geometry leakage across splits")
    if len(train)!=132 or len(val)!=41:
        raise ValueError("incomplete full BFZI train/val history set")

    def measure(s):
        tan=mk_tan(s)
        scores=[]
        with torch.no_grad():
            for item in val:
                r=item["row"]
                pred=rollout(item["case"],atlas,net,ck,r["time"],r["voltage"],
                    tan,init_override=item["init"],extra=par,shape_mix=mix)
                scores.append(metric(pred,r))
        return {k:avg(scores,k) for k in (
            "dynamic_z_rel_L2","R11_dynamic_rel_L2_self",
            "R22_dynamic_rel_L2_self","X11_dynamic_rel_L2_self",
            "X22_dynamic_rel_L2_self","current_rel_L2",
            "material_mean_rmse_K","sea_power_rel_L2",
            "copper_power_rel_L2")}

    print("V9_FULL_COLLECT",len(train),len(val),flush=True)
    base_score=measure(original)
    print("V9_FULL_BASE",json.dumps(base_score),flush=True)

    gs=[];states=[];times=[];volts=[];truth=[];weights=[]
    for i,item in enumerate(train):
        row=item["row"]
        with torch.no_grad():
            pred=rollout(item["case"],atlas,net,ck,
                row["time"],row["voltage"],old_tan,
                init_override=item["init"],extra=par,shape_mix=mix)
        n=len(row["time"])-1
        if n<1:
            continue
        g=np.broadcast_to(item["g"],(n,45))
        z=np.asarray(row["y"][1:,:6],np.float32)
        t=np.asarray(row["time"][1:,None],np.float32)
        v=np.asarray(row["voltage"][1:,None],np.float32)
        true=np.asarray(row["a"][1:],np.float32)
        est=np.asarray(pred["heat_state"][1:],np.float32)
        if est.shape!=true.shape or not np.isfinite(est).all():
            raise ValueError("autonomous heat states invalid")
        for a,w in ((true,1.-args.pred_weight),(est,args.pred_weight)):
            if w<=0:
                continue
            gs.append(g.copy());states.append(a)
            target=z if a is true or phy is None else (
                z+args.phys_weight*phys_delta(item,true,est,phy).astype(np.float32))
            times.append(t);volts.append(v);truth.append(target)
            weights.extend([w/n]*n)
        if (i+1)%24==0:
            print("V9_FULL_HISTORY",i+1,len(train),flush=True)

    g=torch.as_tensor(np.concatenate(gs),dtype=torch.float32)
    a=torch.as_tensor(np.concatenate(states),dtype=torch.float32)
    t=torch.as_tensor(np.concatenate(times),dtype=torch.float32)
    v=torch.as_tensor(np.concatenate(volts),dtype=torch.float32)
    y=torch.as_tensor(np.concatenate(truth),dtype=torch.float32)
    w=torch.as_tensor(np.asarray(weights),dtype=torch.float32)
    w=w/w.mean()
    with torch.no_grad():
        yy=(y-base(g,a[:,:32])-par(g,a,t,v))/sy
        prior=original(g,a[:,:32],a[:,32:])
    opt=torch.optim.AdamW(skin.parameters(),lr=args.lr,weight_decay=.002)
    best=base_score;best_ep=0
    best_state=copy.deepcopy(skin.state_dict())
    history=[]
    for ep in range(1,args.epochs+1):
        skin.train()
        order=torch.randperm(len(g))
        losses=[]
        for inds in order.split(256):
            opt.zero_grad(set_to_none=True)
            z=skin(g[inds],a[inds,:32],a[inds,32:])
            loss=(w[inds,None]*(z-yy[inds]).square()).mean()
            regular=(w[inds,None]*(z-prior[inds]).square()).mean()
            loss=loss+args.prior*regular
            loss.backward()
            torch.nn.utils.clip_grad_norm_(skin.parameters(),1.)
            opt.step()
            losses.append(float(loss.detach()))
        if ep%args.val_every and ep!=args.epochs:
            continue
        skin.eval()
        scv=measure(skin)
        entry=dict(epoch=ep,loss=float(np.mean(losses)),**scv)
        history.append(entry)
        print("V9_FULL_VAL",json.dumps(entry),flush=True)
        improved=(scv["dynamic_z_rel_L2"]<best["dynamic_z_rel_L2"]-0.001
            and scv["R11_dynamic_rel_L2_self"]<=base_score["R11_dynamic_rel_L2_self"]*1.01
            and scv["R22_dynamic_rel_L2_self"]<=base_score["R22_dynamic_rel_L2_self"]*1.01
            and scv["X11_dynamic_rel_L2_self"]<=base_score["X11_dynamic_rel_L2_self"]*1.02
            and scv["X22_dynamic_rel_L2_self"]<=base_score["X22_dynamic_rel_L2_self"]*1.02
            and scv["material_mean_rmse_K"]<=base_score["material_mean_rmse_K"]*1.04
            and scv["current_rel_L2"]<=base_score["current_rel_L2"]*1.04)
        if improved:
            best=scv;best_ep=ep
            best_state=copy.deepcopy(skin.state_dict())
    if best_ep:
        saved=dict(sc)
        saved["state"]=best_state
        args.out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(saved,args.out)
    result=dict(status="candidate" if best_ep else "not_promoted",
        base=base_score,best=best,best_epoch=best_ep,history=history,
        train_histories=len(train),val_histories=len(val),
        train_geometries=len(train_names),val_geometries=len(val_names),
        paired_states=len(g),pred_weight=args.pred_weight,
        phys_weight=args.phys_weight,phys_gain=str(args.phys_gain),
        heat_model=str(args.heat_model),skin_model=str(args.skin_model),
        checkpoint=str(args.out) if best_ep else None)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V9_FULL_RESULT",json.dumps({k:result[k] for k in (
        "status","base","best","best_epoch","paired_states")}),flush=True)


if __name__=="__main__":
    main()
