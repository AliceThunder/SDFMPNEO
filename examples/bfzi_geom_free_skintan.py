"""Train a port-factorized impedance correction from V6 local sea-skin modes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_free import geometry,rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_ztune import restore as load_z
from sdfmpneo_bfzi.geom_skin import restore as load_skin
from sdfmpneo_bfzi.geom_skintan import SkinTan
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def get_init(case,atlas,ps,mu,src):
    x=atlas.predict(case)
    if ps is not None:x=ps.apply(case,x)
    if mu is not None:x=mu.apply(case,x)
    if src is not None:x=src.apply(case,x)
    return x


def collect(specs,plan,atlas,net,ck,tan,skin,ps,mu,src,shape_mix):
    rows=[]
    for spec in specs:
        name=Path(spec["anchor"]).stem.split("_")[2]
        case=plan["cases"][name]
        init=get_init(case,atlas,ps,mu,src)
        _,feat=geometry(case,init)
        for row in prepare(spec,32):
            pred=rollout(case,atlas,net,ck,row["time"],row["voltage"],tan,
                         init_override=init,shape_mix=shape_mix)
            n=len(row["time"])
            g=torch.as_tensor(np.broadcast_to(feat,(n,len(feat))).copy(),
                              dtype=torch.float32)
            a=torch.as_tensor(pred["heat_state"],dtype=torch.float32)
            tm=torch.as_tensor(row["time"][:,None],dtype=torch.float32)
            vin=np.r_[0.,row["voltage"][:-1]]
            v=torch.as_tensor(vin[:,None],dtype=torch.float32)
            with torch.no_grad():
                sf=skin(g,a,tm,v)
                base=tan(g,a)
            target=torch.as_tensor(row["y"][:,:6],dtype=torch.float32)-base
            den=max(float(np.linalg.norm(row["z6"]-row["z6"][0])),1e-12)
            rows.append(dict(name=name,case=case,init=init,row=row,
                g=g,a=a,skin=sf.detach(),target=target,den=den))
        print("SKINTAN_PREP",name,len(rows),flush=True)
    return rows


def make_tan(model,base,skin,sy):
    def predict(g,a,tm,voltage):
        t=torch.full((len(a),1),float(tm),dtype=a.dtype,device=a.device)
        v=torch.full((len(a),1),float(voltage),dtype=a.dtype,device=a.device)
        sf=skin(g,a,t,v)
        return base(g,a)+model(g,a,sf)*sy
    predict.time_conditioned=True
    return predict


def closed(rows,model,base,skin,sy,atlas,net,ck,shape_mix):
    tan=make_tan(model,base,skin,sy)
    out=[]
    with torch.no_grad():
        for r in rows:
            pred=rollout(r["case"],atlas,net,ck,r["row"]["time"],
                r["row"]["voltage"],tan,init_override=r["init"],
                shape_mix=shape_mix)
            out.append(metric(pred,r["row"]))
    return dict(
        dynamic_z_pct=float(np.mean([x["dynamic_z_rel_L2"] for x in out])*100),
        current_pct=float(np.mean([x["current_rel_L2"] for x in out])*100),
        temp_rmse_K=float(np.mean([x["material_mean_rmse_K"] for x in out])),
        temp_max_K=float(np.mean([x["material_mean_max_K"] for x in out])),
        budget_W=float(max(x["energy_budget_W"] for x in out)))


def loss(model,rows,sy):
    g=torch.cat([r["g"] for r in rows])
    a=torch.cat([r["a"] for r in rows])
    s=torch.cat([r["skin"] for r in rows])
    target=torch.cat([r["target"] for r in rows])
    pred=model(g,a,s)*sy
    err=[]
    start=0
    for r in rows:
        n=len(r["target"])
        err.append((pred[start:start+n]-r["target"]).square().sum()/r["den"]**2)
        start+=n
    return torch.stack(err).mean()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_s5_v4_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v4_30g_manifest.json"))
    p.add_argument("--atlas",type=Path,default=Path("runs/geomfrac_init_v1.npz"))
    p.add_argument("--model",type=Path,default=Path("runs/geomfrac_et_v4_8g_quality.pt"))
    p.add_argument("--z-fix-model",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--skin-model",type=Path,default=Path("runs/geomfrac_et_v7_skin_resid.pt"))
    p.add_argument("--port-self-model",type=Path,default=Path("runs/geomfrac_portself_v5.npz"))
    p.add_argument("--mutual-model",type=Path,default=Path("runs/geomfrac_mutual_v6.npz"))
    p.add_argument("--source-model",type=Path,default=Path("runs/geomfrac_src_v6.npz"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_free_skin_tan_v1.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_free_skin_tan_v1_dev.json"))
    p.add_argument("--shape-mix",type=float,default=.5)
    p.add_argument("--epochs",type=int,default=180)
    p.add_argument("--eval-every",type=int,default=10)
    p.add_argument("--patience",type=int,default=60)
    p.add_argument("--lr",type=float,default=.0008)
    args=p.parse_args()
    if args.eval_every<1 or args.epochs<1 or not 0<=args.shape_mix<=1:
        p.error("invalid training schedule or shape mix")
    torch.manual_seed(73)
    torch.set_num_threads(2)
    ck=torch.load(args.model,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=32,
        soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"]);net.eval()
    atlas=GeomInit.load(args.atlas)
    base=load_z(args.z_fix_model)
    skin=load_skin(args.skin_model)
    ps=PortSelf.load(args.port_self_model) if args.port_self_model else None
    mu=MutFix.load(args.mutual_model) if args.mutual_model else None
    src=SrcFix.load(args.source_model) if args.source_model else None
    plan=json.loads(args.plan.read_text(encoding="utf-8"))
    mf=json.loads(args.manifest.read_text(encoding="utf-8"))
    trset={str(x["anchor"]).replace(chr(92),"/") for x in mf["train"]}
    vaset={str(x["anchor"]).replace(chr(92),"/") for x in mf["val"]}
    teset={str(x["anchor"]).replace(chr(92),"/") for x in mf.get("test",[])}
    if (not trset or not vaset or trset&vaset or
            trset&teset or vaset&teset):
        raise ValueError("train, dev and sealed test geometries must be disjoint")
    tr=collect(mf["train"],plan,atlas,net,ck,base,skin,ps,mu,src,args.shape_mix)
    va=collect(mf["val"],plan,atlas,net,ck,base,skin,ps,mu,src,args.shape_mix)
    sy=torch.as_tensor(np.asarray(ck["y_scale"])[:6],dtype=torch.float32)
    all_s=torch.cat([r["skin"] for r in tr]).numpy()
    ss=np.maximum(np.std(all_s,axis=0),.05)
    model=SkinTan(ng=int(ck["ng"]),sa=np.asarray(ck["a_scale"]),ss=ss)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.002)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,mode="min",factor=.6,patience=12)
    baseline=closed(va,model,base,skin,sy,atlas,net,ck,args.shape_mix)
    best=baseline["dynamic_z_pct"]
    best_ep=0
    saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    hist=[dict(epoch=0,**baseline)]
    print("SKINTAN_BASE",json.dumps(baseline),flush=True)
    for ep in range(1,args.epochs+1):
        model.train()
        opt.zero_grad(set_to_none=True)
        obj=loss(model,tr,sy)
        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
        if ep%args.eval_every:
            continue
        model.eval()
        score=closed(va,model,base,skin,sy,atlas,net,ck,args.shape_mix)
        sched.step(score["dynamic_z_pct"])
        hist.append(dict(epoch=ep,**score))
        if score["dynamic_z_pct"]<best-1e-4:
            best=score["dynamic_z_pct"]
            best_ep=ep
            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
        print("SKINTAN_DEV",ep,"loss",round(float(obj.detach()),6),
              "dyn",round(score["dynamic_z_pct"],4),
              "current",round(score["current_pct"],4),
              "best",round(best,4),"epoch",best_ep,flush=True)
        if ep-best_ep>=args.patience:
            break
    model.load_state_dict(saved)
    final=closed(va,model,base,skin,sy,atlas,net,ck,args.shape_mix)
    result=dict(status="development-selected port-factorized local-skin tangent",
        train_geometries=len(mf["train"]),dev_geometries=len(mf["val"]),
        train_histories=len(tr),dev_histories=len(va),
        architecture="TX head uses TX local skin modes; RX head uses RX local skin modes; mutual head uses both",
        shape_mix=args.shape_mix,best_epoch=best_ep,
        baseline_dev=baseline,best_dev=final,history=hist,
        base_model=str(args.z_fix_model),skin_model=str(args.skin_model),
        port_self=str(args.port_self_model),mutual=str(args.mutual_model),
        source=str(args.source_model))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state_dict(),ng=int(ck["ng"]),
        sa=np.asarray(ck["a_scale"]),ss=ss,sy=np.asarray(ck["y_scale"])[:6],
        best_epoch=best_ep,shape_mix=args.shape_mix,
        train_names=sorted({r["name"] for r in tr}),
        dev_names=sorted({r["name"] for r in va})),args.out)
    args.report.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("SKINTAN_FINAL",json.dumps(dict(epoch=best_ep,baseline=baseline,
          best=final,out=str(args.out))),flush=True)


if __name__=="__main__":
    main()
