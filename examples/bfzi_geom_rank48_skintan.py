"""Train a port-factorized impedance tangent from direct rank48 skin states."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_free import geometry,rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z
from sdfmpneo_bfzi.geom_skintan import SkinTanV3,SkinTanV4


def get_init(case,atlas,ps,mu,src):
    out=atlas.predict(case)
    for model in (ps,mu,src):
        if model is not None:
            out=model.apply(case,out)
    return out


def collect(specs,plan,atlas,net,ck,base,ps,mu,src,shape_mix):
    rows=[]
    for spec in specs:
        name=Path(spec["anchor"]).stem.split("_")[2]
        case=plan["cases"][name]
        init=get_init(case,atlas,ps,mu,src)
        _,feat=geometry(case,init)
        for row in prepare(spec,48):
            pred=rollout(case,atlas,net,ck,row["time"],row["voltage"],
                base,init_override=init,shape_mix=shape_mix)
            n=len(row["time"])
            g=torch.as_tensor(np.broadcast_to(feat,(n,len(feat))).copy(),
                              dtype=torch.float32)
            a=torch.as_tensor(pred["heat_state"],dtype=torch.float32)
            with torch.no_grad():
                target=(torch.as_tensor(row["y"][:,:6],dtype=torch.float32)
                        -base(g,a))
            den=max(float(np.linalg.norm(row["z6"]-row["z6"][0])),1e-12)
            rows.append(dict(name=name,case=case,init=init,row=row,
                g=g,a=a,skin=a[:,32:48].detach(),target=target,den=den))
        print("R48_SKIN_PREP",name,len(rows),flush=True)
    return rows


def make_tan(model,base,sy):
    def predict(g,a):
        if a.ndim!=2 or a.shape[1]!=48:
            raise ValueError("rank48 tangent requires 48 heat coordinates")
        return base(g,a)+model(g,a[:,:32],a[:,32:48])*sy
    return predict


def closed(rows,model,base,sy,atlas,net,ck,shape_mix):
    tan=make_tan(model,base,sy)
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
    skin=torch.cat([r["skin"] for r in rows])
    target=torch.cat([r["target"] for r in rows])
    pred=model(g,a[:,:32],skin)*sy
    vals=[]
    start=0
    for r in rows:
        n=len(r["target"])
        vals.append((pred[start:start+n]-target[start:start+n]).square().sum()
                    /r["den"]**2)
        start+=n
    return torch.stack(vals).mean()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_pose42_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v6_36g_hold6_manifest.json"))
    p.add_argument("--atlas",type=Path,default=Path("runs/geomfrac_init_pose42_kell_36g_r48_v1.npz"))
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--z-fix-model",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--port-self-model",type=Path,default=Path("runs/geomfrac_portself_v5.npz"))
    p.add_argument("--mutual-model",type=Path,default=Path("runs/geomfrac_mutual_v6.npz"))
    p.add_argument("--source-model",type=Path,default=Path("runs/geomfrac_src_v6.npz"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v1.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v1.json"))
    p.add_argument("--shape-mix",type=float,default=.5)
    p.add_argument("--epochs",type=int,default=180)
    p.add_argument("--eval-every",type=int,default=10)
    p.add_argument("--patience",type=int,default=60)
    p.add_argument("--lr",type=float,default=.0008)
    p.add_argument("--variant",choices=("v3","v4"),default="v3")
    a=p.parse_args()
    torch.manual_seed(73)
    torch.set_num_threads(2)
    ck=torch.load(a.model,map_location="cpu",weights_only=True)
    if int(ck["rank"])!=48:
        raise ValueError("rank48 thermal checkpoint required")
    net=GeomFracET(ng=int(ck["ng"]),rank=48,
        soe_quad=ck.get("soe_quad","trap"),heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"]);net.eval()
    atlas=GeomInit.load(a.atlas)
    if atlas.rank!=48:
        raise ValueError("rank48 thermal atlas required")
    base=load_z(a.z_fix_model)
    ps=PortSelf.load(a.port_self_model) if a.port_self_model else None
    mu=MutFix.load(a.mutual_model) if a.mutual_model else None
    src=SrcFix.load(a.source_model) if a.source_model else None
    plan=json.loads(a.plan.read_text(encoding="utf-8-sig"))
    mf=json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    trset={str(x["anchor"]).replace(chr(92),"/") for x in mf["train"]}
    vaset={str(x["anchor"]).replace(chr(92),"/") for x in mf["val"]}
    teset={str(x["anchor"]).replace(chr(92),"/") for x in mf.get("test",[])}
    if not trset or not vaset or trset&vaset or trset&teset or vaset&teset:
        raise ValueError("train, dev and sealed test geometries must be disjoint")
    tr=collect(mf["train"],plan,atlas,net,ck,base,ps,mu,src,a.shape_mix)
    va=collect(mf["val"],plan,atlas,net,ck,base,ps,mu,src,a.shape_mix)
    sy=torch.as_tensor(np.asarray(ck["y_scale"])[:6],dtype=torch.float32)
    all_skin=torch.cat([r["skin"] for r in tr]).numpy()
    ss=np.maximum(np.std(all_skin,axis=0),.05)
    cls=SkinTanV3 if a.variant=="v3" else SkinTanV4
    model=cls(ng=int(ck["ng"]),sa=np.asarray(ck["a_scale"])[:32],ss=ss)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.002)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.6,patience=12)
    baseline=closed(va,model,base,sy,atlas,net,ck,a.shape_mix)
    best=baseline["dynamic_z_pct"];best_ep=0
    saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    hist=[dict(epoch=0,**baseline)]
    print("R48_SKIN_BASE",json.dumps(baseline),flush=True)
    for ep in range(1,a.epochs+1):
        model.train();opt.zero_grad(set_to_none=True)
        obj=loss(model,tr,sy);obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
        if ep%a.eval_every:continue
        model.eval()
        score=closed(va,model,base,sy,atlas,net,ck,a.shape_mix)
        sched.step(score["dynamic_z_pct"])
        hist.append(dict(epoch=ep,**score))
        if score["dynamic_z_pct"]<best-1e-4:
            best=score["dynamic_z_pct"];best_ep=ep
            saved={k:v.detach().clone() for k,v in model.state_dict().items()}
        print("R48_SKIN_DEV",ep,"loss",round(float(obj.detach()),6),
            "dyn",round(score["dynamic_z_pct"],4),"best",round(best,4),
            "epoch",best_ep,flush=True)
        if ep-best_ep>=a.patience:break
    model.load_state_dict(saved)
    final=closed(va,model,base,sy,atlas,net,ck,a.shape_mix)
    result=dict(status="rank48 direct-skin tangent development selection",
        rank=48,train_geometries=len(mf["train"]),dev_geometries=len(mf["val"]),
        train_histories=len(tr),dev_histories=len(va),shape_mix=a.shape_mix,
        best_epoch=best_ep,baseline_dev=baseline,best_dev=final,history=hist,
        thermal_model=str(a.model),base_model=str(a.z_fix_model),
        feature_source="predicted rank48 heat states: first32 global, last16 TX/RX sea-skin",
        tangent_variant=a.variant)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=model.state_dict(),ng=int(ck["ng"]),model_version=a.variant,
        sa=torch.as_tensor(np.asarray(ck["a_scale"])[:32],dtype=torch.float32),
        ss=torch.as_tensor(ss,dtype=torch.float32),sy=sy,best_epoch=best_ep,
        shape_mix=a.shape_mix,rank=48),a.out)
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("R48_SKIN_FINAL",json.dumps(dict(epoch=best_ep,baseline=baseline,
        best=final,out=str(a.out))),flush=True)


if __name__=="__main__":
    main()
