"""Train/test continuous BFZI 3-D temperature fields across coil geometries.

Use exact original BFZI FV coordinates and thermal histories for M4/M1/M2,
with the complete M3 geometry excluded from shape training. A frozen shared
fractional-history checkpoint supplies the eight time-dependent states.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.geom_memory import GeomFrac,rollout,thermal_summary
from sdfmpneo_bfzi.geom_field import SpaceTemp,space_feat,time_feat,infer_field


def read(path,probe=None):
    with np.load(path) as d:
        g=np.asarray(d["geometry"] if "geometry" in d else
                     np.load(probe)["geometry"],float)
        sea=np.asarray(d["sea_xyz_m"] if "sea_xyz_m" in d else
                       np.load(probe)["sea_xyz_m"],float)
        vol=np.asarray(d["sea_vol_m3"] if "sea_vol_m3" in d else
                       np.load(probe)["sea_vol_m3"],float)
        pkg=np.asarray(d["pkg_xyz_m"] if "pkg_xyz_m" in d else
                       np.load(probe)["pkg_xyz_m"],float)
        v=np.asarray(d["voltage_v"],float)
        temp=np.asarray(d["temp_K"],float)
        z6=np.asarray(d["z6"],float)
        z0=z6[0,0].copy()
        times=np.asarray(d["time_s"],float)
        amb=float(d["ambient_K"])
        nsea=int(d["nsea"])
        npkg=int(d["npkg"])
        ext=np.asarray(d["external_z"],complex)
        cases=json.loads(str(d["cases_json"]))
        split=np.asarray(d["train"],bool)
        sample=str(d["sample"]) if "sample" in d else "M4"
    if sea.shape!=(nsea,3) or pkg.shape!=(2,npkg,3):
        raise ValueError("original BFZI spatial material coordinates are inconsistent")
    if temp.shape!=v.shape+(nsea+2*npkg+2,):
        raise ValueError("original BFZI temperature field dimension mismatch")
    coord=[sea,pkg[0],pkg[1]]
    static=[space_feat(x,g,z0,pkg,j) for j,x in enumerate(coord)]
    states=np.stack([thermal_summary(a,nsea,npkg,vol,amb)
                     for a in temp])
    return dict(sample=sample,path=str(path),
                geom=g,sea_xyz=sea,sea_vol=vol,pkg_xyz=pkg,
                static=static,states=states,temp=temp,amb=amb,
                z0=z0,ext=ext,v=v,times=times,cases=cases,
                split=split,nsea=nsea,npkg=npkg)


def partitions(files,probe):
    ds=[read(p,probe if Path(p).name.startswith("m4_") else None)
        for p in files]
    fit=[];val=[];hold=[]
    for d in ds:
        for i,case in enumerate(d["cases"]):
            cid=int(case["id"])
            if d["split"][i]:
                target=val if cid%5==1 else fit
            else:
                target=hold
            target.append((d,i))
    return fit,val,hold


def sample_batch(rows,rng,n_hist=40,nsea=80,npkg=24):
    inputs=[];targets=[];weights=[]
    for ix in rng.integers(0,len(rows),size=n_hist):
        d,i=rows[int(ix)]
        k=int(rng.integers(1,len(d["times"])))
        temp=d["temp"][i,k]-d["amb"]
        states=d["states"][i,k]
        dyn=time_feat(states,d["times"][k])
        for j,n in enumerate((nsea,npkg,npkg)):
            start=0 if j==0 else d["nsea"]+(j-1)*d["npkg"]
            nc=len(d["static"][j])
            ix=rng.integers(0,nc,size=n)
            hot=max(float(states[[1,4,5][j]]),1e-8)
            yy=np.clip(temp[start+ix]/hot,0,1)
            x=np.column_stack((d["static"][j][ix],
                               np.broadcast_to(dyn,(len(ix),9))))
            inputs.append(x.astype(np.float32))
            targets.append(yy.astype(np.float32))
            weights.append((1+3*yy**4).astype(np.float32))
    return tuple(torch.from_numpy(np.concatenate(y)) for y in
                 (inputs,targets,weights))


def eval_cases(net,model,ck,rows):
    details=[]
    for d,i in rows:
        times=d["times"]
        actual=d["temp"][i]-d["amb"]
        gt=d["states"][i]
        pred=rollout(model,ck,d["v"][i],times,
                     d["geom"],d["z0"],d["ext"])
        scores={}
        for name,states in (("oracle",gt),
                            ("neural",pred["temp_rise_K"])):
            s,b=infer_field(net,d["static"],states,times,d["sea_vol"])
            ds=abs(s-actual[:,:d["nsea"]])
            dp=abs(b-actual[:,d["nsea"]:d["nsea"]+2*d["npkg"]].
                   reshape(-1,2,d["npkg"]))
            scores[name+"_sea_max_abs_K"]=float(ds.max())
            scores[name+"_pkg_max_abs_K"]=float(dp.max())
            scores[name+"_sea_L2_relative"]=float(
                np.linalg.norm(ds)/max(np.linalg.norm(
                    actual[:,:d["nsea"]]),1e-8))
            scores[name+"_pkg_L2_relative"]=float(
                np.linalg.norm(dp)/max(np.linalg.norm(
                    actual[:,d["nsea"]:d["nsea"]+2*d["npkg"]]),1e-8))
            if name=="neural":
                scores["neural_Tsea_K_last"]=(s[-1]+d["amb"]).tolist()
                scores["neural_Tpkg_K_last"]=(b[-1]+d["amb"]).tolist()
        details.append(dict(sample=d["sample"],case=int(d["cases"][i]["id"]),
                            voltage_v=float(d["cases"][i]["voltage_v"]),**scores))
    labels=[k for k in details[0] if k.endswith(
        ("max_abs_K","L2_relative"))]
    summary=dict(count=len(rows))
    for k in labels:
        summary[k+"_mean"]=float(np.mean([x[k] for x in details]))
        summary[k+"_worst"]=float(np.max([x[k] for x in details]))
    return summary,details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher",type=Path,nargs="+",default=[
        Path("runs/m4_frac_teacher.npz"),
        Path("runs/m1_frac_teacher.npz"),
        Path("runs/m2_frac_teacher.npz"),
        Path("runs/m2_frac_holdout.npz"),
        Path("runs/m3_frac_holdout.npz")])
    p.add_argument("--m4-geom",type=Path,
                   default=Path("runs/m4_geom_probe.npz"))
    p.add_argument("--memory",type=Path,
                   default=Path("runs/bfzi_geom_frac_v5.pt"))
    p.add_argument("--model",type=Path,
                   default=Path("runs/bfzi_geom_field.pt"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_geom_field_metrics.json"))
    p.add_argument("--epochs",type=int,default=1000)
    p.add_argument("--patience",type=int,default=180)
    p.add_argument("--batch-hist",type=int,default=40)
    args=p.parse_args()
    torch.manual_seed(23)
    torch.set_num_threads(2)
    rng=np.random.default_rng(19)
    fit,val,hold=partitions(args.teacher,args.m4_geom)
    if not fit or not val or not hold:
        raise ValueError("independent BFZI train/validation/test histories required")
    if any(d["sample"]=="M3" for d,i in fit+val):
        raise ValueError("M3 geometry must remain completely unseen")
    net=SpaceTemp()
    opt=torch.optim.Adam(net.parameters(),lr=.002)
    best=1e9;best_epoch=0;best_weights=None
    t0=time.perf_counter()
    # Fixed validation subsample: only training geometries, never M3.
    vx,vy,vw=sample_batch(val,np.random.default_rng(110),n_hist=64)
    for epoch in range(args.epochs):
        net.train()
        x,y,w=sample_batch(fit,rng,args.batch_hist)
        opt.zero_grad(set_to_none=True)
        out=net(x)
        loss=(w*(out-y)**2).mean()
        loss.backward()
        opt.step()
        if (epoch+1)%10==0 or epoch==0:
            net.eval()
            with torch.no_grad():
                vl=float((vw*(net(vx)-vy)**2).mean())
            if vl<best-1e-7:
                best=vl;best_epoch=epoch+1
                best_weights={k:v.detach().clone()
                              for k,v in net.state_dict().items()}
            if epoch+1-best_epoch>=args.patience:
                break
    net.load_state_dict(best_weights)
    net.eval()
    ck=torch.load(args.memory,map_location="cpu",weights_only=True)
    memory=GeomFrac(ng=int(ck["ng"]))
    memory.load_state_dict(ck["state"])
    memory.eval()
    out=dict(model=str(args.model),history_model=str(args.memory),
             original_bfzi_samples_train=sorted({d["sample"] for d,i in fit}),
             original_bfzi_samples_validation=sorted({d["sample"] for d,i in val}),
             train_histories=len(fit),validation_histories=len(val),
             epochs=epoch+1,best_epoch=best_epoch,
             training_seconds=time.perf_counter()-t0,
             normalized_shape_validation_loss=best,
             original_bfzi_unseen_geometry="M3",
             material_query="continuous xyz on original BFZI 3-D seawater and dual-package cells",
             field_constraints="nonnegative; FV material mean and hotspot fixed to eight temporal network observables")
    for sample in sorted({d["sample"] for d,i in hold}):
        rows=[(d,i) for d,i in hold if d["sample"]==sample]
        agg,cases=eval_cases(net,memory,ck,rows)
        out[sample]=dict(aggregate=agg,cases=cases)
    args.model.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state=net.state_dict(),model="SpaceTemp",
                    train_samples=out["original_bfzi_samples_train"]),args.model)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps({
        "train":out["original_bfzi_samples_train"],
        "history_model":str(args.memory),
        "best_epoch":best_epoch,
        "results":{k:out[k]["aggregate"] for k in ("M1","M2","M3","M4")
                   if k in out}},indent=2),flush=True)


if __name__=="__main__":
    main()
