"""Train geometry-conditioned fractional surrogate on original BFZI M1/M4.

Both shape families use the original BFZI construction, with different 3-D
seawater grids. Only physically comparable port and material observables are
shared across shapes. M2 is a fully held-out rounded-square geometry.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.frac_memory import zmat
from sdfmpneo_bfzi.geom_memory import (
    GeomFrac,G_SCALE,geom_feat,geom_port,heat_drive,thermal_summary,rollout)
from sdfmpneo_bfzi.pose import default_desc


def data_file(path,probe=None):
    with np.load(path) as d:
        times=np.asarray(d["time_s"],float)
        vin=np.asarray(d["voltage_v"],float)
        temp=np.asarray(d["temp_K"],float)
        z=np.asarray(d["z6"],float)
        train=np.asarray(d["train"],bool)
        ext=np.asarray(d["external_z"],complex)
        ns=int(d["nsea"])
        npkg=int(d["npkg"])
        amb=float(d["ambient_K"])
        cases=json.loads(str(d["cases_json"]))
        if "geometry" in d:
            geom=np.asarray(d["geometry"],float)
        elif probe:
            with np.load(probe) as b:
                geom=np.asarray(b["geometry"],float)
        else:
            raise ValueError("original BFZI geometry descriptor missing")
        if "sea_vol_m3" in d:
            vol=np.asarray(d["sea_vol_m3"],float)
        elif probe:
            with np.load(probe) as b:
                vol=np.asarray(b["sea_vol_m3"],float)
        else:
            raise ValueError("original BFZI sea volumes missing")
        sample=str(d["sample"]) if "sample" in d else (
            "M4" if probe else "unknown")
        pose=np.asarray(d["pose"],float) if "pose" in d else default_desc(geom)
    if temp.shape[-1]!=ns+2*npkg+2 or temp.shape[:2]!=vin.shape:
        raise ValueError("BFZI sea/package/copper dimensions mismatch")
    if (np.any(np.diff(times)<=0) or times[0]!=0 or
            len(vol)!=ns):
        raise ValueError("invalid original BFZI time or FV volume")
    z0=z[0,0]
    if np.max(abs(z[:,0]-z0))>1e-7:
        raise ValueError("original BFZI cold Z must match per geometry")
    summ=np.stack([thermal_summary(a,ns,npkg,vol,amb)
                   for a in temp])
    port=geom_port(geom)
    cold_cu=port.copper(np.full(2,amb))
    sea_delta=z-z0
    slope=np.empty(vin.shape+(4,))
    for i in range(len(vin)):
        for k in range(len(times)):
            tc=temp[i,k,-2:]
            dz=port.copper(tc)-cold_cu
            deriv=(port.copper(tc+.1)-port.copper(tc-.1))/.2
            slope[i,k]=[deriv[0].real,deriv[0].imag,
                        deriv[1].real,deriv[1].imag]
            sea_delta[i,k,0]-=dz[0].real
            sea_delta[i,k,1]-=dz[0].imag
            sea_delta[i,k,4]-=dz[1].real
            sea_delta[i,k,5]-=dz[1].imag
    src=np.zeros(vin.shape+(3,))
    for i in range(len(vin)):
        for k in range(1,len(times)):
            src[i,k]=heat_drive(
                float(vin[i,k]),zmat(z[i,k-1]),ext,port,
                temp[i,k-1,-2:],geom[14:16])
    return dict(sample=sample,geom=geom,pose=pose,z0=z0,ext=ext,
                times=times,v=vin,z=z,train=train,
                src=src,temp=summ,sea_delta=sea_delta,
                cu_slope=slope,cases=cases)


def data_all(files,probe):
    arr=[data_file(path,probe if str(path).lower().find("m4_")>=0
                   else None) for path in files]
    t=arr[0]["times"]
    if any(not np.array_equal(t,d["times"]) for d in arr):
        raise ValueError("first cross-geometry experiment uses a common time grid")
    fit=[];val=[];test={}
    for d in arr:
        train_ids=np.flatnonzero(d["train"])
        val_ids=[i for i in train_ids if d["cases"][i]["id"]%5==1]
        fit_ids=[i for i in train_ids if i not in val_ids]
        for ids,target in ((fit_ids,fit),(val_ids,val)):
            for i in ids:
                target.append((d,int(i)))
        test[d["sample"]]=[(d,int(i)) for i in np.flatnonzero(~d["train"])]
    if not fit or not val:
        raise ValueError("geometry training and validation histories required")
    return fit,val,test


def tensors(rows,src_scale,y_scale,port_sea=False,with_pose=False):
    src=np.stack([d["src"][i] for d,i in rows])/src_scale
    y=np.stack([np.concatenate(
        (d["sea_delta"][i] if port_sea else
         d["z"][i]-d["z0"],d["temp"][i]),axis=-1)
        for d,i in rows])/y_scale
    geom=np.stack([geom_feat(
        d["geom"],d["z0"],d["pose"] if with_pose else None)
        for d,i in rows])
    slope=np.stack([d["cu_slope"][i] for d,i in rows])
    return (torch.tensor(src,dtype=torch.float32),
            torch.tensor(y,dtype=torch.float32),
            torch.tensor(geom,dtype=torch.float32),
            torch.tensor(slope,dtype=torch.float32))


def score(model,ck,rows):
    scores=[]
    for d,i in rows:
        p=rollout(model,ck,d["v"][i],d["times"],
                  d["geom"],d["z0"],d["ext"],
                  d["pose"] if model.ng>len(G_SCALE) else None)
        z=d["z"][i]
        err=p["z6"]-z
        ref=z-z[0]
        temp=p["temp_rise_K"]-d["temp"][i]
        scores.append(dict(
            sample=d["sample"],
            case=int(d["cases"][i]["id"]),
            voltage_v=float(d["cases"][i]["voltage_v"]),
            dynamic_Z_relative_L2=float(
                np.linalg.norm(err)/max(np.linalg.norm(ref),1e-12)),
            full_Z_relative_L2=float(
                np.linalg.norm(err)/np.linalg.norm(z)),
            Tsea_mean_max_K=float(np.max(abs(temp[:,0]))),
            Tsea_hot_max_K=float(np.max(abs(temp[:,1]))),
            Tpkg_mean_max_K=float(np.max(abs(temp[:,2:4]))),
            Tpkg_hot_max_K=float(np.max(abs(temp[:,4:6]))),
            Tcu_max_K=float(np.max(abs(temp[:,6:8])))))
    keys=[x for x in scores[0] if x not in ("sample","case","voltage_v")]
    agg=dict(cases=len(scores))
    for k in keys:
        agg[k+"_mean"]=float(np.mean([r[k] for r in scores]))
        agg[k+"_max"]=float(np.max([r[k] for r in scores]))
    return agg,scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-files",nargs="+",type=Path,default=[
        Path("runs/m4_frac_teacher.npz"),
        Path("runs/m1_frac_teacher.npz")])
    p.add_argument("--m4-geom",type=Path,
                   default=Path("runs/m4_geom_probe.npz"))
    p.add_argument("--test-geo",type=Path,
                   default=Path("runs/m2_frac_holdout.npz"))
    p.add_argument("--model",type=Path,
                   default=Path("runs/bfzi_geom_frac.pt"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_geom_frac_metrics.json"))
    p.add_argument("--epochs",type=int,default=1600)
    p.add_argument("--port-sea",action="store_true",
                   help="Train sea delta and assemble exact temperature-dependent BFZI copper Z")
    p.add_argument("--phys-loss",type=float,default=0.,
                   help="Joint full-Z Jacobian loss coupling sea delta and copper temperature")
    p.add_argument("--pkg-loss",type=float,default=0.,
                   help="Additional two-package mean/hotspot temperature penalty")
    p.add_argument("--sea-loss",type=float,default=0.,
                   help="Additional seawater mean/hotspot temperature penalty")
    p.add_argument("--patience",type=int,default=280)
    p.add_argument("--with-pose",action="store_true",
                   help="Condition on BFZI relative 3D translation, rotation and both package sizes")
    p.add_argument("--init-model",type=Path,default=None,
                   help="Initialize first 22 features and full memory from trained BFZI V5")
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(41)
    fit,val,hold=data_all(args.train_files,args.m4_geom)
    sample,t=fit[0][0]["sample"],fit[0][0]["times"]
    other=data_file(args.test_geo)
    hold[other["sample"]]=[(other,i) for i in range(len(other["v"]))]
    sy=np.stack([np.concatenate(
        (d["sea_delta"][i] if args.port_sea else
         d["z"][i]-d["z0"],d["temp"][i]),-1)
        for d,i in fit])
    ystd=np.std(sy[:,1:].reshape(-1,14),axis=0)
    yscale=np.maximum(ystd,np.r_[
        np.full(6,max(float(max(ystd[:6]))*.001,1e-7)),
        np.full(8,max(float(max(ystd[6:]))*.001,1e-3))])
    src_scale=np.maximum(np.max(
        np.stack([d["src"][i] for d,i in fit]),axis=(0,1)),1e-4)
    init=(torch.load(args.init_model,map_location="cpu",weights_only=True)
          if args.init_model else None)
    if init is not None:
        # Preserve the physical scale of the V5 warm start.
        src_scale=np.asarray(init["src_scale"],float)
        yscale=np.asarray(init["y_scale"],float)
    u,y,g,cu=tensors(fit,src_scale,yscale,args.port_sea,args.with_pose)
    uv,yv,gv,cuv=tensors(val,src_scale,yscale,args.port_sea,args.with_pose)
    if args.phys_loss and not args.port_sea:
        p.error("--phys-loss requires --port-sea")
    ysc=torch.tensor(yscale,dtype=torch.float32)
    def physical_loss(pred,target,deriv):
        if args.phys_loss<=0:
            return pred.new_zeros(())
        error=(pred-target)*ysc
        ez=error[:,:,:6].clone()
        dt=error[:,:,12:14]
        ez[:,:,0]+=deriv[:,:,0]*dt[:,:,0]
        ez[:,:,1]+=deriv[:,:,1]*dt[:,:,0]
        ez[:,:,4]+=deriv[:,:,2]*dt[:,:,1]
        ez[:,:,5]+=deriv[:,:,3]*dt[:,:,1]
        return (ez/ysc[:6]).square().mean()
    def temp_loss(pred,target):
        loss=pred.new_zeros(())
        if args.pkg_loss>0:
            loss=loss+args.pkg_loss*(pred[:,:,8:12]-target[:,:,8:12]).square().mean()
        if args.sea_loss>0:
            loss=loss+args.sea_loss*(pred[:,:,6:8]-target[:,:,6:8]).square().mean()
        return loss
    model=GeomFrac(ng=g.shape[-1])
    if init is not None:
        state=init["state"]
        if model.ng==int(init["ng"]):
            model.load_state_dict(state)
        elif model.ng==int(init["ng"])+18:
            # Zero new pose columns: identical inference to V5 at initialization.
            extra=model.state_dict()
            for key,value in state.items():
                if key=="net.0.weight":
                    extra[key][:,:value.shape[1]]=value
                    extra[key][:,value.shape[1]:]=0.
                else:
                    extra[key]=value
            model.load_state_dict(extra)
        else:
            raise ValueError("pretrained BFZI geometry dimension mismatch")
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    best=1e30;state=None;best_epoch=0
    t0=time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        pred=model(u,t,g)
        loss=((pred-y).square().mean()+.45*
              (pred[:,:,:6]-y[:,:,:6]).square().mean()
              +args.phys_loss*physical_loss(pred,y,cu)
              +temp_loss(pred,y))
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vpred=model(uv,t,gv)
            vloss=float((vpred-yv).square().mean()+.45*
                        (vpred[:,:,:6]-yv[:,:,:6]).square().mean()
                        +args.phys_loss*physical_loss(vpred,yv,cuv)
                        +temp_loss(vpred,yv))
        if vloss<best-1e-7:
            best=vloss;best_epoch=epoch+1
            state={k:v.detach().clone()
                   for k,v in model.state_dict().items()}
        if epoch+1-best_epoch>=args.patience:
            break
    model.load_state_dict(state)
    model.eval()
    ck=dict(
        state=model.state_dict(),
        src_scale=torch.tensor(src_scale,dtype=torch.float32),
        y_scale=torch.tensor(yscale,dtype=torch.float32),
        ng=model.ng,with_pose=bool(args.with_pose),
        port_sea=bool(args.port_sea),
        history="fractional SOE + exponential",
        trained_samples=[d["sample"] for d in
                         {id(d):d for d,i in fit}.values()])
    args.model.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck,args.model)
    output=dict(
        model=str(args.model),
        train_samples=sorted(set(d["sample"] for d,i in fit)),
        heldout_geometry=other["sample"],port_sea=bool(args.port_sea),
        with_pose=bool(args.with_pose),
        physics_loss_weight=float(args.phys_loss),
        package_loss_weight=float(args.pkg_loss),
        sea_loss_weight=float(args.sea_loss),
        train_cases=len(fit),val_cases=len(val),
        trained_steps=len(t),epochs=epoch+1,best_epoch=best_epoch,
        val_loss=best,
        train_seconds=time.perf_counter()-t0,
        fractional_orders=model.orders().detach().numpy().tolist())
    for sample,rows in hold.items():
        if rows:
            agg,details=score(model,ck,rows)
            output[sample]=dict(aggregate=agg,cases=details)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(output,indent=2),encoding="utf-8")
    print(json.dumps(dict(
        model=str(args.model),train_samples=output["train_samples"],
        heldout_geometry=other["sample"],train_cases=len(fit),
        val_cases=len(val),best_epoch=best_epoch,
        result={k:v["aggregate"] for k,v in output.items()
                if isinstance(v,dict) and "aggregate" in v}),
        indent=2),flush=True)


if __name__=="__main__":
    main()
