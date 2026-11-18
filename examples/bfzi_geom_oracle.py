"""Diagnose seawater tangent errors with teacher vs autonomous 32D heat states."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_geom_multi import prepare
from sdfmpneo_bfzi.geom_free import geometry,rollout
from sdfmpneo_bfzi.geom_init import GeomInit
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_portself import PortSelf
from sdfmpneo_bfzi.geom_mutual import MutFix
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_ztune import restore as load_z


def init_net(path):
    ck=torch.load(path,map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=32,
                   soe_quad=ck.get("soe_quad","trap"),
                   heat_dt=float(ck.get("heat_dt",1.875)),
                   heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    net.eval()
    return net,ck


def evaluate(plan,mf,atlas,ps,mu,src,net,ck,tan,mix):
    out=[]
    for spec in mf["test"]:
        name=Path(spec["anchor"]).stem.split("_")[2]
        case=plan["cases"][name]
        init=atlas.predict(case)
        for fix in (ps,mu,src):
            init=fix.apply(case,init)
        _,feat=geometry(case,init)
        for r in prepare(spec,32):
            pred=rollout(case,atlas,net,ck,r["time"],r["voltage"],
                         tan,init_override=init,shape_mix=mix)
            a0=torch.as_tensor(r["a"],dtype=torch.float32)
            a1=torch.as_tensor(pred["heat_state"],dtype=torch.float32)
            g=torch.as_tensor(np.broadcast_to(feat,(len(a0),45)).copy(),
                              dtype=torch.float32)
            with torch.no_grad():
                true_state_tan=tan(g,a0).numpy()
                pred_state_tan=tan(g,a1).numpy()
            truth=r["y"][:,:6]
            total=r["z6"]-r["z6"][0]
            den=max(np.linalg.norm(total[1:]),1e-12)
            sea_den=max(np.linalg.norm(truth[1:]),1e-12)
            cu_model=r["rom"].port
            hot0=cu_model.copper(np.full(2,r["rom"].amb))
            # Teacher copper term equals dynamic total Z minus true sea Z.
            copper_true=total-truth
            copper_pred=np.asarray([
                cu_model.copper(np.asarray(t)[3:5])-hot0
                for t in pred["temp_group_K"]])
            copper_err=np.zeros_like(total)
            copper_err[:,[0,1]]=np.column_stack((copper_pred[:,0].real,copper_pred[:,0].imag))
            copper_err[:,[4,5]]=np.column_stack((copper_pred[:,1].real,copper_pred[:,1].imag))
            err_cu=copper_err-copper_true
            err_or=true_state_tan-truth
            err_pr=pred_state_tan-truth
            q=dict(geometry=name,file=r["file"],
                   sea_teacher_norm=float(sea_den),
                   full_dynamic_norm=float(den),
                   thermal_state_l2=float(np.linalg.norm(a1[1:].numpy()-a0[1:].numpy())/
                              max(np.linalg.norm(a0[1:].numpy()),1e-12)),
                   sea_oracle_err_total_pct=float(100*np.linalg.norm(err_or[1:])/den),
                   sea_online_err_total_pct=float(100*np.linalg.norm(err_pr[1:])/den),
                   sea_oracle_rel_pct=float(100*np.linalg.norm(err_or[1:])/sea_den),
                   sea_online_rel_pct=float(100*np.linalg.norm(err_pr[1:])/sea_den),
                   copper_temp_err_total_pct=float(100*np.linalg.norm(err_cu[1:])/den),
                   total_online_dynamic_pct=float(100*np.linalg.norm(
                        pred["z6"][1:]-pred["z6"][0]-total[1:])/den))
            for i,key in [(0,"R11"),(1,"X11"),(2,"R12"),(3,"X12"),(4,"R22"),(5,"X22")]:
                q["oracle_"+key+"_pct"]=float(100*np.linalg.norm(err_or[1:,i])/den)
                q["online_"+key+"_pct"]=float(100*np.linalg.norm(err_pr[1:,i])/den)
            out.append(q)
        print("ORACLE",name,flush=True)
    nums=[k for k,v in out[0].items() if isinstance(v,(float,int))]
    return dict(histories=len(out),
        geometry_count=len({r["geometry"] for r in out}),
        summary={k:float(np.mean([r[k] for r in out])) for k in nums},
        per_geometry={g:{k:float(np.mean([r[k] for r in out if r["geometry"]==g]))
                    for k in nums}
                    for g in sorted({r["geometry"] for r in out})},rows=out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--atlas",type=Path,default=Path("runs/geomfrac_init_v1.npz"))
    p.add_argument("--port-self-model",type=Path,default=Path("runs/geomfrac_portself_v5.npz"))
    p.add_argument("--mutual-model",type=Path,default=Path("runs/geomfrac_mutual_v6.npz"))
    p.add_argument("--source-model",type=Path,default=Path("runs/geomfrac_src_v6.npz"))
    p.add_argument("--z-fix-model",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--shape-mix",type=float,default=.5)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    plan=json.loads(args.plan.read_text(encoding="utf-8-sig"))
    mf=json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    if plan.get("invalid_for_physics") or len(mf.get("test",[]))!=len(plan.get("test",[])):
        raise ValueError("physical plan or test manifest invalid")
    net,ck=init_net(args.model)
    atlas=GeomInit.load(args.atlas)
    ps=PortSelf.load(args.port_self_model)
    mu=MutFix.load(args.mutual_model)
    src=SrcFix.load(args.source_model)
    tan=load_z(args.z_fix_model)
    report=evaluate(plan,mf,atlas,ps,mu,src,net,ck,tan,args.shape_mix)
    report.update(model=str(args.model),plan=str(args.plan),
        manifest=str(args.manifest),z_fix_model=str(args.z_fix_model),
        note="True 32D teacher state is geometry-specific S5 projection and is used only for offline diagnosis")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("ORACLE_SUMMARY",json.dumps(report["summary"]),flush=True)


if __name__=="__main__":
    main()
