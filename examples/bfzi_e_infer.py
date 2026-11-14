"""E: input new BFZI geometry, seawater medium and arbitrary causal voltage CSV."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np

from examples.bfzi_param import create, originals
from examples.bfzi_obb_layer_study import GRIDS
from examples.bfzi_d_shared import make_geo, TARGET
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import pack
from sdfmpneo_bfzi.geom_shared_rom import SharedROM
from sdfmpneo_bfzi.end_to_end import complete, metrics

def curve(args, spec, runs):
    if args.history:
        arr=np.loadtxt(args.history,delimiter=",",ndmin=2)
        if arr.shape[1]!=2:
            raise ValueError("CSV history must have two columns: time_s,voltage_v")
        return arr[:,0],arr[:,1]
    if args.check and args.target:
        if args.target=="g3":
            with np.load(runs/"bfzi_d_g3_truth.npz") as x:
                return x["time_s"].copy(),x["voltage_v"].copy()
        with np.load(runs/f"bfzi_pose_{args.target}_skin_source_subcell_extra.npz") as x:
            return x["time_s"].copy(),x["voltage_v"].copy()
    dt=float(spec.get("dt",args.dt))
    dur=float(spec.get("duration",args.duration))
    voltage=float(spec.get("voltage",args.voltage))
    t=np.arange(0.,dur+dt*.1,dt)
    v=np.full_like(t,voltage)
    for raw in spec.get("off",args.off):
        a,b=map(float,raw.split(":")) if isinstance(raw,str) else raw
        if not 0<=a<b<=dur:
            raise ValueError("shutdown times outside 0..duration")
        v[(t>=a)&(t<b)]=0.
    return t,v

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--input",type=Path,help="JSON geometry and medium parameters")
    p.add_argument("--target",choices=tuple(TARGET))
    p.add_argument("--donor",choices=("g1","g2"))
    p.add_argument("--memory",choices=("none","frac","exp"),default="frac")
    p.add_argument("--history",type=Path,help="time_s,voltage_v CSV, without header")
    p.add_argument("--voltage",type=float,default=75.)
    p.add_argument("--duration",type=float,default=600.)
    p.add_argument("--dt",type=float,default=15.)
    p.add_argument("--off",action="append",default=[])
    p.add_argument("--xyz",type=Path,help="Nx3 CSV physical XYZ in meters")
    p.add_argument("--out",type=Path,default=Path("runs/bfzi_e_infer.npz"))
    p.add_argument("--check",action="store_true",help="compare preset BFZI V150 truth")
    a=p.parse_args()
    if a.target is None and a.input is None:
        p.error("provide either --target or --input")
    spec=json.loads(a.input.read_text()) if a.input else {}
    if a.target is not None:
        _,geo=make_geo(a.runs,a.target)
        donor=TARGET[a.target]
    else:
        geo=spec["geometry"]
        donor=spec.get("donor",a.donor)
        if donor is None:
            p.error("geometry JSON requires donor g1 or g2")
    bf,v150=originals(a.bfzi_root)
    xz,y=GRIDS["extra"]
    t0=time.perf_counter()
    data,_=create(bf,v150,"M4",**geo,sea_void="oriented",
                  physical_box_only=True,grid_xz=xz,grid_y=y)
    em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                       fv_faces="exact",source_mask="legacy")
    medium=spec.get("medium",{})
    if "sigma_S_m" in medium:
        em.sigma0=float(medium["sigma_S_m"])
    if "alpha_per_K" in medium:
        em.alpha=float(medium["alpha_per_K"])
    cold=em.solve(np.full(len(em.vol),293.15),np.full(2,293.15))
    shared=SharedROM.load(a.runs,donor)
    cold_z=pack(cold.total_z)
    prep=time.perf_counter()-t0
    t,v=curve(a,spec,a.runs)
    pts=(np.loadtxt(a.xyz,delimiter=",",ndmin=2) if a.xyz else None)
    t1=time.perf_counter()
    alpha_scale=em.alpha/.02
    model=shared.target(cold_z,alpha_scale=alpha_scale)
    net=ck=None
    if a.memory!="none":
        import torch
        from sdfmpneo_bfzi.frac_memory import FracNet
        torch.set_num_threads(1)
        path=a.runs/f"bfzi_frac_rom_{donor}_{a.memory}.pt"
        ck=torch.load(path,map_location="cpu",weights_only=False)
        net=FracNet(nout=6,mode=a.memory,hidden=ck["hidden"])
        net.load_state_dict(ck["state"])
        net.eval()
        # Causal rated-voltage normalization: given at the initial time.
        rated=max(abs(float(v[0])),1e-8)
        ratio=rated/ck["vref"]
        ck=dict(ck,vref=rated,scale=np.asarray(ck["scale"])*ratio**2)
    pred=shared.rollout(cold_z,t,v,net=net,ck=ck,sea_xyz=em.points,
                        alpha_scale=alpha_scale)
    if pts is None:
        # Four native points with strongest cold-Joule source, close to windings.
        cur=pred["current_A"]
        idx=int(np.argmax(abs(v)))
        raw=.5*np.real(np.einsum(
            "i,nij,j->n",np.conj(cur[idx]),cold.weak_q,cur[idx]))
        pts=em.points[np.argsort(raw)[-4:]]
    out=complete(em,cold,model,pred,v,xyz=pts)
    elapsed=time.perf_counter()-t1
    report=dict(target=a.target or "custom",donor=donor,
                memory=a.memory,
                original_BFZI_cold_solves=1,
                original_BFZI_dynamic_solves=0,
                build_sec=float(prep),infer_sec=float(elapsed),
                steps=len(t),sea_cells=len(em.vol),
                metrics=metrics(out))
    if a.check:
        if not a.target:
            p.error("--check needs --target p03, p07 or g3")
        if a.target=="g3":
            src=a.runs/"bfzi_d_g3_truth.npz"
            direct=19.262646900009713
        else:
            src=a.runs/f"bfzi_pose_{a.target}_skin_source_subcell_extra.npz"
            base=json.loads((a.runs/"bfzi_obb_skin_source.json").read_text())
            direct=float(base["geometries"][a.target.upper()][
                "modes"]["subcell"]["levels"]["extra"]["direct_sec"])
        with np.load(src) as rr:
            if not np.allclose(t,rr["time_s"]) or not np.allclose(v,rr["voltage_v"]):
                raise ValueError("validation history does not match locked BFZI truth")
            dz=rr["z6"]-rr["z6"][0]
            pdz=out["z6"]-out["z6"][0]
            x=out["temp_K"]-rr["temp_K"]
            ns=len(em.vol)
            report["heldout"]=dict(
                delta_Z_rel_L2=float(np.linalg.norm(pdz-dz)/
                                    max(np.linalg.norm(dz),1e-30)),
                pkg_max_abs_K=float(np.max(abs(x[:,ns:ns+64]))),
                copper_max_abs_K=float(np.max(abs(x[:,ns+64:ns+66]))))
            report["heldout"]["direct_sec"]=direct
            report["heldout"]["online_speedup"]=float(direct/elapsed)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**out)
    a.out.with_name(a.out.stem+"_report.json").write_text(
        json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__=="__main__":
    main()
