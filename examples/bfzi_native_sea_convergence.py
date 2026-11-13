"""BFZI V150 native near-package sea-FV mesh convergence on held-out poses.

Runs the ORIGINAL volume-integrated mesh, original temperature-dependent
EM solver, physical 3-D package heat FV, and causal 6-direction hybrid
at the finest mesh. Does not train on P03/P07 truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args,values
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import basis
from sdfmpneo_bfzi.geom_tangent_subspace import build as tangent_build
from sdfmpneo_bfzi.geom_memory import GeomFrac
import torch


GRIDS = (
    ("legacy",None,None),
    ("face_coarse",.20,.15),
    ("face_medium",.14,.12),
    ("face_fine",.11,.09),
)


def direct(et, t, volt):
    """Original warm BFZI transient and five-material conservation residual."""
    def v_at(x):
        i=int(np.argmin(abs(t-x)))
        return np.array([volt[i],0.],complex)
    ts=time.perf_counter()
    seq=et.trajectory(t,v_at)
    sec=time.perf_counter()-ts
    T=et.amb+np.stack([s.temp for s in seq])
    z=np.stack([pack(s.impedance) for s in seq])
    flux=np.stack([et.heat_flux(s.temp)[1] for s in seq])
    budget=[]
    for i in range(1,len(seq)):
        dt=t[i]-t[i-1]
        heat=np.zeros(et.n)
        heat[:et.count]=seq[i].sea_q*et.model.vol
        heat[et.cu]=seq[i].copper_power
        residual=et.cap*(seq[i].temp-seq[i-1].temp)/dt+et.H@seq[i].temp-heat
        s=et.count;m=et.pkg_n
        budget.append([
            float(np.sum(residual[:s])),
            float(np.sum(residual[s:s+m])),
            float(np.sum(residual[s+m:s+2*m])),
            float(residual[et.cu.start]),
            float(residual[et.cu.start+1])])
    return dict(time_s=t,voltage_v=volt,temp_K=T,z6=z,
                sea_xyz=et.model.points.copy(),sea_vol=et.model.vol.copy(),
                nsea=et.count,elapsed_sec=sec,
                max_budget_W=float(np.max(np.abs(budget))) if budget else 0.,
                max_pkg_flux_W=np.max(abs(flux),axis=0).tolist())


def compare(s,ref):
    """Compare only matching physical points, respecting different sea meshes."""
    T,Z=s["temp_K"],s["z6"]
    F,R=ref["temp_K"],ref["z6"]
    ns,nr=s["nsea"],ref["nsea"]
    tree=cKDTree(s["sea_xyz"])
    e_sea=0.
    for a,b in zip(T,F):
        t=GeomHeat._interp(tree,a[:ns],ref["sea_xyz"])
        e_sea=max(e_sea,float(np.max(abs(t-b[:nr]))))
    pkg=np.max(abs(T[:,ns:-2]-F[:,nr:-2]))
    cu=np.max(abs(T[:,-2:]-F[:,-2:]))
    dz=Z-Z[0]
    dr=R-R[0]
    return dict(
        cold_Z_relative_L2=float(np.linalg.norm(Z[0]-R[0])/
                                 max(np.linalg.norm(R[0]),1e-12)),
        thermal_delta_Z_relative_L2=float(np.linalg.norm(dz-dr)/
                                          max(np.linalg.norm(dr),1e-12)),
        absolute_Z_difference_over_thermal_delta=float(
            np.linalg.norm(Z-R)/max(np.linalg.norm(dr),1e-12)),
        sea_interpolated_max_abs_K=e_sea,
        package_max_abs_K=float(pkg),copper_max_abs_K=float(cu))


def pack_save(file,s):
    np.savez_compressed(file,time_s=s["time_s"],voltage_v=s["voltage_v"],
                        temp_K=s["temp_K"],z6=s["z6"],
                        sea_xyz=s["sea_xyz"],sea_vol=s["sea_vol"])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_native_sea_convergence.json"))
    p.add_argument("--postprocess",action="store_true",
                   help="Update convergence metrics using stored physical trajectories")
    p.add_argument("--full-surrogate",action="store_true",
                   help="Evaluate original-physics k6 surrogate at finest mesh")
    p.add_argument("--max-nodes",type=int,default=1400)
    a=p.parse_args()
    if a.postprocess:
        out=json.loads(a.out.read_text())
        for name in ("P03","P07"):
            runs={}
            for label in out["geometries"][name]["levels"]:
                file=a.runs/f"bfzi_pose_{name.lower()}_native_{label}.npz"
                with np.load(file) as d:
                    runs[label]=dict(
                        temp_K=d["temp_K"].copy(),z6=d["z6"].copy(),
                        sea_xyz=d["sea_xyz"].copy(),nsea=len(d["sea_xyz"]))
            fine=runs["face_fine"]
            for label,s in runs.items():
                out["geometries"][name]["levels"][label][
                    "vs_face_fine"]=compare(s,fine)
        a.out.write_text(json.dumps(out,indent=2))
        print("UPDATED",a.out,flush=True)
        return
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=net=None
    if a.full_surrogate:
        ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                      map_location="cpu",weights_only=True)
        net=GeomFrac(ng=int(ck["ng"]))
        net.load_state_dict(ck["state"])
        net.eval()
    out=dict(method="original BFZI native sea FV boundary-face convergence",
             geometry_training_used=False,
             interface="oriented physical face-resistance",
             h_W_m2K=20.,full_surrogate=bool(a.full_surrogate),
             geometries={})
    for name,cid,file in (
        ("P03",29,"bfzi_pose_p03_holdout.npz"),
        ("P07",18,"bfzi_pose_p07_15s_holdout.npz")):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        old=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        with np.load(a.runs/file) as d:
            cases=json.loads(str(d["cases_json"]))
            i=next(j for j,x in enumerate(cases) if int(x["id"])==cid)
            t=d["time_s"].copy()
            volt=d["voltage_v"][i].copy()
        geo=dict(original_program=cid,voltage_V=float(volt.max()),
                 time_points=len(t),levels={})
        runs={}
        for label,xz,y in GRIDS:
            ts=time.perf_counter()
            d,info=create(
                bf,v150,"M4",**build_args(meta),
                grid_xz=xz,grid_y=y,grid_layers=0,
                grid_max_nodes=a.max_nodes)
            em=BFZIV148Adapter(d,bf,heat_mode="port-weak")
            et=BFZIV148Pkg3D(
                em,external=old.ext,pkg_shape=(4,2,4),
                pkg_k=.25,pkg_sea_model="face-resistance",
                pkg_sea_h=20.,pkg_sea_quad=2,pkg_sea_near=4)
            head=GeomHeat.from_original(em,et,old.ext)
            cold_sec=time.perf_counter()-ts
            phys=direct(et,t,volt)
            file_direct=a.runs/f"bfzi_pose_{name.lower()}_native_{label}.npz"
            pack_save(file_direct,phys)
            diag=dict(grid_xz_m=xz,grid_y_m=y,
                      sea_cells=len(em.vol),
                      source_3D_volume_m3=float(np.sum(em.vol)),
                      mesh_rel_volume_error=float(
                          d["vol_meta"]["relative_volume_error"]),
                      cold_precompute_sec=cold_sec,
                      direct_time_sec=phys["elapsed_sec"],
                      direct_material_budget_W=phys["max_budget_W"],
                      peak_pkg_interface_flux_W=phys["max_pkg_flux_W"],
                      interface_near=values(old,em.points),
                      cold_z6=head.z0.tolist(),
                      physical_file=str(file_direct))
            if a.full_surrogate and label=="face_fine":
                rom=GeomHeatROM.build(head,rank=32)
                sea=basis(head.vol,rom.phi[:head.ns],rank=8)
                tangent,td=tangent_build(
                    em,head,rom,sea,6,amplitude=.2)
                ts=time.perf_counter()
                pred=rom.rollout(net,ck,volt,t,tangent=tangent,blend=1.)
                from examples.bfzi_pkg_interface_study import metrics
                diag["surrogate_k6"]=dict(
                    **metrics(pred,phys["temp_K"],phys["z6"],head),
                    inference_sec=time.perf_counter()-ts,
                    directional_EM_solves=tangent.nsolve,
                    directional_EM_sec=td["directional_sec"])
                tangent.save(a.runs/f"bfzi_pose_{name.lower()}_native_fine_k6.npz")
                head.save(a.runs/f"bfzi_pose_{name.lower()}_native_fine_anchor.npz")
                rom.save(a.runs/f"bfzi_pose_{name.lower()}_native_fine_rom.npz")
            geo["levels"][label]=diag
            runs[label]=phys
            print(name,label,"n",len(em.vol),
                  "cold",round(cold_sec,3),"direct",
                  round(phys["elapsed_sec"],3),"dZcold",
                  round(float(np.linalg.norm(head.z0-old.z0)),6),
                  flush=True)
        fine=runs["face_fine"]
        for label,s in runs.items():
            geo["levels"][label]["vs_face_fine"]=compare(s,fine)
        out["geometries"][name]=geo
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(out,indent=2))
        print("COMPARE",name,{label:round(
            geo["levels"][label]["vs_face_fine"][
                "dynamic_Z_relative_L2"]*100,5)
            for label in geo["levels"]},flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
