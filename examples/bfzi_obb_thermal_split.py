"""Separate BFZI 3-D thermal-grid error from electromagnetic source error.

Use cold unit-power copper and sea heat drives, fixed for the entire
physical 600s program. The same two sealed oriented packages and original
BFZI V150 FV operators are used at 572 and 1312 wet-cell resolutions.
No warm EM solve or neural model is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import factorized

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.v148_pkg3d import BFZIV148Pkg3D
from sdfmpneo_bfzi.geom_heat import GeomHeat

GRIDS={"medium":(.14,.12),"extra":(.075,.06)}


def thermal(head,t,force):
    cap=head.cap
    H=head.H
    state=np.zeros(len(cap))
    hist=[state.copy()]
    dt_prev=None
    for dt in np.diff(t):
        dt=float(dt)
        if dt!=dt_prev:
            lu=factorized((diags(cap)+dt*H).tocsc())
            dt_prev=dt
        state=lu(cap*state+dt*force)
        hist.append(state.copy())
    return np.asarray(hist)


def summary(T,head):
    ns=head.ns
    pack=T[:,ns:-2]
    cu=T[:,-2:]
    return dict(
        package_final_K=float(np.max(pack[-1])),
        package_peak_K=float(np.max(pack)),
        copper_final_K=float(np.max(cu[-1])),
        copper_peak_K=float(np.max(cu)),
        sea_peak_K=float(np.max(T[:,:ns])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--source-mask",choices=("legacy","oriented","physical"),
                   default="legacy")
    p.add_argument("--source-radius",type=float,default=.06)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_obb_thermal_split.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    study=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    out=dict(experiment="Original BFZI constant unit-power thermal forcing",
             source_mask=a.source_mask,source_radius_m=a.source_radius,
             material_sea_k_W_mK=.6,material_sea_rhoc_J_m3K=4e6,
             diffusion_length_600s_m=float(np.sqrt(.6/4e6*600)),
             geometries={})
    for name in ("P03","P07"):
        meta=json.loads((a.runs/f"bfzi_pose_{name.lower()}_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(
            study["geometries"][name]["new_shift_mm"],float)
        prev=GeomHeat.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        t=np.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz")["time_s"]
        r={}
        hist={}
        for label,(xz,y) in GRIDS.items():
            data,_=create(bf,v150,"M4",**geo,
                          sea_void="oriented",physical_box_only=True,
                          grid_xz=xz,grid_y=y)
            em=BFZIV148Adapter(
                data,bf,heat_mode="port-weak",fv_faces="exact",
                source_mask=a.source_mask,source_radius=a.source_radius)
            et=BFZIV148Pkg3D(
                em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
                pkg_sea_model="face-resistance",
                pkg_sea_h=20.,pkg_sea_quad=2,pkg_sea_near=4)
            head=GeomHeat.from_original(em,et,prev.ext)
            sea,_,I=head.circuit(np.zeros(len(head.cap)),
                                 np.zeros((2,2),complex),100.)
            _,q,_,psea,_=head.source(np.zeros(len(head.cap)),I,sea)
            forces=dict(
                copper=np.zeros(len(head.cap)),
                seawater=np.zeros(len(head.cap)))
            forces["copper"][head.cu]=1.
            forces["seawater"][:head.ns]=q*head.vol/max(psea,1e-30)
            if not np.isclose(sum(forces["seawater"]),1.,atol=1e-12):
                raise ValueError("sea Joule power must integrate to 1 W")
            outmat={}
            for key,f in forces.items():
                z=thermal(head,t,f)
                outmat[key]=z
            hist[label]=outmat
            local=[]
            pts=head.xyz[:head.ns]
            for mid,half,rot in zip(head._pkg_mid,head._pkg_half,head._pkg_axes):
                v=(pts-mid)@rot
                local.append(np.linalg.norm(np.maximum(abs(v)-half,0),axis=1))
            near=np.min(local,axis=0)<.02
            r[label]=dict(
                sea_cells=head.ns,
                pkg_G_W_K=[float(sum(x[2] for x in li))
                           for li in et.pkg_links],
                sea_power_near_20mm=float(sum(q[near]*head.vol[near])/psea),
                sea_power_mean_xyz_m=(np.sum((q*head.vol)[:,None]*pts,axis=0)/
                                      psea).tolist(),
                sea_power_total_W=float(sum(q*head.vol)),
                copper=summary(outmat["copper"],head),
                seawater=summary(outmat["seawater"],head))
            print(name,label,"sea_n",head.ns,"G",
                  [round(z,5) for z in r[label]["pkg_G_W_K"]],
                  "20mm_power_frac",round(r[label]["sea_power_near_20mm"],5),
                  "cu_pkg_K",round(r[label]["copper"]["package_peak_K"],5),
                  "sea_pkg_K",round(r[label]["seawater"]["package_peak_K"],5),
                  flush=True)
        comp={}
        for key in ("copper","seawater"):
            A=hist["medium"][key]
            B=hist["extra"][key]
            # Package/Cu states are the same independent physical box grid.
            ns=r["medium"]["sea_cells"]
            nr=r["extra"]["sea_cells"]
            comp[key]=dict(
                package_max_K=float(np.max(abs(A[:,ns:-2]-B[:,nr:-2]))),
                copper_max_K=float(np.max(abs(A[:,-2:]-B[:,-2:]))),
                package_final_K=float(np.max(abs(A[-1,ns:-2]-B[-1,nr:-2]))))
        out["geometries"][name]=dict(levels=r,
                                    medium_vs_extra=comp)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(out,indent=2))
        print("CROSS",name,comp,flush=True)
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
