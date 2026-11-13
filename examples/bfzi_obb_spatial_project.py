"""Test temperature-dependent charge-conserving FV projection of cold J.

Uses original BFZI WeightedFV on the corrected oriented FV grid without
resolving the Green/GMRES operator for candidate surrogate J. Independent
warm original BFZI EM fields provide spatial reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_void import disjoint_y_shift
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter,WeightedFV


def relative(x,y,vol):
    w=vol.reshape(-1,*([1]*(x.ndim-1)))
    return float(np.sqrt(np.sum(w*abs(x-y)**2)/
                         max(np.sum(w*abs(y)**2),1e-30)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_obb_spatial_project.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    out={}
    for name in ("P03","P07"):
        oldpre=a.runs/f"bfzi_pose_{name.lower()}"
        old=GeomHeat.load(oldpre.with_name(oldpre.name+"_anchor.npz"))
        meta=json.loads(oldpre.with_name(oldpre.name+"_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(geo["shift_mm"],float)
        geo["shift_mm"][1]+=1000*disjoint_y_shift(
            old._pkg_mid,old._pkg_half,old._pkg_axes)
        d,_=create(bf,v150,"M4",**geo,
                   sea_void="oriented",physical_box_only=True,
                   grid_xz=.14,grid_y=.12)
        em=BFZIV148Adapter(d,bf,heat_mode="port-weak")
        head=GeomHeat.load(a.runs/
                 f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        cold=em.solve(np.full(head.ns,head.amb),np.full(2,head.amb))
        p0=cold.unit_j/(-1j*em.omega*em.mu*em.sigma0)
        cold_proj=WeightedFV(
            em.proj,np.full(head.ns,em.sigma0)).project(
            [p0[:,i] for i in range(2)])
        with np.load(a.runs/
             f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz") as s:
            t=s["time_s"].copy()
            T=s["temp_K"].copy()
            v=s["voltage_v"].copy()
        picks=np.unique([0,len(t)//4,len(t)//2,len(t)-1])
        rows=[]
        for j in picks:
            state=em.solve(T[j,:head.ns],T[j,head.cu])
            I=np.linalg.solve(
                state.total_z+head.ext,
                np.array([v[j],0],complex))
            target=np.einsum("npc,p->nc",state.unit_j,I,optimize=True)
            _,qbase,Jbase,psea,_=head.source(
                T[j]-head.amb,I,state.sea_z)
            sigma=em.sigma0*(1+em.alpha*(T[j,:head.ns]-head.amb))
            probe=WeightedFV(em.proj,sigma)
            fields=probe.project([p0[:,i] for i in range(2)])
            unit=np.stack([
                cold.unit_j[:,i]*(sigma/em.sigma0)[:,None]
                -1j*em.omega*em.mu*sigma[:,None]*
                    (fields[i]-cold_proj[i])
                for i in range(2)],axis=1)
            Jnew=np.einsum("npc,p->nc",unit,I,optimize=True)
            eps=np.maximum(np.sum(abs(Jbase)**2,axis=1),1e-30)
            weight=np.sum(abs(Jnew)**2,axis=1)/eps
            qnew=qbase*weight
            qnew=qnew*psea/max(float(qnew@head.vol),1e-30)
            qref,_=em.heat(state,I)
            rows.append(dict(
                t_s=float(t[j]),V=float(v[j]),
                coldJ_error=relative(Jbase,target,head.vol),
                weightedFV_J_error=relative(Jnew,target,head.vol),
                coldq_error=relative(qbase,qref,head.vol),
                weightedFV_q_error=relative(qnew,qref,head.vol),
                max_temp_rise_K=float(np.max(T[j]-head.amb))))
        out[name]=rows
        print(name,[(round(x["t_s"]),round(x["coldJ_error"]*100,5),
                     round(x["weightedFV_J_error"]*100,5),
                     round(x["coldq_error"]*100,5),
                     round(x["weightedFV_q_error"]*100,5))
                    for x in rows],flush=True)
    a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
