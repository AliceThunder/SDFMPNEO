"""Check whether existing BFZI thermal EM probes can also predict warm J/q.

Uses exactly the same 6 sea-temperature perturbation directions as the
deployed impedance tangent; no extra direction solves. True heldout
temperature and original BFZI warm fields are used solely for evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.geom_void import disjoint_y_shift


def rel(x,y,vol):
    x,y=np.asarray(x),np.asarray(y)
    w=vol.reshape(-1,*([1]*(x.ndim-1)))
    return float(np.sqrt(np.sum(w*abs(x-y)**2)/
                         max(np.sum(w*abs(y)**2),1e-30)))


def q_from(weak,I,vol,psea):
    shape=.5*np.real(np.einsum(
        "i,pij,j->p",np.conj(I),weak,I,optimize=True))
    shape=np.maximum(shape,0.)
    return shape*(psea/max(float(shape@vol),1e-30))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_obb_field_tangent.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    out={}
    for name in ("P03","P07"):
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        old=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(geo["shift_mm"],float)
        geo["shift_mm"][1]+=1000*disjoint_y_shift(
            old._pkg_mid,old._pkg_half,old._pkg_axes)
        d,_=create(bf,v150,"M4",**geo,sea_void="oriented",
                   physical_box_only=True,grid_xz=.14,grid_y=.12)
        em=BFZIV148Adapter(d,bf,heat_mode="port-weak")
        head=GeomHeat.load(a.runs/
            f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        tangent=SeaTangent.load(a.runs/
            f"bfzi_pose_{name.lower()}_disjoint_k6.npz",head)
        cold=em.solve(np.full(head.ns,head.amb),np.full(2,head.amb))
        dj=[];dq=[]
        for i in range(tangent.modes.shape[1]):
            h=.2/max(float(np.max(abs(tangent.modes[:,i]))),1e-12)
            state=em.solve(head.amb+h*tangent.modes[:,i]+
                           np.zeros(head.ns),np.full(2,head.amb))
            dj.append((state.unit_j-cold.unit_j)/h)
            dq.append((state.weak_q-cold.weak_q)/h)
        dj=np.asarray(dj)
        dq=np.asarray(dq)
        with np.load(a.runs/
             f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz") as file:
            t=file["time_s"].copy()
            T=file["temp_K"].copy()
            v=file["voltage_v"].copy()
        selected=np.flatnonzero(v>0)
        picks=selected[np.unique(np.linspace(
            0,len(selected)-1,min(6,len(selected))).astype(int))]
        rows=[]
        for j in picks:
            warm=em.solve(T[j,:head.ns],T[j,head.cu])
            I=np.linalg.solve(warm.total_z+head.ext,np.array([v[j],0],complex))
            theta=T[j,:head.ns]-head.amb
            coeff=tangent.modes.T@(head.vol*theta)
            jhat=cold.unit_j+np.einsum("k,knpc->npc",coeff,dj,optimize=True)
            qhat=cold.weak_q+np.einsum("k,knab->nab",coeff,dq,optimize=True)
            psea=float(.5*np.vdot(I,warm.sea_z@I).real)
            real_j=np.einsum("npc,p->nc",warm.unit_j,I,optimize=True)
            base_j=np.einsum("npc,p->nc",head.unit,I,optimize=True)*(
                1+head.alpha*theta)[:,None]
            pred_j=np.einsum("npc,p->nc",jhat,I,optimize=True)
            qref,_=em.heat(warm,I)
            _,base_q,_,_,_=head.source(T[j]-head.amb,I,warm.sea_z)
            qlin=q_from(qhat,I,head.vol,psea)
            dz=warm.sea_z-cold.sea_z
            dzlin=tangent.delta(theta)
            rows.append(dict(
                t_s=float(t[j]),voltage_V=float(v[j]),
                max_sea_K=float(np.max(theta)),
                Z_thermal_lin_relative_L2=float(np.linalg.norm(dzlin-dz)/
                                  max(np.linalg.norm(dz),1e-12)),
                J_cold_relative_L2=rel(base_j,real_j,head.vol),
                J_tangent_relative_L2=rel(pred_j,real_j,head.vol),
                q_cold_relative_L2=rel(base_q,qref,head.vol),
                q_tangent_relative_L2=rel(qlin,qref,head.vol),
                q_tangent_port_gap_W=float(abs(qlin@head.vol-psea))))
        out[name]=dict(EM_directional_solves=len(dj),samples=rows)
        print(name,[(int(x["t_s"]),
                    round(x["Z_thermal_lin_relative_L2"]*100,4),
                    round(x["J_cold_relative_L2"]*100,4),
                    round(x["J_tangent_relative_L2"]*100,4),
                    round(x["q_cold_relative_L2"]*100,4),
                    round(x["q_tangent_relative_L2"]*100,4))
                    for x in rows],flush=True)
    a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
