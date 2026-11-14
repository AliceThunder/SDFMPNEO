"""Isolate original BFZI sea electromagnetic mesh sensitivity from heating.

Apply identical physical sigma(T) fields to 572/1312 OBB sea meshes, with
no heat equation and no neural model. Compare only thermally induced
original V150 complex two-port impedance changes, subtracting each
mesh's own cold result. This separates EM FV/Green discretization error
from copper/package/water thermal dynamics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat, pack


def pattern(pts, head, scale):
    if scale is None:
        return np.ones(len(pts))
    gap=np.full(len(pts),np.inf)
    for center,half,rot in zip(
            head._pkg_mid,head._pkg_half,head._pkg_axes):
        loc=(pts-center)@rot
        gap=np.minimum(gap,np.linalg.norm(
            np.maximum(abs(loc)-half,0),axis=1))
    return np.exp(-.5*(gap/scale)**2)


def mean_pattern(rec,head,scale):
    """Volume integral of the same analytic T over original wet FV quadrature."""
    values=[]
    for cell in rec:
        q=np.asarray(cell["quadrature_points"],float)
        w=np.asarray(cell["quadrature_weights"],float)
        if len(q)==0 or w.sum()<=0:
            raise ValueError("BFZI oriented wet FV quadrature has no volume")
        values.append(float(w@pattern(q,head,scale)/w.sum()))
    return np.asarray(values)


def calc(em,amb,head,integrate=False):
    cold=em.solve(np.full(len(em.vol),amb),np.full(2,amb))
    cs=pack(cold.sea_z)
    ct=pack(cold.total_z)
    vals={}
    modes=[("uniform",None),("layer_3mm",.003),
           ("layer_10mm",.01),("layer_30mm",.03),
           ("layer_80mm",.08)]
    if integrate:
        modes += [("integral_3mm",.003),("integral_10mm",.01),
                  ("integral_30mm",.03),("integral_80mm",.08)]
    for name,scale in modes:
        if name.startswith("integral_"):
            theta=mean_pattern(em.base["records"],head,scale)*5.
        else:
            theta=pattern(em.points,head,scale)*5.
        warm=em.solve(amb+theta,np.full(2,amb))
        ds=pack(warm.sea_z)-cs
        dz=pack(warm.total_z)-ct
        # Original BFZI sea_z = bulk Green response + additive near-
        # source mass moment. Resolve these two contributions exactly
        # algebraically, with no additional EM/GMRES solves.
        dm=((em.omega*em.mu)**2*em.sigma0*em.alpha*
            np.einsum("n,nij->ij",theta,em.mass,optimize=True))
        dmass=pack(dm)
        dcore=ds-dmass
        vals[name]=dict(sea_delta_z6=ds.tolist(),
                        total_delta_z6=dz.tolist(),
                        core_delta_z6=dcore.tolist(),
                        mass_delta_z6=dmass.tolist(),
                        peak_K=float(theta.max()),
                        vol_mean_K=float(theta@em.vol/em.vol.sum()))
    cu=em.solve(np.full(len(em.vol),amb),np.full(2,amb+5.))
    vals["copper_only"]=dict(
        sea_delta_z6=(pack(cu.sea_z)-cs).tolist(),
        total_delta_z6=(pack(cu.total_z)-ct).tolist(),
        core_delta_z6=(pack(cu.sea_z)-cs).tolist(),
        mass_delta_z6=np.zeros(6).tolist(),
        peak_K=5.,vol_mean_K=0.)
    return vals


def error(x,y):
    a=np.asarray(x,float)
    b=np.asarray(y,float)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-20))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--integrate-temp",action="store_true",
                   help="Integrate analytic sigma(T) over true oriented wet FV quadrature")
    p.add_argument("--source-mask",choices=("legacy","oriented",
                    "physical","all"),default="legacy")
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),
                   default=["P03","P07"])
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_obb_em_grid_probe.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    study=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    out=dict(method="original BFZI V150 fixed analytic T sea impedance mesh experiment",
             no_thermal_solver=True,reference="1312 true OBB water FV",
             source_mask=a.source_mask,
             integrate_temperature=a.integrate_temp,
             geometries={})
    for name in a.poses:
        meta=json.loads((a.runs/
            f"bfzi_pose_{name.lower()}_anchor.json").read_text())
        head=GeomHeat.load(a.runs/
            f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(
            study["geometries"][name]["new_shift_mm"],float)
        run={}
        for level,(xz,y) in (("medium",(.14,.12)),
                             ("extra",(.075,.06))):
            data,_=create(bf,v150,"M4",**geo,
                          sea_void="oriented",physical_box_only=True,
                          grid_xz=xz,grid_y=y)
            em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                               fv_faces="exact",source_mask=a.source_mask)
            vals=calc(em,head.amb,head,integrate=a.integrate_temp)
            run[level]=dict(nodes=len(em.vol),fields=vals)
            print(name,level,"nodes",len(em.vol),flush=True)
        compare={}
        for key in run["medium"]["fields"]:
            low=run["medium"]["fields"][key]
            high=run["extra"]["fields"][key]
            compare[key]=dict(
                sea_delta_Z_rel=error(low["sea_delta_z6"],
                                      high["sea_delta_z6"]),
                total_delta_Z_rel=error(low["total_delta_z6"],
                                        high["total_delta_z6"]),
                bulk_Green_delta_Z_rel=error(low["core_delta_z6"],
                                             high["core_delta_z6"]),
                local_mass_delta_Z_rel=error(low["mass_delta_z6"],
                                             high["mass_delta_z6"]),
                mass_fraction_high=float(
                    np.linalg.norm(high["mass_delta_z6"])/
                    max(np.linalg.norm(high["sea_delta_z6"]),1e-20)),
                mean_temperature_low_K=low["vol_mean_K"],
                mean_temperature_high_K=high["vol_mean_K"])
        out["geometries"][name]=dict(levels=run,
                                     low_vs_high=compare)
        print("PROBE",name,[
            (key,round(v["sea_delta_Z_rel"]*100,4),
             round(v["bulk_Green_delta_Z_rel"]*100,4),
             round(v["local_mass_delta_Z_rel"]*100,4))
            for key,v in compare.items()],flush=True)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
