"""Original BFZI V150 finite-depth seawater skin: real OBB convergence study.

The original BFZI EM cell grid, Green, ports and weak Joule field remain
unchanged. Skin FV states split and conserve the existing wet-cell heat
capacity and weak Joule power. Run P03/P07 572/1312-cell thermal and full
temperature-dependent BFZI direct histories.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create,originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_refine import load
from examples.bfzi_obb_thermal_split import thermal
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D

GRIDS={"medium":(.14,.12),"extra":(.075,.06)}


def view(t,et):
    """Independent physical copper / package states for a given mesh."""
    n=et.count
    return t[:,n:n+2*et.pkg_n],t[:,et.cu]


def compare(a,b):
    """Same physical package and winding nodes across wet FV resolutions."""
    ta,za=a["temp_K"],a["z6"]
    tb,zb=b["temp_K"],b["z6"]
    pa,ca=view(ta,a["et"])
    pb,cb=view(tb,b["et"])
    return dict(
        dynamic_delta_Z_relative_L2=float(
            np.linalg.norm((za-za[0])-(zb-zb[0]))/
            max(np.linalg.norm(zb-zb[0]),1e-30)),
        full_Z_relative_L2=float(
            np.linalg.norm(za-zb)/max(np.linalg.norm(zb),1e-30)),
        pkg_max_abs_K=float(np.max(abs(pa-pb))),
        copper_max_abs_K=float(np.max(abs(ca-cb))),
        pkg_final_max_abs_K=float(np.max(abs(pa[-1]-pb[-1]))))


def cold_force(et):
    """Unit copper and sea power, with exact redistributed BFZI Joule heat."""
    _,_,_,q,_=et.em(et.zeros(),np.array([100.,0],complex))
    p=float(q@et.model.vol)
    cu=np.zeros(et.n)
    cu[et.cu]=1.
    sea=et.heat_vec(q/p,np.zeros(2))
    if not np.isclose(sea.sum(),1.,atol=1e-12):
        raise ValueError("BFZI sea weak Joule power must be 1W")
    return cu,sea


def run(et,t,volt):
    t0=time.perf_counter()
    def drive(now):
        j=int(np.argmin(abs(t-now)))
        return np.array([volt[j],0],complex)
    seq=et.trajectory(t,drive)
    elapsed=time.perf_counter()-t0
    temp=et.amb+np.array([v.temp for v in seq])
    z=np.array([pack(v.impedance) for v in seq])
    budget=0.
    for i in range(1,len(seq)):
        dt=t[i]-t[i-1]
        heat=et.heat_vec(seq[i].sea_q,seq[i].copper_power)
        res=(et.cap*(seq[i].temp-seq[i-1].temp)/dt+
             et.H@seq[i].temp-heat)
        budget=max(budget,float(np.max(abs(res))))
    return dict(temp_K=temp,z6=z,
                max_node_budget_W=budget,direct_sec=elapsed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--poses",nargs="+",choices=("P03","P07"),
                   default=["P03","P07"])
    p.add_argument("--levels",nargs="+",choices=tuple(GRIDS),
                   default=["medium","extra"])
    p.add_argument("--depth",type=float,default=.002)
    p.add_argument("--layers",type=int,default=8)
    p.add_argument("--ratio",type=float,default=1.4)
    p.add_argument("--skin-map",type=int,choices=(1,2),default=2)
    p.add_argument("--gap-bridge",action="store_true")
    p.add_argument("--gap-max",type=float,default=.012)
    p.add_argument("--gap-quad",type=int,default=4)
    p.add_argument("--gap-align",type=float,default=.5)
    p.add_argument("--thermal-only",action="store_true")
    p.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_obb_layer.json"))
    a=p.parse_args()
    bf,v150=originals(a.bfzi_root)
    study=json.loads((a.runs/"bfzi_obb_physical_study.json").read_text())
    out=dict(method="original BFZI V150 passive conservative finite sea skin",
             skin_depth_m=a.depth,skin_layers=a.layers,skin_ratio=a.ratio,
             skin_map_order=a.skin_map,
             gap_bridge=a.gap_bridge,gap_max_m=a.gap_max,
             gap_quad=a.gap_quad,gap_align=a.gap_align,
             skin_q_remapped=True,sea_volume_conserved=True,
             geometries={})
    for name in a.poses:
        meta=json.loads((a.runs/f"bfzi_pose_{name.lower()}_anchor.json").read_text())
        geo=build_args(meta)
        geo["shift_mm"]=np.asarray(
            study["geometries"][name]["new_shift_mm"],float)
        prev=GeomHeat.load(
            a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        prog=load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz")
        g=dict(levels={})
        measured={}
        for label in a.levels:
            xz,y=GRIDS[label]
            tic=time.perf_counter()
            data,_=create(bf,v150,"M4",**geo,
                          sea_void="oriented",physical_box_only=True,
                          grid_xz=xz,grid_y=y)
            em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                               fv_faces="exact",source_mask="legacy")
            et=BFZIV148Skin3D(
                em,external=prev.ext,pkg_shape=(4,2,4),pkg_k=.25,
                skin_depth=a.depth,skin_layers=a.layers,
                skin_ratio=a.ratio,skin_quad=2,skin_near=4,
                skin_map=a.skin_map,gap_bridge=a.gap_bridge,
                gap_max=a.gap_max,gap_quad=a.gap_quad,
                gap_align=a.gap_align,pkg_sea_h=20.)
            setup=time.perf_counter()-tic
            cu,sea=cold_force(et)
            unit={}
            for kind,src in (("copper",cu),("seawater",sea)):
                T=thermal(et,prog["time_s"],src)
                pkg,coil=view(T,et)
                unit[kind]=dict(
                    pkg_final_K=float(np.max(pkg[-1])),
                    pkg_peak_K=float(np.max(pkg)),
                    copper_final_K=float(np.max(coil[-1])),
                    copper_peak_K=float(np.max(coil)))
                measured[(label,kind)]=(pkg,coil)
            row=dict(nsea=et.count,skin_states=et.skin.stop-et.skin.start,
                     original_em_nodes=len(em.vol),setup_sec=setup,
                     skin_info=et.skin_info,
                     unit_power=unit)
            if not a.thermal_only:
                pred=run(et,prog["time_s"],prog["voltage_v"])
                tag="bridge" if a.gap_bridge else "layer"
                f=a.runs/f"bfzi_pose_{name.lower()}_{tag}_{label}.npz"
                np.savez_compressed(
                    f,time_s=prog["time_s"],voltage_v=prog["voltage_v"],
                    temp_K=pred["temp_K"],z6=pred["z6"],
                    sea_xyz=em.points,sea_vol=em.vol,
                    bulk_vol=et.bulk_vol,skin_owner=et.skin_owner,
                    skin_vol=et.skin_vol,n_pkg=et.pkg_n)
                row.update(physical_file=str(f),
                           direct_sec=pred["direct_sec"],
                           material_node_residual_W=pred["max_node_budget_W"])
                base=load(a.runs/f"bfzi_pose_{name.lower()}_exact_{label}.npz")
                # Archived reference contains the same original physical
                # package and winding nodes, with no skin tail.
                Tb=base["temp_K"]
                z=base["z6"]
                pb=Tb[:,base["nsea"]:base["nsea"]+2*et.pkg_n]
                cb=Tb[:,base["nsea"]+2*et.pkg_n:
                      base["nsea"]+2*et.pkg_n+2]
                ps,cs=view(pred["temp_K"],et)
                row["vs_no_layer"]=dict(
                    dynamic_delta_Z_relative_L2=float(
                        np.linalg.norm((pred["z6"]-pred["z6"][0])-
                                       (z-z[0]))/
                        max(np.linalg.norm(z-z[0]),1e-30)),
                    pkg_max_abs_K=float(np.max(abs(ps-pb))),
                    copper_max_abs_K=float(np.max(abs(cs-cb))))
                pred["et"]=et
                measured[(label,"dynamic")]=pred
            g["levels"][label]=row
            print(name,label,"sea",et.count,
                  "skin",row["skin_states"],"maxfrac",
                  round(et.skin_info["max_wet_fraction"],5),
                  "unitCuPkg",round(unit["copper"]["pkg_final_K"],6),
                  "unitSeaPkg",round(unit["seawater"]["pkg_final_K"],6),
                  "bridge",et.skin_info["gap"]["bridges"],
                  "bridge_G",round(et.skin_info["gap"]["G_W_K"],6),
                  "time",round(setup,2),flush=True)
        if all(key in g["levels"] for key in GRIDS):
            m,f="medium","extra"
            g["cross_grid"]={}
            for kind in ("copper","seawater"):
                ap,ac=measured[(m,kind)]
                bp,bc=measured[(f,kind)]
                g["cross_grid"][kind]=dict(
                    pkg_max_abs_K=float(np.max(abs(ap-bp))),
                    copper_max_abs_K=float(np.max(abs(ac-bc))))
            if not a.thermal_only:
                g["cross_grid"]["dynamic"]=compare(
                    measured[(m,"dynamic")],measured[(f,"dynamic")])
            print("CROSS",name,json.dumps(g["cross_grid"]),flush=True)
        out["geometries"][name]=g
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
