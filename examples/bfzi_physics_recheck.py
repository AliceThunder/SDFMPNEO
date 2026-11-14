"""Recheck BFZI V150 direct physics with one fixed material/skin model.

No surrogate is imported or evaluated.  Untrusted high-temperature runs are
recorded as out-of-domain, never silently used as reference truth.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import originals,create
from examples.bfzi_obb_layer_study import GRIDS as OLD_GRIDS

# F12 tilted/offset geometry needs finer seawater FVs for spatial closure.
GRIDS={**OLD_GRIDS,"fine":(.06,.05),"ultra":(.05,.04),
       "superfine":(.045,.035)}
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D


def build(data,bf,ext,opts):
    em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                      fv_faces="exact",source_mask="subcell")
    em.alpha=opts["alpha"]
    et=BFZIV148Skin3D(
        em,external=ext,pkg_shape=(4,2,4),pkg_k=.25,
        skin_depth=opts["skin"],skin_layers=8,skin_ratio=1.4,
        skin_quad=2,skin_near=4,skin_map=2,pkg_sea_h=20.,
        skin_heat="source",cu_ins_k=opts["ins_k"],cu_ins_t=opts["ins_t"],
        sea_max_K=opts["sea_max"],cu_max_K=opts["cu_max"],
        pkg_max_K=opts["pkg_max"])
    return et


def drive(spec,now):
    voltage=float(spec["voltage"])
    for t0,t1 in spec["off"]:
        if t0<=now<t1:
            voltage=0.
            break
    return np.array([voltage,0.],complex)


def march(et,spec,dt,method="be"):
    t=np.arange(0.,float(spec["duration"])+dt*.5,dt)
    if abs(t[-1]-spec["duration"])>1e-8:
        raise ValueError("time step must exactly divide experiment duration")
    theta=et.zeros()
    prev=theta.copy()
    zs=[];temps=[];ps=[];cs=[];power=[];curr=[];residual=[]
    therm=[];qrel=[]
    err=None
    start=time.perf_counter()
    for k,now in enumerate(t):
        v=drive(spec,now)
        prev_v=drive(spec,t[k-1]) if k else v
        try:
            if k:
                # Preserve the actual on/off duration at a switching time.
                theta=et.advance(prev,prev_v,dt,method=method)
            z,I,heat,q,cu=et.em(theta,v)
            # Source used by the implicit step and the endpoint source
            # differ exactly at a voltage transition. Keep the stored
            # energy/heat-budget check on the previous interval.
            step_heat=(heat if np.array_equal(prev_v,v)
                       else et.em(theta,prev_v)[2])
            if k:
                if method=="cn":
                    old_heat=et.em(prev,prev_v)[2]
                    used_heat=.5*(old_heat+step_heat)
                    avg_theta=.5*(prev+theta)
                else:
                    used_heat=step_heat
                    avg_theta=theta
        except (ValueError,RuntimeError) as exc:
            err=dict(time_s=float(now),reason=str(exc),
                     kind=("material" if "model exceeded" in str(exc)
                           else "solver"))
            break
        sea=float(q@et.model.vol)
        pext=float(.5*np.vdot(I,et.external@I).real)
        pleads=float(.5*np.dot(z.lead_z.real,abs(I)**2))
        pin=float(.5*np.vdot(v,I).real)
        pbal=pin-(sea+float(cu.sum())+pleads+pext)
        port_sea=float(.5*np.vdot(I,z.sea_z@I).real)
        qrel.append(abs(port_sea-sea)/max(abs(port_sea),1e-12))
        if k:
            bal=et.cap*(theta-prev)/dt+et.H@avg_theta-used_heat
            residual.append(float(np.max(abs(bal))))
            stored=float(et.cap@(theta-prev)/dt)
            ambient_loss=float(et.amb_g@avg_theta)
            therm.append([stored,ambient_loss,float(used_heat.sum()),
                          stored+ambient_loss-float(used_heat.sum())])
        else:
            residual.append(0.)
            therm.append([0.,0.,0.,0.])
        zs.append(pack(z.total_z))
        temps.append(et.amb+theta.copy())
        ps.append(sea)
        cs.append(cu.copy())
        curr.append(I.copy())
        power.append([pin,sea,float(cu.sum()),pleads,pext,pbal])
        prev=theta.copy()
    node=max(residual,default=0.)
    global_budget=max((abs(x[3]) for x in therm),default=0.)
    port_rel=max((abs(p[-1])/max(abs(p[0]),1e-12)
                  for p in power),default=0.)
    qmax=max(qrel,default=0.)
    status=("outside_material_domain" if err is not None and
            err["kind"]=="material" else "solver_error" if err is not None else
            "numerical_error" if node>1e-4 or global_budget>1e-4 or
                                  port_rel>1e-10 or qmax>1e-10 else
            "candidate")
    out=dict(status=status,
             failed_at=err,dt_s=float(dt),integrator=method,
             completed_s=float(t[len(zs)-1]) if len(zs) else None,
             elapsed_s=float(time.perf_counter()-start),
             copper_to_package_G_W_K=list(map(float,et.cu_pkg_total)),
             insulation_layer_m=list(map(float,et.cu_ins_layers)),
             skin_depth_m=float(et.skin_depth),
             skin_clip_fraction=float(et.skin_info["clipped_shell_fraction"]),
             skin_water_capacity_error_J_K=float(et.skin_info["sea_cap_delta_J_K"]),
             peak_thermal_residual_W=node,
             peak_global_budget_W=global_budget,
             max_port_power_residual_W=max((abs(p[-1]) for p in power),default=0.),
             max_rel_port_power=port_rel,max_rel_q_power=qmax,
             heat_cap_J_K=dict(
                 seawater=float(et.cap[:et.count].sum()+et.cap[et.skin].sum()),
                 package=float(et.cap[et.pkg].sum()),
                 copper=float(et.cap[et.cu].sum())))
    if len(zs):
        temp=np.asarray(temps)
        out.update(max_copper_K=float(temp[:,et.cu].max()),
                   max_package_K=float(temp[:,et.pkg].max()),
                   max_water_K=float(max(temp[:,:et.count].max(),
                                         temp[:,et.skin].max())),
                   max_copper_loss_W=float(np.max(np.sum(cs,axis=1))),
                   max_lead_loss_W=float(np.max(np.asarray(power)[:,3])),
                   max_sea_loss_W=float(np.max(ps)),
                   max_source_power_W=float(np.max(np.asarray(power)[:,0])),
                   peak_ambient_loss_W=float(np.max(np.asarray(therm)[:,1])),
                   max_storage_rate_W=float(np.max(np.asarray(therm)[:,0])))
    return out,dict(time_s=t[:len(zs)],temp_K=np.asarray(temps),
                    Tcu_K=np.asarray(temps)[:,et.cu],
                    Tpkg_K=np.asarray(temps)[:,et.pkg],
                    z6=np.asarray(zs),Psea_W=np.asarray(ps),
                    Pcu_W=np.asarray(cs),power_W=np.asarray(power),
                    thermal_W=np.asarray(therm),currents_A=np.asarray(curr))


def compare(coarse,fine):
    if not len(coarse["time_s"]) or not len(fine["time_s"]):
        return None
    a=np.asarray(coarse["time_s"])
    b=np.asarray(fine["time_s"])
    until=float(min(a[-1],b[-1]))
    ai=np.flatnonzero(a<=until+1e-8)
    if len(ai)<2:
        return None
    ids=np.searchsorted(b,a[ai])
    if np.any(ids>=len(b)) or not np.allclose(a[ai],b[ids],atol=1e-8):
        return None
    z0=coarse["z6"][ai]-coarse["z6"][0]
    z1=fine["z6"][ids]-fine["z6"][0]
    return dict(shared_until_s=until,shared_points=int(len(ai)),
                prefix_only=bool(a[-1]!=b[-1]),
                delta_Z_relative_L2=float(
                    np.linalg.norm(z0-z1)/max(np.linalg.norm(z1),1e-30)),
                copper_abs_K=float(np.max(abs(
                    coarse["Tcu_K"][ai]-fine["Tcu_K"][ids]))),
                package_abs_K=float(np.max(abs(
                    coarse["Tpkg_K"][ai]-fine["Tpkg_K"][ids]))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--spec-file",type=Path,default=None,
                   help="Additional independent direct-physics case specs")
    p.add_argument("--keys",nargs="+",default=["F12","F14","F16"])
    p.add_argument("--dt",type=float,nargs="+",default=[30.,15.])
    p.add_argument("--grids",nargs="+",choices=tuple(GRIDS),
                   default=["medium","extra"])
    p.add_argument("--grid-max-nodes",type=int,default=4000)
    p.add_argument("--integrator",choices=("be","cn"),default="be",
                   help="Implicit Euler or energy-conserving trapezoidal thermal step")
    p.add_argument("--sea-max-K",type=float,default=373.15)
    p.add_argument("--cu-max-K",type=float,default=1357.77)
    p.add_argument("--pkg-max-K",type=float,default=None)
    p.add_argument("--ins-k",type=float,default=.25)
    p.add_argument("--ins-t",type=float,default=None,
                   help="Radial insulation thickness; default derives from original BFZI wire")
    p.add_argument("--skin",type=float,default=.002)
    p.add_argument("--out",type=Path,default=None)
    a=p.parse_args()
    if (min(a.ins_k,a.skin,*a.dt)<=0 or a.grid_max_nodes<1 or
            (a.ins_t is not None and a.ins_t<=0)):
        p.error("insulation and time-step parameters must be positive")
    bf,v150=originals(a.bfzi_root)
    saved=json.loads((a.runs/"bfzi_final_accept.json").read_text())
    selected={key:item["spec"] for key,item in saved["cases"].items()}
    if a.spec_file is not None:
        selected.update(json.loads(a.spec_file.read_text())["cases"])
    results={}
    for key in a.keys:
        if key not in selected:
            p.error(f"unknown case: {key}")
        spec=selected[key]
        parent="p03" if spec["donor"]=="g1" else "p07"
        ext=GeomHeat.load(a.runs/f"bfzi_pose_{parent}_disjoint_anchor.npz").ext
        rows={}
        history={}
        for grid in a.grids:
            xz,y=GRIDS[grid]
            data,_=create(bf,v150,"M4",**spec["geometry"],
                          sea_void="oriented",physical_box_only=True,
                          grid_xz=xz,grid_y=y,
                          grid_max_nodes=a.grid_max_nodes)
            opts=dict(alpha=spec["alpha_per_K"],skin=a.skin,
                      ins_k=a.ins_k,ins_t=a.ins_t,
                      sea_max=a.sea_max_K,cu_max=a.cu_max_K,
                      pkg_max=a.pkg_max_K)
            try:
                et=build(data,bf,ext,opts)
            except ValueError as exc:
                rows[grid]=dict(status="invalid_spatial_grid",reason=str(exc))
                continue
            for dt in a.dt:
                tag=f"{grid}_dt{dt:g}"
                rec,raw=march(et,spec,dt,method=a.integrator)
                rows[tag]=rec
                history[tag]=raw
                if len(raw["time_s"]):
                    np.savez_compressed(
                        a.runs/f"bfzi_phys_{key}_{tag}"+
                        ("_cn.npz" if a.integrator=="cn" else ".npz"),
                        **raw)
                print("PHYS",key,tag,rec["status"],
                      rec["completed_s"],rec.get("max_copper_K"),
                      flush=True)
        checks={}
        for grid in a.grids:
            if len(a.dt)<2:
                continue
            c=history.get(f"{grid}_dt{a.dt[0]:g}")
            f=history.get(f"{grid}_dt{a.dt[1]:g}")
            if c is not None and f is not None:
                # Compare only valid shared history before any phase-limit
                # crossing; this does not turn the full case into a pass.
                checks[grid]=compare(c,f)
        if len(a.grids)==2:
            g0,g1=a.grids
            for dt in a.dt:
                med=f"{g0}_dt{dt:g}"
                ex=f"{g1}_dt{dt:g}"
                if med in history and ex in history:
                    checks[f"space_dt{dt:g}"]=compare(
                        history[med],history[ex])
        # A physically valid prefix remains diagnostic until the full drive
        # and all requested resolution checks meet the locked gate.
        full=all(r.get("status")=="candidate" for r in rows.values())
        def gate(v,kind):
            if v is None:
                return False
            zlim,clim,plim=((.01,.5,.5) if kind=="time"
                            else (.02,1.,1.5))
            return (v["delta_Z_relative_L2"]<=zlim and
                    v["copper_abs_K"]<=clim and
                    v["package_abs_K"]<=plim and
                    not v["prefix_only"])
        time_ok=(len(a.dt)>=2 and
                 all(gate(checks.get(g),"time") for g in a.grids))
        space_ok=(len(a.grids)>=2 and
                  all(gate(checks.get(f"space_dt{dt:g}"),"space")
                      for dt in a.dt))
        accepted=bool(full and time_ok and space_ok)
        results[key]=dict(runs=rows,comparisons=checks,
                          physical_full_horizon=bool(full),
                          time_convergence_pass=bool(time_ok),
                          space_convergence_pass=bool(space_ok),
                          accepted=accepted)
        dest=a.out or a.runs/"bfzi_physics_recheck.json"
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(dict(
            method="BFZI V150 corrected copper geometry; fixed skin depth",
            settings=dict(spec_file=str(a.spec_file) if a.spec_file else None,
                          integrator=a.integrator,
                          dt=a.dt,grids=a.grids,
                          grid_sizes_m={g:list(GRIDS[g]) for g in a.grids},
                          grid_max_nodes=a.grid_max_nodes,skin_m=a.skin,
                          ins_k=a.ins_k,ins_t=a.ins_t,
                          sea_max_K=a.sea_max_K,cu_max_K=a.cu_max_K,
                          pkg_max_K=a.pkg_max_K),
            cases=results),indent=2)+"\n")
    print("FINISHED",len(results))


if __name__=="__main__":
    main()
