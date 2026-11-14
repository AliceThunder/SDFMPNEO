"""Frozen-model, 24 unseen BFZI V150 electrothermal acceptance cases."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_param import create, originals
from examples.bfzi_obb_layer_study import GRIDS
from examples.bfzi_d_shared import make_geo, build_et
from examples.bfzi_obb_subcell_study import measure
from examples.bfzi_frac_rom_eval import load_net
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat, pack
from sdfmpneo_bfzi.geom_shared_rom import SharedROM
from sdfmpneo_bfzi.end_to_end import complete, metrics

def cases(runs):
    """Fixed deterministic design. No old P03/P07/G1/G2/G3 geometry."""
    _,a=make_geo(runs,"p03")
    _,b=make_geo(runs,"p07")
    out=[]
    # Four complementary geometry families, six excitation/material settings.
    families=[
      ("round",dict(alpha1=1.,core1_mm=0.,alpha2=1.,core2_mm=0.,
                    box1_mm=220.,box2_mm=224.)),
      ("blend",dict(alpha1=.95,core1_mm=7.,alpha2=.86,core2_mm=20.,
                    box1_mm=228.,box2_mm=236.)),
      ("square",dict(alpha1=.79,core1_mm=24.,alpha2=.76,core2_mm=27.,
                    box1_mm=238.,box2_mm=242.)),
      ("mixed",dict(alpha1=1.,core1_mm=0.,alpha2=.81,core2_mm=25.,
                    box1_mm=224.,box2_mm=240.))]
    shifts=[(2.,35.,2.),(13.,43.,-5.),(-12.,39.,11.),
            (6.,50.,-9.),(19.,37.,8.),(-7.,46.,-13.)]
    pose=[(4.,-3.,2.),(16.,8.,-7.),(29.,-11.,12.),
          (38.,14.,-9.),(12.,17.,16.),(26.,-18.,-13.)]
    volts=[20.,45.,75.,110.,65.,95.]
    alphas=[.02,.016,.025,.02,.012,.028]
    off=[[],[(180.,300.)],[(120.,240.),(420.,480.)],
         [(90.,150.),(300.,390.)],[(300.,540.)],
         [(60.,120.),(240.,300.),(480.,540.)]]
    for i,(tag,shape) in enumerate(families):
        for j in range(6):
            base=dict(a if i%2==0 else b)
            base.update(shape)
            base["shift_mm"]=list(shifts[(j+i)%6])
            base["tilt_deg"],base["yaw_deg"],base["pitch_deg"]=pose[(j+2*i)%6]
            if not (i==0 and j<2):
                base["shift_mm"][1]+=38.
                base["tilt_deg"]*=.65
            out.append(dict(id=f"F{i+1}{j+1}",family=tag,geometry=base,
                            donor="g1" if i in (0,3) else "g2",
                            voltage=volts[(j+i)%6],
                            alpha_per_K=alphas[(j+2*i)%6],
                            off=off[(j+3*i)%6],dt=30.,duration=600.))
    return out

def score(pred,truth,ns):
    z=truth["z6"]-truth["z6"][0]
    zz=pred["z6"]-pred["z6"][0]
    e=pred["temp_K"]-truth["temp_K"]
    return dict(
      delta_Z_rel_L2=float(np.linalg.norm(z-zz)/max(np.linalg.norm(z),1e-30)),
      copper_max_abs_K=float(np.max(abs(e[:,ns+64:ns+66]))),
      pkg_max_abs_K=float(np.max(abs(e[:,ns:ns+64]))),
      sea_mean_max_abs_K=float(np.max(abs(
        np.mean(e[:,:ns],axis=1)))),
      cold_Z_abs=float(np.linalg.norm(pred["z6"][0]-truth["z6"][0])))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--count",type=int,default=24)
    p.add_argument("--mode",choices=("frac","exp","none"),default="frac")
    a=p.parse_args()
    torch.set_num_threads(1)
    bf,v150=originals(a.bfzi_root)
    xz,y=GRIDS["extra"]
    all_cases=cases(a.runs)
    report_path=a.runs/"bfzi_final_accept.json"
    report=json.loads(report_path.read_text()) if report_path.exists() else {}
    report.setdefault("method","Original BFZI V150 source-skin independent unseen final acceptance")
    report.setdefault("cases",{})
    for item in all_cases[a.start:a.start+a.count]:
        key=item["id"]
        geo=item["geometry"]
        t=np.arange(0.,item["duration"]+1e-9,item["dt"])
        v=np.full_like(t,item["voltage"])
        for x,yoff in item["off"]:
            v[(t>=x)&(t<yoff)]=0.
        t0=time.perf_counter()
        data,_=create(bf,v150,"M4",**geo,sea_void="oriented",
                      physical_box_only=True,grid_xz=xz,grid_y=y)
        legacy=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                               fv_faces="exact",source_mask="legacy")
        legacy.alpha=item["alpha_per_K"]
        cold=legacy.solve(np.full(len(legacy.vol),293.15),
                          np.full(2,293.15))
        shared=SharedROM.load(a.runs,item["donor"])
        cold_z=pack(cold.total_z)
        offline=time.perf_counter()-t0
        net=ck=None
        if a.mode!="none":
            net,ck=load_net(a.runs/f"bfzi_frac_rom_{item['donor']}_{a.mode}.pt")
            ck=dict(ck,vref=item["voltage"],
                    scale=np.asarray(ck["scale"])*(item["voltage"]/ck["vref"])**2)
        t1=time.perf_counter()
        model=shared.target(cold_z,alpha_scale=legacy.alpha/.02)
        pred=shared.rollout(cold_z,t,v,net=net,ck=ck,
                            sea_xyz=legacy.points,
                            alpha_scale=legacy.alpha/.02)
        full=complete(legacy,cold,model,pred,v)
        infer=time.perf_counter()-t1
        # Physically independent BFZI source-skin full nonlinear time march.
        parent="p03" if item["donor"]=="g1" else "p07"
        prev=GeomHeat.load(a.runs/f"bfzi_pose_{parent}_disjoint_anchor.npz")
        t2=time.perf_counter()
        skin=.002
        try:
            et=build_et(data,bf,prev.ext)
        except ValueError as exc:
            if "skin water capacity" not in str(exc):
                raise
            from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D
            em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
                               fv_faces="exact",source_mask="subcell")
            for skin in (.0008,.0002,.00005):
                try:
                    et=BFZIV148Skin3D(em,external=prev.ext,
                        pkg_shape=(4,2,4),pkg_k=.25,skin_depth=skin,
                        skin_layers=8,skin_ratio=1.4,skin_quad=2,
                        skin_near=4,skin_map=2,pkg_sea_h=20.,
                        skin_heat="source")
                    break
                except ValueError as err:
                    if "skin water capacity" not in str(err):
                        raise
            else:
                raise ValueError("all physical BFZI skin depths invalid")
        et.model.alpha=item["alpha_per_K"]
        setup=time.perf_counter()-t2
        truth=measure(et,t,v)
        error=score(full,truth,len(legacy.vol))
        energy=metrics(full)
        physics_ok=(truth["max_node_budget_W"]<=1e-4 and
                    truth["max_total_budget_W"]<=1e-4)
        passed=(physics_ok and
                error["delta_Z_rel_L2"]<=.03 and
                error["copper_max_abs_K"]<=2. and
                error["pkg_max_abs_K"]<=2. and
                truth["direct_sec"]/max(infer,1e-9)>=20. and
                energy["max_rel_port_power"]<=1e-8 and
                energy["max_rel_q_power"]<=1e-8 and
                energy["max_rel_thermal_power"]<=1e-8)
        report["cases"][key]=dict(
            spec=item,skin_depth_m=skin,score=error,energy=energy,
            physical_gate_passed=bool(physics_ok),
            physical_node_budget_W=float(truth["max_node_budget_W"]),
            passed=bool(passed),
            timing=dict(offline_sec=offline,online_sec=infer,
                        physical_setup_sec=setup,
                        physical_direct_sec=truth["direct_sec"],
                        online_speedup=truth["direct_sec"]/max(infer,1e-9)),
            physical_budget_W=float(truth["max_total_budget_W"]))
        (a.runs/"bfzi_final_accept.json").write_text(
            json.dumps(report,indent=2))
        np.savez_compressed(a.runs/f"bfzi_final_{key}.npz",
            time_s=t,voltage_v=v,
            z6_pred=full["z6"],z6_truth=truth["z6"],
            temp_pred=full["temp_K"],temp_truth=truth["temp_K"],
            qsea_pred=full["qsea_W_m3"],Jsea_pred=full["Jsea_A_m2"],
            sea_xyz=full["sea_xyz"],sea_vol=full["sea_vol"],
            eta=full["eta"],Psea_pred=full["Psea_W"],
            Psea_truth=truth["Psea_W"],Pcu_pred=full["Pcu_W"],
            Pcu_truth=truth["Pcu_W"])
        print("FINAL",key,"PASS" if passed else "FAIL",
              json.dumps(dict(error=error,energy=energy,
                              offline=round(offline,2),
                              direct=round(truth["direct_sec"],2),
                              online=round(infer,3))),flush=True)
    print("FINISHED",len(report["cases"]),flush=True)

if __name__=="__main__":
    main()
