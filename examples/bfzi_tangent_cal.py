"""Test geometry-specific warm original BFZI V150 probe on frozen truth."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_final_accept import cases
from examples.bfzi_param import create, originals
from examples.bfzi_obb_layer_study import GRIDS
from examples.bfzi_frac_rom_eval import load_net
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_shared_rom import SharedROM
from sdfmpneo_bfzi.geom_heat import pack

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--keys",nargs="*",default=["F11","F12","F21","F26","F31","F35","F44","F45"])
    a=p.parse_args()
    torch.set_num_threads(1)
    bf,v150=originals(a.bfzi_root)
    xz,y=GRIDS["extra"]
    specs={x["id"]:x for x in cases(a.runs)}
    saved=a.runs/'bfzi_tangent_cal.json'
    res=json.loads(saved.read_text()) if saved.exists() else {}
    for key in a.keys:
        spec=specs[key]
        x=np.load(a.runs/f"bfzi_final_{key}.npz")
        t=x["time_s"];v=x["voltage_v"];truth=x["z6_truth"]
        geom=spec["geometry"]
        data,_=create(bf,v150,"M4",**geom,sea_void="oriented",
            physical_box_only=True,grid_xz=xz,grid_y=y)
        em=BFZIV148Adapter(data,bf,heat_mode="port-weak",
            fv_faces="exact",source_mask="subcell")
        em.alpha=spec["alpha_per_K"]
        sea=np.full(len(em.vol),293.15)
        cu=np.full(2,293.15)
        z0=pack(em.solve(sea,cu).total_z)
        shared=SharedROM.load(a.runs,spec["donor"])
        model=shared.target(z0,alpha_scale=em.alpha/.02)
        net,ck=load_net(a.runs/f"bfzi_frac_rom_{spec['donor']}_frac.pt")
        ck=dict(ck,vref=spec["voltage"],
            scale=np.asarray(ck["scale"])*(spec["voltage"]/ck["vref"])**2)
        baseline=model.rollout(shared.rom,t,v,net,ck)
        u=shared.rom.encode(baseline["temp_K"][-1])
        rise=x["temp_pred"][-1]-293.15
        amp=max(np.max(abs(rise)),1e-6)
        profile=rise/amp
        zp=pack(em.solve(sea+profile[:len(sea)],
            cu+profile[len(sea)+64:len(sea)+66]).total_z)
        donor=model.jac@u
        exact=(zp-z0)*amp
        model.jac+=np.outer(exact-donor,u)/float(u@u)
        pred=model.rollout(shared.rom,t,v,net,ck)
        dz=truth-truth[0]
        zp=pred["z6"]-pred["z6"][0]
        err=float(np.linalg.norm(zp-dz)/np.linalg.norm(dz))
        orig=float(np.linalg.norm(
            (x["z6_pred"]-x["z6_pred"][0])-dz)/np.linalg.norm(dz))
        res[key]=dict(original=orig,calibrated=err,
            donor_slope=donor.tolist(),target_slope=exact.tolist(),
            cold_gap=float(np.linalg.norm(z0-truth[0])))
        print(key,"orig",round(orig*100,3),"cal",round(err*100,3),
            "slope_rel",round(np.linalg.norm(donor-exact)/
                max(np.linalg.norm(exact),1e-30)*100,3),flush=True)
    (a.runs/"bfzi_tangent_cal.json").write_text(json.dumps(res,indent=2))

if __name__=="__main__":
    main()
