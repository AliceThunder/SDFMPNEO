"""Separate full FV heat error from 32-mode ROM on exact-OBB BFZI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.geom_memory import GeomFrac


def score(pred,physical,ns):
    z=physical["z6"]
    t=physical["temp_K"]
    d=pred["temp_K"]-t
    return dict(
        dynamic_Z_relative_L2=float(
            np.linalg.norm(pred["z6"]-z)/
            max(np.linalg.norm(z-z[0]),1e-12)),
        sea_max_abs_K=float(np.max(abs(d[:,:ns]))),
        pkg_max_abs_K=float(np.max(abs(d[:,ns:-2]))),
        cu_max_abs_K=float(np.max(abs(d[:,-2:]))),
        q_port_gap_W=float(np.max(abs(
            pred["qsea_W_m3"]@physical["sea_vol"]-pred["Psea_W"]))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_obb_fv_rom_comparison.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"]);net.eval()
    out={}
    for name in ("P03","P07"):
        root=a.runs/f"bfzi_pose_{name.lower()}_disjoint"
        head=GeomHeat.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_anchor.npz")
        rom=GeomHeatROM.load(head,a.runs/
                            f"bfzi_pose_{name.lower()}_disjoint_rom.npz")
        tan=SeaTangent.load(a.runs/
                           f"bfzi_pose_{name.lower()}_disjoint_k6.npz",head)
        with np.load(a.runs/f"bfzi_pose_{name.lower()}_disjoint_oriented_medium.npz") as z:
            t=z["time_s"].copy()
            v=z["voltage_v"].copy()
            physical=dict(temp_K=z["temp_K"].copy(),z6=z["z6"].copy(),
                          sea_vol=z["sea_vol"].copy())
        full=head.rollout(net,ck,v,t,tangent=tan,blend=1.)
        reduced=rom.rollout(net,ck,v,t,tangent=tan,blend=1.)
        q=score(full,physical,head.ns)
        r=score(reduced,physical,head.ns)
        diff=np.max(abs(full["temp_K"]-reduced["temp_K"]),axis=0)
        out[name]=dict(full_FV=q,rom32=r,rom_fv_max_K=dict(
            sea=float(np.max(diff[:head.ns])),
            package=float(np.max(diff[head.ns:-2])),
            copper=float(np.max(diff[-2:]))))
        print(name,"FV_Z_pct",round(q["dynamic_Z_relative_L2"]*100,6),
              "ROM_Z_pct",round(r["dynamic_Z_relative_L2"]*100,6),
              "ROM_FV_pkg_K",round(out[name][
                  "rom_fv_max_K"]["package"],6),flush=True)
    a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
