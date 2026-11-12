"""Compare geometry-new BFZI direct physics against SOE+full 3-D and 32D ROM.

Dynamic teacher is the original BFZI probe, excluded from all neural and ROM
training. The geometry-specific anchor comes solely from the cold solve.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch

from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_memory import GeomFrac


def ratio(d,ref):
    return float(np.linalg.norm(d)/max(np.linalg.norm(ref),1e-12))


def errors(est,base,geom):
    ns=geom.ns;pn=2*geom.npkg
    return dict(
        sea_max_K=float(np.max(abs(est[:,:ns]-base[:,:ns]))),
        pkg_max_K=float(np.max(abs(est[:,ns:ns+pn]-base[:,ns:ns+pn]))),
        cu_max_K=float(np.max(abs(est[:,ns+pn:]-base[:,ns+pn:]))))


def evaluate(obj,model,ck,t,voltage,ref_t,ref_z):
    start=time.perf_counter()
    p=obj.rollout(model,ck,np.full(len(t),voltage),t)
    sec=time.perf_counter()-start
    err=errors(p["temp_K"],ref_t,obj)
    err.update(
        seconds=sec,
        dynamic_Z_relative_L2=ratio(p["z6"]-ref_z,
                                  ref_z-ref_z[0]),
        sea_source_power_gap_W=float(np.max(abs(
            p["qsea_W_m3"]@obj.vol-p["Psea_W"]))),
        thermal_node_residual_W=float(max(p["fv_node_residual_W"])))
    if "material_budget_W" in p:
        err["material_budget_W"]=float(np.max(abs(p["material_budget_W"])))
        err["rom_rank"]=p["rom_rank"]
    return err,p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor",type=Path,
                        default=Path("runs/bfzi_param_anchor.npz"))
    parser.add_argument("--probe",type=Path,
                        default=Path("runs/bfzi_param_anchor_probe.npz"))
    parser.add_argument("--model",type=Path,
                        default=Path("runs/bfzi_geom_frac_v5.pt"))
    parser.add_argument("--rank",type=int,default=32)
    parser.add_argument("--rom-file",type=Path,
                        default=Path("runs/bfzi_param_rom.npz"))
    parser.add_argument("--voltage",type=float,default=75.)
    parser.add_argument("--out",type=Path,
                        default=Path("runs/bfzi_param_eval.json"))
    a=parser.parse_args()
    torch.set_num_threads(2)
    head=GeomHeat.load(a.anchor)
    with np.load(a.probe) as p:
        t=np.asarray(p["time_s"],float)
        tr=np.asarray(p["temp_K"],float)
        zr=np.asarray(p["z6"],float)
    ck=torch.load(a.model,map_location="cpu",weights_only=True)
    model=GeomFrac(ng=int(ck["ng"]))
    model.load_state_dict(ck["state"])
    model.eval()
    cold_err=ratio(head.z0-zr[0],zr[0])
    if cold_err>1e-10:
        raise ValueError("original BFZI cold Z and saved geometry anchor differ")
    full,fp=evaluate(head,model,ck,t,a.voltage,tr,zr)
    start=time.perf_counter()
    if a.rom_file.exists():
        rom=GeomHeatROM.load(head,a.rom_file)
    else:
        rom=GeomHeatROM.build(head,a.rank)
        a.rom_file.parent.mkdir(parents=True,exist_ok=True)
        rom.save(a.rom_file)
    setup=time.perf_counter()-start
    reduced,rp=evaluate(rom,model,ck,t,a.voltage,tr,zr)
    d=errors(rp["temp_K"],fp["temp_K"],head)
    result=dict(
        original_geometry_anchor=str(a.anchor),
        original_BFZI_direct_probe=str(a.probe),
        pretrained_fractional_model=str(a.model),
        original_cold_Z_relative_error=cold_err,
        original_direct_time_points=len(t),
        rom_setup_seconds=setup,
        full_FV=full,conservative_ROM=reduced,
        ROM_vs_full_FV=dict(
            **d,speedup=full["seconds"]/reduced["seconds"],
            Z_dynamic_additive_relative_L2=ratio(
                rp["z6"]-fp["z6"],fp["z6"]-fp["z6"][0]),
            q_relative_L2=ratio(
                rp["qsea_W_m3"]-fp["qsea_W_m3"],fp["qsea_W_m3"])))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
