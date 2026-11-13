"""BFZI V150 axis-aligned control: isolate k6/k8 tangent and heat-ROM error.

Uses stored original 3-D direct thermal trajectories; only cold BFZI
perturbation solves are newly performed. No new dynamic BFZI teacher.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_param import create,originals
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat,pack
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.frac_memory import zmat
from examples.bfzi_axis_grid_extend import GRIDS


def score(head,tan,truth,net,ck,rom):
    t=truth["time_s"]
    v=truth["voltage_v"]
    T=truth["temp_K"]
    z=truth["z6"]
    denom=max(float(np.linalg.norm(z-z[0])),1e-12)
    oracle=[]
    capture=[]
    for temp in T:
        sea=head.sea0+tan.delta(temp[:head.ns]-head.amb)
        cu=head.port.copper(temp[head.cu])
        oracle.append(pack(sea+1j*head.port.omega*head.port.air+
                           np.diag(cu)))
        capture.append(tan.capture(temp[:head.ns]-head.amb))
    row=dict(oracle_dynamic_Z_relative_L2=float(
        np.linalg.norm(np.asarray(oracle)-z)/denom),
        max_sea_mode_projection_rel=float(max(capture)))
    for label,h in (("full_FV",head),("ROM32",rom)):
        ts=time.perf_counter()
        p=h.rollout(net,ck,v,t,tangent=tan,blend=1.)
        e=p["temp_K"]-T
        row[label]=dict(
            dynamic_Z_relative_L2=float(
                np.linalg.norm(p["z6"]-z)/denom),
            sea_max_abs_K=float(np.max(abs(e[:,:head.ns]))),
            package_max_abs_K=float(
                np.max(abs(e[:,head.ns:head.cu.start]))),
            copper_max_abs_K=float(np.max(abs(e[:,head.cu]))),
            q_port_gap_W=float(np.max(abs(
                p["qsea_W_m3"]@head.vol-p["Psea_W"]))),
            inference_sec=time.perf_counter()-ts)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--levels",nargs="+",choices=("face_extra","face_ultra"),
                   default=("face_extra","face_ultra"))
    p.add_argument("--out",type=Path,default=Path(
        "runs/bfzi_axis_rank_gate.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",
                  map_location="cpu",weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"])
    net.eval()
    out=dict(
        test="BFZI original warm 3D FV vs physical tangent and 32D ROM",
        original_temperature_and_Z_used_only_for_evaluation=True,
        levels={})
    for label in a.levels:
        xz,y=GRIDS[label]
        head=GeomHeat.load(
            a.runs/f"bfzi_axis_grid_{label}_anchor.npz")
        rom=GeomHeatROM.load(
            head,a.runs/f"bfzi_axis_grid_{label}_rom.npz")
        with np.load(a.runs/f"bfzi_axis_grid_{label}.npz") as d:
            truth={k:d[k].copy() for k in ("time_s","voltage_v","temp_K","z6")}
        k6=SeaTangent.load(
            a.runs/f"bfzi_axis_grid_{label}_k6.npz",head)
        direct,_=create(bf,v150,"M4",physical_box_only=True,
                        grid_xz=xz,grid_y=y,grid_layers=0)
        em=BFZIV148Adapter(direct,bf,heat_mode="port-weak")
        if (not np.allclose(em.points,head.xyz,atol=1e-12,rtol=0)
                or not np.allclose(em.vol,head.vol,atol=1e-15,rtol=1e-10)):
            raise ValueError("original BFZI reference thermal grid differs")
        k8=SeaTangent.build(
            em,head,rom.phi,rank=8,amplitude=.2,method="forward")
        k8.save(a.runs/f"bfzi_axis_grid_{label}_k8.npz")
        rows={}
        for name,tan in (("cold_response_k6",k6),("forward_k8",k8)):
            rows[name]=score(head,tan,truth,net,ck,rom)
            print(label,name,"oracle%=",round(
                rows[name]["oracle_dynamic_Z_relative_L2"]*100,6),
                  "fullFV%=",round(
                rows[name]["full_FV"]["dynamic_Z_relative_L2"]*100,6),
                  "ROM32%=",round(
                rows[name]["ROM32"]["dynamic_Z_relative_L2"]*100,6),
                  flush=True)
        out["levels"][label]=rows
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(out,indent=2))
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
