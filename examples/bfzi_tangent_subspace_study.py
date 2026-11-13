"""Original BFZI physical heat-response subspace: 2/4/6 vs 8 EM probes.

The probe directions depend only on the cold BFZI heat matrices/sources.
P03/P07 direct EM and time trajectories are used only for final evaluation.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_tangent_study import cold_em,full_score
from examples.bfzi_param import originals
from examples.bfzi_tangent_eval import score
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.geom_tangent_subspace import build,directions

HOLD=("P03","P07")


def projected(full, head, dirs):
    """Full 16-solve Jacobian restricted to cold-selected k directions."""
    Q=full.modes.T@(head.vol[:,None]*dirs)
    jac=np.einsum("ij,iab->jab",Q,full.jac,optimize=True)
    t=SeaTangent(dirs,jac,head.vol,head.sea0,head.geom)
    t.method="projected-central16"
    t.nsolve=0
    return t


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root",type=Path,required=True)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--rank",type=int,nargs="+",default=[2,4,6])
    p.add_argument("--amplitude",type=float,default=.2)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_tangent_subspace_study.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    bf,v150=originals(a.bfzi_root)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",map_location="cpu",
                  weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"])
    net.eval()
    out=dict(method="original BFZI cold heat-response rank-k thermal probes",
             fixed_ranks=list(a.rank),amplitude_K=a.amplitude,
             train_dynamic_truth_used=False,heldout={})
    for name in HOLD:
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        head=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        rom=GeomHeatROM.load(head,pre.with_name(pre.name+"_rom.npz"))
        meta=json.loads(pre.with_name(pre.name+"_anchor.json").read_text())
        t0=time.perf_counter()
        em=cold_em(bf,v150,head,meta)
        init_sec=time.perf_counter()-t0
        full=SeaTangent.load(a.runs/f"bfzi_pose_{name.lower()}_fd_central.npz",
                             head)
        old=SeaTangent.load(a.runs/f"bfzi_pose_{name.lower()}_fd_forward.npz",
                            head)
        files=[a.runs/f"bfzi_pose_{name.lower()}_holdout.npz"]
        if name=="P07":
            files.append(a.runs/"bfzi_pose_p07_15s_holdout.npz")
        result=dict(cold_setup_sec=init_sec,forward8_EM_solves=old.nsolve,
                    ranks={},forward8={})
        for teacher in files:
            result["forward8"][teacher.name]=dict(
                true_T=score(head,old,teacher),
                autonomous=full_score(head,rom,old,net,ck,teacher))
        for rank in a.rank:
            t0=time.perf_counter()
            reduced,stat=build(em,head,rom,full.modes,rank,
                               amplitude=a.amplitude)
            stat["probe_design_sec"]=time.perf_counter()-t0-stat[
                "directional_sec"]
            path=a.runs/f"bfzi_pose_{name.lower()}_cold_rank{rank}.npz"
            reduced.save(path)
            dirs,_=directions(head,rom,full.modes,rank)
            proj=projected(full,head,dirs)
            metrics={}
            for teacher in files:
                metrics[teacher.name]=dict(
                    true_T=score(head,reduced,teacher),
                    projected_central16_true_T=score(head,proj,teacher),
                    autonomous=full_score(head,rom,reduced,net,ck,teacher))
            result["ranks"][str(rank)]=dict(
                statistics=stat,tangent_file=str(path),heldout=metrics)
            print("GEOM",name,"RANK",rank,
                  "cold_energy",round(stat["explained_fraction"],8),
                  "EM_sec",round(stat["directional_sec"],4),
                  "dynamic_error",[round(v["autonomous"][0][
                      "dynamic_Z_relative_L2"]*100,6)
                      for v in metrics.values()],flush=True)
        out["heldout"][name]=result
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
