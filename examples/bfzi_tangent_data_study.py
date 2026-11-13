"""Zero-new-EM cross-geometry BFZI tangent training from archived 3-D trajectories.

Runs without the independent BFZI V150 checkout. P03/P07 remain untouched
until final held-out oracle/full-3D evaluation; five train geometries only.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_tangent_eval import score
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_memory import GeomFrac
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_tangent import SeaTangent
from sdfmpneo_bfzi.geom_tangent_data import fit
from sdfmpneo_bfzi.geom_tangent_transfer import TangentTransfer

TRAIN=("P01","P02","P04","P05","P06")
HOLD=("P03","P07")


def eval_3d(head,rom,tan,net,ck,teacher):
    with np.load(teacher) as d:
        t=np.asarray(d["time_s"],float)
        v=np.asarray(d["voltage_v"],float)
        temp=np.asarray(d["temp_K"],float)
        z=np.asarray(d["z6"],float)
        ids=json.loads(str(d["cases_json"]))
    results=[]
    for i in range(len(v)):
        start=time.perf_counter()
        pred=rom.rollout(net,ck,v[i],t,tangent=tan,blend=1.)
        elapsed=time.perf_counter()-start
        d=pred["temp_K"]-temp[i]
        results.append(dict(
            case=int(ids[i]["id"]),voltage_V=float(ids[i]["voltage_v"]),
            dynamic_Z_relative_L2=float(np.linalg.norm(pred["z6"]-z[i])/
                max(np.linalg.norm(z[i]-z[i,0]),1e-12)),
            sea_max_abs_K=float(np.max(abs(d[:,:head.ns]))),
            pkg_max_abs_K=float(np.max(abs(d[:,head.ns:head.cu.start]))),
            cu_max_abs_K=float(np.max(abs(d[:,head.cu]))),
            q_port_gap_W=float(np.max(abs(pred["qsea_W_m3"]@head.vol-
                                          pred["Psea_W"]))),
            material_budget_gap_W=float(np.max(abs(pred["material_budget_W"]))),
            inference_sec=elapsed))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--epochs",type=int,default=350)
    p.add_argument("--out",type=Path,
                   default=Path("runs/bfzi_tangent_data_study.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    results=dict(
        source="already archived original BFZI V150 3-D physical trajectories",
        new_directional_BFZI_EM_solves=0,
        train_geometries=list(TRAIN),heldout_geometries=list(HOLD),
        train={},heldout={})
    examples=[]
    start=time.perf_counter()
    for name in TRAIN:
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        head=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        t0=time.perf_counter()
        rom=GeomHeatROM.build(head,rank=32)
        rom_sec=time.perf_counter()-t0
        tan,diag=fit(head,rom.phi,
                     pre.with_name(pre.name+"_teacher.npz"),rank=8)
        tan.save(a.runs/f"bfzi_pose_{name.lower()}_data_tangent.npz")
        diag["cold_ROM_build_sec"]=rom_sec
        results["train"][name]=diag
        examples.append((name,head,tan))
        print("TRAIN",name,json.dumps(diag),flush=True)
    t0=time.perf_counter()
    model=TangentTransfer.train(examples,epochs=a.epochs)
    results["neural_training_sec"]=time.perf_counter()-t0
    results["training_LOO_errors"]=model.cv.tolist()
    results["neural_residual_weight"]=model.weight
    model_path=a.runs/"bfzi_tangent_transfer_data.npz"
    model.save(model_path)
    ck=torch.load(a.runs/"bfzi_geom_diverse.pt",map_location="cpu",
                  weights_only=True)
    net=GeomFrac(ng=int(ck["ng"]))
    net.load_state_dict(ck["state"])
    net.eval()
    for name in HOLD:
        pre=a.runs/f"bfzi_pose_{name.lower()}"
        head=GeomHeat.load(pre.with_name(pre.name+"_anchor.npz"))
        rom=GeomHeatROM.load(head,pre.with_name(pre.name+"_rom.npz"))
        t0=time.perf_counter()
        predicted=model.predict(head,rom)
        infer_sec=time.perf_counter()-t0
        pred_file=a.runs/f"bfzi_pose_{name.lower()}_data_neural_tangent.npz"
        predicted.save(pred_file)
        central=SeaTangent.load(pre.with_name(
            pre.name+"_tangent_anchor_sea_tangent.npz"),head)
        tests=[pre.with_name(pre.name+"_holdout.npz")]
        if name=="P07":
            tests.append(a.runs/"bfzi_pose_p07_15s_holdout.npz")
        case={}
        for teacher in tests:
            case[teacher.name]={}
            for label,tan in (("original_central16",central),
                              ("data_neural0",predicted)):
                case[teacher.name][label]=dict(
                    oracle_true_3D_T=score(head,tan,teacher),
                    autonomous_3D=eval_3d(head,rom,tan,net,ck,teacher))
        results["heldout"][name]=dict(
            neural_tangent_file=str(pred_file),
            new_geometry_tangent_inference_sec=infer_sec,
            trajectories=case)
        print("HELDOUT",name,json.dumps(case),flush=True)
    results["total_wall_sec"]=time.perf_counter()-start
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print("SAVED",a.out,flush=True)


if __name__=="__main__":
    main()
