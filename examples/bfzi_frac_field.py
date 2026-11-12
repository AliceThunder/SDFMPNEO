"""Train a compact sea J/q field head and verify it on held-out BFZI M4.

Inputs are the genuine 30/150 V BFZI V150 three-material 3-D histories.
The test 100 V run has never been used to fit this spatial field decoder.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_frac import load
from sdfmpneo_bfzi.frac_field import SeaHead,complex_z
from sdfmpneo_bfzi.frac_memory import predict


def rel(a,b):
    a=np.asarray(a)
    b=np.asarray(b)
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-12))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train",nargs="+",type=Path,default=[
        Path("runs/m4_pkg3d_weak_30v.npz"),
        Path("runs/m4_pkg3d_weak_150v.npz")])
    p.add_argument("--test",type=Path,
                   default=Path("runs/m4_pkg3d_weak_100v.npz"))
    p.add_argument("--model",type=Path,
                   default=Path("runs/m4_frac_multi_hybrid.pt"))
    p.add_argument("--field",type=Path,
                   default=Path("runs/m4_frac_field.npz"))
    p.add_argument("--out",type=Path,
                   default=Path("runs/m4_frac_field_metrics.json"))
    args=p.parse_args()
    t0=time.perf_counter()
    head=SeaHead.fit(args.train,rank=6)
    args.field.parent.mkdir(parents=True,exist_ok=True)
    head.save(args.field)
    training_sec=time.perf_counter()-t0
    d=json.loads(args.test.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(args.test) as fields:
        times=np.asarray(fields["time_s"],float)
        temp=np.asarray(fields["Tsea_K"],float)
        j_ref=np.asarray(fields["Jsea_A_m2"],complex)
        q_ref=np.asarray(fields["qsea_W_m3"],float)
    vin=np.array([float(d["source_voltage_peak_V"])
                  if row["drive"]=="on" else 0.
                  for row in d["trace"]])
    I_true=np.array([
        np.asarray(row["I_A"],float)[:,0]
        +1j*np.asarray(row["I_A"],float)[:,1]
        for row in d["trace"]])
    z_ref=np.array([
        complex_z(row["Zsea_ohm"]) for row in d["trace"]])
    net,ck=load(args.model)
    torch.set_num_threads(2)
    t1=time.perf_counter()
    pr=predict(net,ck,vin,times)
    J_true_input=[];q_true_input=[];z_true_input=[]
    J_neural=[];q_neural=[];z_neural=[]
    for k in range(len(times)):
        j,q,z=head.at(temp[k],I_true[k])
        J_true_input.append(j)
        q_true_input.append(q)
        z_true_input.append(z)
        j,q,z=head.at(pr["temp_K"][k,:len(head.vol)],
                      pr["current_A"][k])
        J_neural.append(j)
        q_neural.append(q)
        z_neural.append(z)
    neural_sec=time.perf_counter()-t1
    J_true_input=np.asarray(J_true_input)
    q_true_input=np.asarray(q_true_input)
    z_true_input=np.asarray(z_true_input)
    J_neural=np.asarray(J_neural)
    q_neural=np.asarray(q_neural)
    z_neural=np.asarray(z_neural)
    power_neural=q_neural@head.vol
    port_neural=np.array([.5*np.vdot(i,z@i).real
                         for i,z in zip(pr["current_A"],z_neural)])
    power_ref=q_ref@head.vol
    result=dict(
        teacher_train_files=[str(p) for p in args.train],
        test_file=str(args.test),
        neural_model=str(args.model),
        field_file=str(args.field),
        train_seconds=training_sec,
        rollout_seconds=neural_sec,
        sea_cells=len(head.vol),
        temperatures_used="direct+neural comparisons",
        direct_temperature_inputs={
            "sea_J_relative_L2":rel(J_true_input,j_ref),
            "weak_q_relative_L2":rel(q_true_input,q_ref),
            "Zsea_relative_L2":rel(z_true_input,z_ref)},
        neural_temperature_inputs={
            "sea_J_relative_L2":rel(J_neural,j_ref),
            "weak_q_relative_L2":rel(q_neural,q_ref),
            "Zsea_relative_L2":rel(z_neural,z_ref),
            "sea_port_power_relative_L2":rel(power_neural,power_ref),
            "q_min_W_m3":float(q_neural.min()),
            "q_integral_vs_port_max_abs_W":float(
                np.max(abs(power_neural-port_neural)))},
    )
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(
        args.out.with_suffix(".npz"),
        time_s=times,sea_xyz_m=np.asarray(
            np.load(args.test)["sea_xyz_m"],float),
        J_true_A_m2=j_ref,J_pred_A_m2=J_neural,
        q_true_W_m3=q_ref,q_pred_W_m3=q_neural,
        zsea_true=z_ref,zsea_pred=z_neural,
        port_power_pred_W=port_neural)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
