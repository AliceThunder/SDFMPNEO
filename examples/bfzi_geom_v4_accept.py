"""Freeze V4 checkpoint on five dev geometries; evaluate five unseen tests once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_neural import eval_rollout
from sdfmpneo_bfzi.geom_neural import GeomFracET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path,
                   default=Path("runs/geomfrac_s5_v4_30g_manifest.json"))
    p.add_argument("--out", type=Path,
                   default=Path("runs/geomfrac_et_v4_final.json"))
    a = p.parse_args()
    torch.set_num_threads(2)
    mf = json.loads(a.manifest.read_text(encoding="utf-8"))
    train = mf["train"]
    val = mf["val"]
    test = mf["test"]
    if (len(train), len(val), len(test)) != (30, 5, 5):
        raise ValueError("V4 final evaluation requires 30/5/5 geometries")
    dev = {f for x in val for f in x["teacher"]}
    unseen = {f for x in test for f in x["teacher"]}
    seen = {f for x in train for f in x["teacher"]}
    if seen & dev or dev & unseen or seen & unseen:
        raise ValueError("geometry or teacher leakage")

    cfg = (
        ("5g", "runs/geomfrac_et_v4_5g_z0.pt",
         "runs/geomfrac_et_v4_5g_val5.json", "results"),
        ("8g_quality", "runs/geomfrac_et_v4_8g_quality.pt",
         "runs/geomfrac_et_v4_8g_quality_val5.json", "results"),
        ("30g", "runs/geomfrac_et_v4_30g.pt",
         "runs/geomfrac_et_v4_30g.json", "autonomous_validation"),
        ("30g_quality", "runs/geomfrac_et_v4_30g_quality.pt",
         "runs/geomfrac_et_v4_30g_quality.json", "autonomous_validation"),
    )
    rows = []
    for name, model, report, key in cfg:
        d = json.loads(Path(report).read_text(encoding="utf-8"))
        vals = d[key]
        if len(vals) != len(dev) or {v["file"] for v in vals} != dev:
            raise RuntimeError(f"{name}: incomplete or mismatched dev evaluation")
        z = float(np.mean([x["dynamic_Z_rel_L2"] for x in vals]))
        rows.append(dict(name=name,model=model,report=report,dev_Z_mean_rel_L2=z))
    selected = min(rows,key=lambda x:x["dev_Z_mean_rel_L2"])
    ck = torch.load(selected["model"],map_location="cpu",weights_only=True)
    if set(ck["train_files"]) & unseen or set(ck["val_files"]) & unseen:
        raise RuntimeError("selected model has seen final tests")
    net = GeomFracET(
        ng=int(ck["ng"]),rank=int(ck["rank"]),
        soe_quad=ck.get("soe_quad","trap"),
        heat_dt=float(ck.get("heat_dt",1.875)),heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    hist = [r for spec in test for r in prepare(spec,int(ck["rank"]))]
    result = eval_rollout(None,net,ck,hist)
    out = dict(
        selection="minimum mean autonomous dynamic Z relative L2 on T04/D01-D04",
        training_geometries=len(train),dev_geometries=len(val),
        test_geometries=len(test),models=rows,
        selected=selected["name"],selected_model=selected["model"],
        test_dynamic_Z_mean_rel_L2=float(np.mean([
            z["dynamic_Z_rel_L2"] for z in result])),
        test_dynamic_Z_max_rel_L2=float(np.max([
            z["dynamic_Z_rel_L2"] for z in result])),
        test_histories=result)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("FINAL_MODEL",selected["name"],
          "dev",round(100*selected["dev_Z_mean_rel_L2"],3),
          "test",round(100*out["test_dynamic_Z_mean_rel_L2"],3),
          "test_max",round(100*out["test_dynamic_Z_max_rel_L2"],3),flush=True)
    for r in result:
        print("UNSEEN",r["file"],
              round(100*r["dynamic_Z_rel_L2"],3),
              "tempK",round(r["temperature_max_K"],3),flush=True)


if __name__ == "__main__":
    main()
