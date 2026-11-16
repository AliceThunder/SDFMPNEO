"""Frozen V6 independent BFZI S5 electrothermal test on R01-R03."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_neural import eval_rollout
from examples.bfzi_z_tangent import restore as load_tan
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_pkg import restore as load_pkg


def summarize(vals):
    return dict(
        dynamic_Z_mean_rel_L2=float(np.mean([v["dynamic_Z_rel_L2"] for v in vals])),
        dynamic_Z_max_rel_L2=float(max(v["dynamic_Z_rel_L2"] for v in vals)),
        temperature_mean_max_K=float(np.mean([v["temperature_max_K"] for v in vals])),
        temperature_max_K=float(max(v["temperature_max_K"] for v in vals)),
        material_budget_max_W=float(max(v["material_budget_max_W"] for v in vals)),
        mean_cpu_s=float(np.mean([v["inference_cpu_s"] for v in vals])),
        results=vals)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v6_manifest.json"))
    p.add_argument("--selection",type=Path,default=Path("runs/geomfrac_et_v6_selection.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_et_v6_final.json"))
    a=p.parse_args()
    torch.set_num_threads(2)
    cfg=json.loads(a.manifest.read_text(encoding="utf-8"))
    freeze=json.loads(a.selection.read_text(encoding="utf-8"))
    specs=cfg.get("test",[])
    want=freeze["fresh_test_geometries"]
    cases=[Path(s["anchor"]).stem.split("_")[2] for s in specs]
    if sorted(cases)!=sorted(want):
        p.error("all frozen R01-R03 geometries must be present")
    ck=torch.load(freeze["original_heat_model"],map_location="cpu",weights_only=True)
    net=GeomFracET(ng=int(ck["ng"]),rank=int(ck["rank"]),
        soe_quad=ck.get("soe_quad","trap"),heat_dt=float(ck.get("heat_dt",1.875)),
        heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    net.eval()
    tan=load_tan(freeze["sea_impedance_model"])
    pkg=load_pkg(freeze["package_model"])
    with np.load(freeze["sea_lift_model"]) as src:
        lift=src["w"]*float(freeze["lift_gain"])
    rows=[r for s in specs for r in prepare(s,int(ck["rank"]))]
    if len(rows)!=3*len(specs):
        p.error("R01-R03 require three teacher histories per geometry")
    opts={"V5":dict(tangent=tan),
          "V6_package":dict(tangent=tan,pkg=pkg),
          "V6_coupled":dict(tangent=tan,pkg=pkg,lift=lift)}
    outcomes={}
    for name,kw in opts.items():
        vals=eval_rollout(None,net,ck,rows,**kw)
        outcomes[name]=summarize(vals)
        print("ACCEPT",name,"Z_pct",
              round(100*outcomes[name]["dynamic_Z_mean_rel_L2"],3),
              "T_max_K_mean",round(outcomes[name]["temperature_mean_max_K"],3),
              flush=True)
    out=dict(description="frozen independent V6 R01-R03 acceptance",
             test_geometries=want,test_histories=len(rows),
             selection_file=str(a.selection),manifest_file=str(a.manifest),
             selection_unchanged=True,models=outcomes)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2),encoding="utf-8")
    print("FINAL",a.out,flush=True)


if __name__=="__main__":
    main()
