"""One-shot geometry-disjoint BFZI V150 acceptance for frozen V9 heat candidate."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v9_full_blind_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v9full_manifest.json"))
    p.add_argument("--script",type=Path,default=Path("D:/work/geomfrac-v8/examples/bfzi_v8_accept.py"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v9_full_blind_summary.json"))
    a=p.parse_args()
    plan=read(a.plan)
    mf=read(a.manifest)
    specs=mf.get("test",[])
    names=[Path(s["anchor"].replace("\\","/")).stem.split("_")[2] for s in specs]
    if (len(specs)!=4 or set(names)!=set(plan["test"])
            or any(len(s["teacher"])!=2 for s in specs)
            or not all(Path(f.replace("\\","/")).is_file()
                for spec in specs for f in spec["teacher"])):
        raise ValueError("V9 independent full-physics holdout is not complete")

    common=["--plan",str(a.plan),"--manifest",str(a.manifest),
            "--split","test",
            "--config","runs/geomfrac_free_rank48_v8_selection.json",
            "--skin-tangent","runs/geomfrac_rank48_skin_tan_v3.pt",
            "--skin-base","runs/geomfrac_et_v9_ztune.pt",
            "--parallel-model","runs/geomfrac_v8_par48.pt",
            "--k-mix","0.25"]
    results={}
    for name,ck in (("V8","runs/geomfrac_et_v8_heat_tune.pt"),
                    ("V9_heat","runs/geomfrac_v9_full_heat.pt")):
        dest=f"runs/geomfrac_v9_full_blind_{'base' if name=='V8' else 'heat'}.json"
        cmd=[sys.executable,"-u",str(a.script),*common,
             "--model",ck,"--out",dest]
        subprocess.run(cmd,check=True)
        results[name]=read(dest)
        print("V9_BLIND_MODEL",name,json.dumps(results[name]["overall"]),flush=True)
    keys=("dynamic_z_rel_L2","R11_dynamic_rel_L2_self",
          "R22_dynamic_rel_L2_self","X11_dynamic_rel_L2_self",
          "X22_dynamic_rel_L2_self","material_mean_rmse_K",
          "current_rel_L2","sea_power_rel_L2","copper_power_rel_L2")
    before=results["V8"]["overall"];after=results["V9_heat"]["overall"]
    changes={k:{"V8":before[k],"V9_heat":after[k],
                "delta":after[k]-before[k]} for k in keys}
    passed=(after["dynamic_z_rel_L2"]<before["dynamic_z_rel_L2"]
            and after["R11_dynamic_rel_L2_self"]<=1.02*before["R11_dynamic_rel_L2_self"]
            and after["R22_dynamic_rel_L2_self"]<=1.02*before["R22_dynamic_rel_L2_self"]
            and after["material_mean_rmse_K"]<=1.05*before["material_mean_rmse_K"]
            and after["current_rel_L2"]<=1.05*before["current_rel_L2"])
    per={n:{k:{"V8":results["V8"]["per_geometry"][n][k],
                 "V9_heat":results["V9_heat"]["per_geometry"][n][k]}
             for k in keys} for n in names}
    result=dict(status="blind_improved" if passed else "blind_not_improved",
        geometry_count=4,history_count=8,
        source_test=plan["source_test"],frozen_before_physics=True,
        no_blind_tuning=True,
        candidate="runs/geomfrac_v9_full_heat.pt",
        baseline="runs/geomfrac_et_v8_heat_tune.pt",
        metrics=changes,per_geometry=per)
    a.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V9_BLIND_FINAL",json.dumps({k:result[k] for k in
        ("status","geometry_count","history_count","metrics")}),flush=True)


if __name__=="__main__":
    main()
