"""Paired, frozen BFZI V10 blind acceptance (V8 vs V10 ColdNet).

This script consumes newly generated complete BFZI physical histories and
never changes weights, optimizers or validation thresholds.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

KEYS=("cold_z_rel_L2","dynamic_z_rel_L2",
      "R11_dynamic_rel_L2_self","X11_dynamic_rel_L2_self",
      "R22_dynamic_rel_L2_self","X22_dynamic_rel_L2_self",
      "current_rel_L2","material_mean_rmse_K","sea_power_rel_L2",
      "copper_power_rel_L2")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_v10_blind_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v10err_manifest.json"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_s5_v10err_build_report.json"))
    p.add_argument("--script",type=Path,default=Path("D:/work/geomfrac-v8/examples/bfzi_v8_accept.py"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v10_blind_summary.json"))
    a=p.parse_args()
    plan=read(a.plan);mf=read(a.manifest);report=read(a.report)
    specs=mf.get("test",[])
    names=[Path(s["anchor"].replace("\\","/")).stem.split("_")[2] for s in specs]
    if (len(specs)!=4 or sorted(names)!=sorted(plan["test"]) or
            len(set(names))!=4 or
            not all(len(s["teacher"])==2 for s in specs) or
            not all(Path(str(f).replace("\\","/")).is_file()
                    for s in specs for f in s["teacher"]) or
            not all(report["cases"][n]["status"]=="complete" for n in names)):
        raise ValueError("incomplete frozen V10 full-physics reference; abort acceptance")
    common=["--plan",str(a.plan),"--manifest",str(a.manifest),
            "--split","test","--config","runs/geomfrac_free_rank48_v8_selection.json",
            "--skin-tangent","runs/geomfrac_rank48_skin_tan_v3.pt",
            "--skin-base","runs/geomfrac_et_v9_ztune.pt",
            "--parallel-model","runs/geomfrac_v8_par48.pt","--k-mix","0.25"]
    models={
        "V8":("runs/geomfrac_et_v8_heat_tune.pt","runs/geomfrac_cold_v8_frozen.pt"),
        "V10_cold":("runs/geomfrac_et_v8_heat_tune.pt","runs/geomfrac_v10_cold.pt")}
    results={}
    for name,(heat,cold) in models.items():
        dest="runs/geomfrac_v10_blind_"+("base" if name=="V8" else "cold")+".json"
        cmd=[sys.executable,"-u",str(a.script),*common,"--model",heat,
             "--cold-model",cold,"--out",dest]
        subprocess.run(cmd,check=True)
        results[name]=read(dest)
        print("V10_BLIND_MODEL",name,json.dumps(results[name]["overall"]),flush=True)
    base=results["V8"]["overall"]
    cur=results["V10_cold"]["overall"]
    delta={k:dict(V8=base[k],V10_cold=cur[k],delta=cur[k]-base[k])
           for k in KEYS}
    passed=(cur["dynamic_z_rel_L2"]<base["dynamic_z_rel_L2"]
            and cur["R11_dynamic_rel_L2_self"]<=1.02*base["R11_dynamic_rel_L2_self"]
            and cur["R22_dynamic_rel_L2_self"]<=1.02*base["R22_dynamic_rel_L2_self"]
            and cur["current_rel_L2"]<=1.03*base["current_rel_L2"]
            and cur["material_mean_rmse_K"]<=1.03*base["material_mean_rmse_K"])
    per={name:{k:dict(V8=results["V8"]["per_geometry"][name][k],
                       V10_cold=results["V10_cold"]["per_geometry"][name][k])
               for k in KEYS} for name in names}
    out=dict(status="blind_improved" if passed else "blind_not_improved",
             source_test=plan["source_test"],geometry_count=4,history_count=8,
             train_candidate="runs/geomfrac_v10_cold.pt",
             physics="BFZI V150 S5 ultra matched 60s/300s",
             frozen_before_generation=True,changed_module="ColdNet only",
             metrics=delta,per_geometry=per)
    a.out.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print("V10_BLIND_FINAL",json.dumps({k:out[k] for k in
          ("status","geometry_count","history_count","metrics")}),flush=True)


if __name__=="__main__":
    main()
