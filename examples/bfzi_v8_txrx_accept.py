"""Complete frozen V8 TX/RX two-geometry acceptance once physics manifest is ready."""
import json,subprocess,sys,time
from pathlib import Path
mf=Path("runs/geomfrac_s5_v8txrx_manifest.json")
for _ in range(1440):
 if mf.exists():
  data=json.loads(mf.read_text(encoding="utf-8-sig"))
  specs=data.get("test",[])
  if (len(specs)==2 and all(len(s.get("teacher",[]))==2
                            for s in specs)):
   break
 time.sleep(5)
else:
 raise RuntimeError("frozen BFZI reference histories incomplete")
print("TXRX_BLIND_READY",len(specs),sum(len(s["teacher"]) for s in specs),flush=True)
script=r"D:\work\geomfrac-v8\examples\bfzi_v8_accept.py"
common=[
"--plan","runs/geomfrac_v8_txrx_holdout_plan.json",
"--manifest",str(mf),
"--split","test",
"--model","runs/geomfrac_et_v8_heat_tune.pt",
"--config","runs/geomfrac_free_rank48_v8_selection.json",
"--skin-tangent","runs/geomfrac_rank48_skin_tan_v3.pt",
"--skin-base","runs/geomfrac_et_v9_ztune.pt",
"--parallel-model","runs/geomfrac_v8_par48.pt",
"--k-mix","0.25"]
for b,n in ((0.,"base"),(.5,"phys")):
 out=f"runs/geomfrac_v8_txrx_holdout_{n}.json"
 subprocess.run([sys.executable,"-u",script,*common,
   "--weak-sea-mix",str(b),"--out",out],check=True)
 d=json.loads(Path(out).read_text(encoding="utf-8-sig"))
 print("TXRX_BLIND_RESULT",n,json.dumps(d["overall"]),flush=True)
b=json.loads(Path("runs/geomfrac_v8_txrx_holdout_base.json").read_text())
p=json.loads(Path("runs/geomfrac_v8_txrx_holdout_phys.json").read_text())
changes={k:{"base":b["overall"][k],"physical":p["overall"][k]}
 for k in ("dynamic_z_rel_L2","R11_dynamic_rel_L2_self",
           "R22_dynamic_rel_L2_self","X11_dynamic_rel_L2_self",
           "X22_dynamic_rel_L2_self","material_mean_rmse_K",
           "current_rel_L2","sea_power_rel_L2","copper_power_rel_L2")}
sel=(p["overall"]["dynamic_z_rel_L2"]<b["overall"]["dynamic_z_rel_L2"]
 and p["overall"]["R11_dynamic_rel_L2_self"] <=
     b["overall"]["R11_dynamic_rel_L2_self"]*1.02
 and p["overall"]["R22_dynamic_rel_L2_self"] <=
     b["overall"]["R22_dynamic_rel_L2_self"]*1.02)
res=dict(status="blind_improved" if sel else "blind_not_improved",
   geometry_count=2,history_count=4,k_mix=.25,weak_sea_mix=.5,
   physical_used_to_select_mix=False,metrics=changes,
   source="V8S0001,V8S0002 new frozen geometry-disjoint BFZI V150 S5 ultra")
Path("runs/geomfrac_v8_txrx_holdout_summary.json").write_text(
 json.dumps(res,indent=2)+"\n",encoding="utf-8")
print("TXRX_BLIND_FINAL",json.dumps(res),flush=True)
