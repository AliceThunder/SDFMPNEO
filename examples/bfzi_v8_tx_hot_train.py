"""Fine-tune existing ParZ48 with high-drive TX histories, then validate closed-loop."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from examples.bfzi_v8_par48 import pack
from sdfmpneo_bfzi.geom_canon import labels
from sdfmpneo_bfzi.geom_free import rollout,geometry
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_skintan import restore_rank48

torch.manual_seed(7432);np.random.seed(7432);torch.set_num_threads(2)
def read(p): return json.loads(Path(p).read_text(encoding="utf-8-sig"))
cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
plan=read("runs/geomfrac_v8_full_plan.json")
hot=read("runs/geomfrac_v8_tx_hot_plan.json")
plan["cases"].update(hot["cases"])
mf=read("runs/geomfrac_s5_v8_full_manifest.json")
mfhot=read("runs/geomfrac_s5_v8txhot_manifest.json")
if len(mfhot["train"])!=4 or any(len(s["teacher"])!=2 for s in mfhot["train"]):
    raise RuntimeError("high drive physical reference incomplete")
mf["train"].extend(mfhot["train"])
ck=torch.load("runs/geomfrac_et_v8_heat_tune.pt",map_location="cpu",weights_only=True)
net=GeomFracET(ng=int(ck["ng"]),rank=48,soe_quad=ck.get("soe_quad","trap"),
 heat_dt=float(ck.get("heat_dt",1.875)),heat_z=bool(ck.get("heat_z",False)))
net.load_state_dict(ck["state"]);net.eval()
atlas=GeomInit.load(cfg["atlas"])
cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
sf=SrcFix.load(cfg["source_model"])
tan=restore_rank48("runs/geomfrac_rank48_skin_tan_v3.pt","runs/geomfrac_et_v9_ztune.pt")
model=ParZ48.load("runs/geomfrac_v8_par48.pt")
old=copy.deepcopy(model);old.eval()
mix=float(cfg["shape_mix"])

def collect(split):
    rows=[]; names=[]
    with torch.no_grad():
        for spec in mf[split]:
            n=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
            case=plan["cases"][n]
            init=thermal_mix(atlas,sf.apply(case,cold.apply(case,atlas.predict(case))),.25)
            _,g=geometry(case,init)
            names.append(n)
            for row in prepare(spec,48):
                pred=rollout(case,atlas,net,ck,row["time"],row["voltage"],tan,
                     init_override=init,shape_mix=mix)
                dr=row["z6"]-row["z6"][0]
                dz=pred["z6"]-pred["z6"][0]
                rows.append(dict(name=n,case=case,init=init,row=row,g=g,
                    h=pred["heat_state"],dr=dr,dz=dz,target=dr-dz))
            if len(names)%12==0:
                print("TXHOT_COLLECT",split,len(names),len(rows),flush=True)
    return rows,names

tr,tn=collect("train")
va,vn=collect("val")
if (len(tn),len(vn),len(tr),len(va))!=(52,15,140,41) or set(tn)&set(vn):
    raise RuntimeError("TX high-drive partition mismatch")
gt,at,tt,vt,yt=pack(tr)
ys=model.ys.clamp_min(1e-6)
weights=[]
for r in tr:
    num=len(r["h"])
    # Upweight new high-drive histories; keep all original training geometries.
    w=3. if r["name"].startswith("V8X") else 1.
    weights.extend([w]*num)
w=torch.as_tensor(weights,dtype=torch.float32)
with torch.no_grad():
    prior=old(gt,at,tt,vt)
opt=torch.optim.AdamW(model.parameters(),lr=8e-5,weight_decay=.002)
fields=("dynamic_z_rel_L2","R11_dynamic_error_L2_over_total_delta",
 "X11_dynamic_error_L2_over_total_delta","current_rel_L2",
 "material_mean_rmse_K","sea_power_rel_L2","copper_power_rel_L2")
def score(mod):
    metrics={k:[] for k in fields}
    with torch.no_grad():
        for r in va:
            p=rollout(r["case"],atlas,net,ck,r["row"]["time"],r["row"]["voltage"],tan,
                      init_override=r["init"],extra=mod,shape_mix=mix)
            m=metric(p,r["row"])
            for k in fields: metrics[k].append(float(m[k]))
    return {k:float(np.mean(v)) for k,v in metrics.items()}

baseline=score(old)
print("TXHOT_BASE",json.dumps(baseline),flush=True)
best=baseline
best_sd=copy.deepcopy(model.state_dict())
best_ep=0;hist=[]
for ep in range(1,101):
    model.train()
    for ix in torch.randperm(len(gt)).split(192):
        pred=model(gt[ix],at[ix],tt[ix],vt[ix])
        loss=((((pred-yt[ix])/ys).square()*
               torch.tensor([2.5,1.25,1.,1.,1.,1.]))*w[ix,None]).mean()
        loss+=.05*(((pred-prior[ix])/ys).square()).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),.5)
        opt.step()
    if ep%10: continue
    model.eval()
    val=score(model)
    hist.append(dict(epoch=ep,train_loss=float(loss.detach()),**val))
    print("TXHOT_EVAL",ep,json.dumps(val),flush=True)
    good=(val["dynamic_z_rel_L2"]<best["dynamic_z_rel_L2"]-.0005
        and val["R11_dynamic_error_L2_over_total_delta"]<
          baseline["R11_dynamic_error_L2_over_total_delta"]*.99
        and val["current_rel_L2"]<=baseline["current_rel_L2"]*1.03
        and val["sea_power_rel_L2"]<=baseline["sea_power_rel_L2"]*1.05
        and val["material_mean_rmse_K"]<=baseline["material_mean_rmse_K"]*1.03)
    if good:
        best=val;best_ep=ep;best_sd=copy.deepcopy(model.state_dict())
    if ep-best_ep>=50:break
model.load_state_dict(best_sd)
status="candidate" if best_ep else "not_promoted"
if best_ep: model.save("runs/geomfrac_v8_tx_hot.pt")
report=dict(status=status,base=baseline,best=best,best_epoch=best_ep,
   epochs_trained=ep,train_geometries=48,train_geometry_cases=len(tn),
   train_histories=len(tr),val_geometries=15,val_histories=len(va),
   k_mix=.25,hot_plan="runs/geomfrac_v8_tx_hot_plan.json",
   method="TX-weighted fine-tuning of existing ParZ48; no new network",history=hist)
Path("runs/geomfrac_v8_tx_hot.json").write_text(json.dumps(report,indent=2)+"\n")
print("TXHOT_RESULT",json.dumps({k:report[k] for k in
   ("status","base","best","best_epoch","epochs_trained")}),flush=True)
