"""Physically weighted BFZI cold Joule source for V8 sea resistance."""
import json
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_canon import labels
from sdfmpneo_bfzi.geom_free import rollout,geometry
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_par48 import ParZ48
torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
pl=read("runs/geomfrac_v8_full_plan.json")
mf=read("runs/geomfrac_s5_v8_full_manifest.json")
ck=torch.load("runs/geomfrac_et_v8_heat_tune.pt",map_location="cpu",weights_only=True)
net=GeomFracET(ng=int(ck["ng"]),rank=48,soe_quad=ck.get("soe_quad","trap"),
 heat_dt=float(ck.get("heat_dt",1.875)),heat_z=bool(ck.get("heat_z",False)))
net.load_state_dict(ck["state"]);net.eval()
atlas=GeomInit.load(cfg["atlas"])
cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
sf=SrcFix.load(cfg["source_model"])
tan=restore_rank48("runs/geomfrac_rank48_skin_tan_v3.pt","runs/geomfrac_et_v9_ztune.pt")
par=ParZ48.load("runs/geomfrac_v8_par48.pt")
shape=float(cfg["shape_mix"])
parts=(0,2,4)
keys=("dynamic_z_rel_L2","R11_dynamic_error_L2_over_total_delta",
      "R22_dynamic_error_L2_over_total_delta","current_rel_L2",
      "material_mean_rmse_K","sea_power_rel_L2","copper_power_rel_L2")
cases=[]
for spec in mf["val"]:
 name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
 case=pl["cases"][name]
 init=thermal_mix(atlas,sf.apply(case,cold.apply(case,atlas.predict(case))),.25)
 _,feat=geometry(case,init)
 rows=prepare(spec,48)
 cases.append(dict(name=name,case=case,init=init,
                   g=torch.as_tensor(feat[None],dtype=torch.float32),rows=rows))
 if len(cases)%5==0:print("WEAK_PREP",len(cases),flush=True)
if len(cases)!=15:raise RuntimeError("geometry split count")
def extra(g,init,case,beta):
 scale=np.sqrt(np.asarray(init["cap"])[labels(48)])
 srcmat=np.asarray(init["src"])
 alpha=float(case["alpha_per_K"])
 def call(eg,st,tm,volt):
  with torch.no_grad():
   t=tm
   v=volt
   parvec=par(eg,st,t,v)[0].cpu().numpy()
   if beta==0:return torch.as_tensor(parvec[None],dtype=torch.float32)
   old=tan(eg,st)[0].cpu().numpy()
   phy=alpha*((st[0].cpu().numpy()*scale)@srcmat)
   ans=parvec.copy()
   for j,k in enumerate(parts):
    ans[k]=(1-beta)*parvec[k]+beta*(phy[j]-old[k])
   return torch.as_tensor(ans[None],dtype=torch.float32)
 call.shared_geom=True
 return call
def oracle():
 rr={k:[] for k in ("net_R11","phy_R11","net_R22","phy_R22","net_R12","phy_R12")}
 for r in cases:
  sc=np.sqrt(np.asarray(r["init"]["cap"])[labels(48)])
  src=np.asarray(r["init"]["src"])
  a=float(r["case"]["alpha_per_K"])
  for row in r["rows"]:
   h=np.asarray(row["a"])
   ph=a*(h*sc[None,:])@src
   n=len(h)
   gg=r["g"].repeat(n,1)
   with torch.no_grad():
    nn=tan(gg,torch.as_tensor(h,dtype=torch.float32)).cpu().numpy()
    t=torch.as_tensor(row["time"][:,None].copy(),dtype=torch.float32)
    v=torch.as_tensor(row["voltage"][:,None].copy(),dtype=torch.float32)
    nn+=par(gg,torch.as_tensor(h,dtype=torch.float32),t,v).cpu().numpy()
   tr=np.asarray(row["y"][:,:6])
   for j,sym in enumerate(("R11","R12","R22")):
    k=parts[j];den=max(np.linalg.norm(tr[:,k]),1e-12)
    rr["net_"+sym].append(float(np.linalg.norm(nn[:,k]-tr[:,k])/den))
    rr["phy_"+sym].append(float(np.linalg.norm(ph[:,j]-tr[:,k])/den))
 return {k:float(np.mean(v)) for k,v in rr.items()}

def validate(beta):
 vals={k:[] for k in keys}
 own={"r11":[],"r22":[]}
 with torch.no_grad():
  for r in cases:
   ex=extra(r["g"],r["init"],r["case"],beta)
   for row in r["rows"]:
    p=rollout(r["case"],atlas,net,ck,row["time"],row["voltage"],
      tan,init_override=r["init"],extra=ex,shape_mix=shape)
    m=metric(p,row)
    for k in keys:vals[k].append(float(m[k]))
    dr=np.asarray(row["z6"])-row["z6"][0]
    dz=p["z6"]-p["z6"][0]
    for key,j in (("r11",0),("r22",4)):
     own[key].append(float(np.linalg.norm((dz-dr)[:,j])/
                            max(np.linalg.norm(dr[:,j]),1e-12)))
 out={k:float(np.mean(v)) for k,v in vals.items()}
 out.update({k+"_self":float(np.mean(v)) for k,v in own.items()})
 return out
ref=oracle()
print("WEAK_ORACLE",json.dumps(ref),flush=True)
scores={}
for beta in (0.,.25,.5,.75,1.):
 res=validate(beta)
 scores[str(beta)]=res
 print("WEAK_CLOSED",beta,json.dumps(res),flush=True)
base=scores["0.0"]
chosen=None;best=base
for key,r in scores.items():
 if (r["dynamic_z_rel_L2"]<best["dynamic_z_rel_L2"]-.001
     and r["r11_self"]<base["r11_self"]*.99
     and r["r22_self"]<=base["r22_self"]*1.03
     and r["current_rel_L2"]<=base["current_rel_L2"]*1.03
     and r["material_mean_rmse_K"]<=base["material_mean_rmse_K"]*1.03
     and r["sea_power_rel_L2"]<=base["sea_power_rel_L2"]*1.05):
  best=r;chosen=float(key)
rep=dict(status="candidate" if chosen is not None else "not_promoted",
      geometry_count=15,history_count=41,
      physical="alpha * projected cold BFZI weak Joule source; no new network",
      baseline=base,oracle_em=ref,selected_beta=chosen,best=best,trials=scores)
Path("runs/geomfrac_v8_txrx_weak_eval.json").write_text(
 json.dumps(rep,indent=2)+"\n",encoding="utf-8")
print("WEAK_RESULT",json.dumps({k:rep[k] for k in
 ("status","oracle_em","selected_beta","baseline","best")}),flush=True)
