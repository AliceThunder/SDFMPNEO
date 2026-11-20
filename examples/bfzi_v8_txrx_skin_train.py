"""Physics-supervised SkinTan V3 retraining (frozen base, ParZ48 and heat)."""
from pathlib import Path
import copy,json
import numpy as np
import torch
from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_free import geometry,rollout
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_skintan import SkinTanV3
from sdfmpneo_bfzi.geom_ztune import restore as load_base
from sdfmpneo_bfzi.geom_par48 import ParZ48

torch.manual_seed(8401);np.random.seed(8401);torch.set_num_threads(2)
def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
pl=read("runs/geomfrac_v8_full_plan.json")
hot=read("runs/geomfrac_v8_tx_hot_plan.json")
pl["cases"].update(hot["cases"])
mf=read("runs/geomfrac_s5_v8_full_manifest.json")
mf["train"].extend(read("runs/geomfrac_s5_v8txhot_manifest.json")["train"])
ck=torch.load("runs/geomfrac_et_v8_heat_tune.pt",map_location="cpu",weights_only=True)
net=GeomFracET(ng=int(ck["ng"]),rank=48,soe_quad=ck.get("soe_quad","trap"),
               heat_dt=float(ck.get("heat_dt",1.875)),
               heat_z=bool(ck.get("heat_z",False)))
net.load_state_dict(ck["state"]);net.eval()
atlas=GeomInit.load(cfg["atlas"])
cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
src=SrcFix.load(cfg["source_model"])
base=load_base("runs/geomfrac_et_v9_ztune.pt")
par=ParZ48.load("runs/geomfrac_v8_par48.pt")
sc=torch.load("runs/geomfrac_rank48_skin_tan_v3.pt",map_location="cpu",weights_only=True)
skin=SkinTanV3(ng=int(sc["ng"]),sa=np.asarray(sc["sa"]),ss=np.asarray(sc["ss"]))
skin.load_state_dict(sc["state"])
orig=copy.deepcopy(skin);orig.eval()
ys=torch.as_tensor(np.asarray(sc["sy"])[:6],dtype=torch.float32)
shape=float(cfg["shape_mix"])
p=Path("runs/geomfrac_v8_txrx_fd_train.npz")
with np.load(p) as d:
 dg=torch.as_tensor(d["g"],dtype=torch.float32)
 da=torch.as_tensor(d["a"],dtype=torch.float32)
 dy=torch.as_tensor(d["sea"],dtype=torch.float32)
 dt=torch.tensor([[300.]]*len(dg),dtype=torch.float32)
 # Use true corresponding drive (hot training plan), no history-level leakage.
 dv=torch.tensor([[hot["cases"][name]["drives"][1]["voltage"]]
                  for name in d["geometry"]],dtype=torch.float32)
 labels=list(d["mode"])
 names=list(d["geometry"])
if len(dg)!=44 or len(set(names))!=4:
 raise RuntimeError("BFZI fixed-temperature data incomplete")
pairs=[]
for i in range(len(labels)):
 if labels[i].endswith("_plus"):
  if i+1>=len(labels) or labels[i+1]!=labels[i].replace("_plus","_minus"):
   raise ValueError("BFZI FD pair ordering incorrect")
  kind=labels[i][:-5]
  if kind in ("skin_tx","skin_rx","wet_all"):
   pairs.append((i,i+1,kind))
if len(pairs)!=12:raise RuntimeError("not all physical sensitivity pairs loaded")

def fixed(g,a,t,v):
 with torch.no_grad():
  return base(g,a[:,:32])+par(g,a,t,v)

def select_rows(split):
 dat=[];names_out=[]
 for spec in mf[split]:
  name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
  c=pl["cases"][name]
  init=thermal_mix(atlas,src.apply(c,cold.apply(c,atlas.predict(c))),.25)
  g=geometry(c,init)[1]
  hist=prepare(spec,48)
  names_out.append(name)
  for row in hist:
   dat.append(dict(case=c,init=init,row=row,g=g,name=name))
  if len(names_out)%12==0:print("PHYS_MAP_COLLECT",split,len(names_out),len(dat),flush=True)
 return dat,names_out
tr,tn=select_rows("train")
va,vn=select_rows("val")
if (len(tn),len(vn),len(tr),len(va))!=(52,15,140,41) or set(tn)&set(vn):
 raise RuntimeError("train validation geometry split mismatch")
G=[];A=[];T=[];V=[];Y=[];W=[]
for r in tr:
 row=r["row"];n=len(row["a"])
 G.append(np.broadcast_to(r["g"],(n,45)).copy())
 A.append(np.asarray(row["a"],dtype=np.float32))
 T.append(np.asarray(row["time"][:,None],dtype=np.float32))
 V.append(np.asarray(row["voltage"][:,None],dtype=np.float32))
 Y.append(np.asarray(row["y"][:,:6],dtype=np.float32))
 weight=(2. if r["name"].startswith("V8X") else 1.)/n
 W.extend([weight]*n)
g=torch.tensor(np.concatenate(G),dtype=torch.float32)
a=torch.tensor(np.concatenate(A),dtype=torch.float32)
t=torch.tensor(np.concatenate(T),dtype=torch.float32)
v=torch.tensor(np.concatenate(V),dtype=torch.float32)
truth=torch.tensor(np.concatenate(Y),dtype=torch.float32)
w=torch.tensor(np.asarray(W),dtype=torch.float32)
w=w/w.mean()
with torch.no_grad():
 target=(truth-fixed(g,a,t,v))/ys
 ft=(dy-fixed(dg,da,dt,dv))/ys
 prior=orig(g,a[:,:32],a[:,32:])
 fp=orig(dg,da[:,:32],da[:,32:])
def predict_tan(g,a):
 return base(g,a[:,:32])+skin(g,a[:,:32],a[:,32:])*ys
predict_tan.time_conditioned=False

def validate():
 vals=[]
 with torch.no_grad():
  skin.eval()
  for r in va:
   row=r["row"]
   p=rollout(r["case"],atlas,net,ck,row["time"],row["voltage"],
      predict_tan,init_override=r["init"],extra=par,shape_mix=shape)
   m=metric(p,row)
   z=p["z6"];zr=row["z6"]
   dz=z-z[0];dr=zr-zr[0]
   m["r11_self"]=float(np.linalg.norm((dz-dr)[:,0])/
                  max(np.linalg.norm(dr[:,0]),1e-12))
   m["r22_self"]=float(np.linalg.norm((dz-dr)[:,4])/
                  max(np.linalg.norm(dr[:,4]),1e-12))
   vals.append(m)
 return {k:float(np.mean([r[k] for r in vals])) for k in
   ("dynamic_z_rel_L2","R11_dynamic_error_L2_over_total_delta",
    "R22_dynamic_error_L2_over_total_delta","r11_self","r22_self",
    "material_mean_rmse_K","current_rel_L2","sea_power_rel_L2")}
def em_val():
    scores={"sea_rel":[], "tx_rel":[], "rx_rel":[]}
    with torch.no_grad():
      skin.eval()
      for r in va:
        row=r["row"]; n=len(row["a"])
        gg=torch.as_tensor(np.broadcast_to(r["g"],(n,45)).copy(),dtype=torch.float32)
        aa=torch.as_tensor(row["a"].copy(),dtype=torch.float32)
        tt=torch.as_tensor(row["time"][:,None].copy(),dtype=torch.float32)
        vv=torch.as_tensor(row["voltage"][:,None].copy(),dtype=torch.float32)
        pred=fixed(gg,aa,tt,vv)+skin(gg,aa[:,:32],aa[:,32:])*ys
        truth=torch.as_tensor(row["y"][:,:6].copy(),dtype=torch.float32)
        err=(pred-truth).numpy(); ref=truth.numpy()
        scores["sea_rel"].append(float(np.linalg.norm(err)/max(np.linalg.norm(ref),1e-12)))
        scores["tx_rel"].append(float(np.linalg.norm(err[:,0])/max(np.linalg.norm(ref[:,0]),1e-12)))
        scores["rx_rel"].append(float(np.linalg.norm(err[:,4])/max(np.linalg.norm(ref[:,4]),1e-12)))
    return {k:float(np.mean(v)) for k,v in scores.items()}

def fd_score():
 with torch.no_grad():
  pred=skin(dg,da[:,:32],da[:,32:])
  comps=[]
  for i,j,kind in pairs:
   d=(pred[i]-pred[j])-(ft[i]-ft[j])
   comps.append(float(torch.linalg.norm(d*ys)))
  return float(np.mean(comps))
baseline=validate();base_fd=fd_score();em_base=em_val()
print("PHYS_MAP_BASE",json.dumps(baseline),"fd_err_ohm",base_fd,"teacher_em",json.dumps(em_base),flush=True)
opt=torch.optim.AdamW(skin.parameters(),lr=0.00012,weight_decay=.001)
best=baseline;best_epoch=0
best_weights=copy.deepcopy(skin.state_dict())
history=[]
for ep in range(1,181):
 skin.train()
 opt.zero_grad(set_to_none=True)
 yp=skin(g,a[:,:32],a[:,32:])
 lhist=(w[:,None]*(yp-target).square()).mean()
 fd_pred=skin(dg,da[:,:32],da[:,32:])
 differ=[]
 for i,j,kind in pairs:
  d=(fd_pred[i]-fd_pred[j])-(ft[i]-ft[j])
  # Express physically meaningful local susceptibility in absolute ohms.
  diffscale=(3e-4 if kind!="wet_all" else 5e-3)
  differ.append(((d*ys/diffscale)**2).mean())
 lfd=torch.stack(differ).mean()
 reg=((yp-prior)**2).mean()+.2*((fd_pred-fp)**2).mean()
 loss=lhist+0.4*lfd+.03*reg
 loss.backward()
 torch.nn.utils.clip_grad_norm_(skin.parameters(),1.)
 opt.step()
 if ep%15:continue
 report=validate()
 fde=fd_score()
 em=em_val()
 info=dict(epoch=ep,train=float(lhist.detach()),fd=float(lfd.detach()),
           fd_err_ohm=fde,oracle_sea_rel=em["sea_rel"],
           oracle_tx_rel=em["tx_rel"],oracle_rx_rel=em["rx_rel"],**report)
 history.append(info)
 print("PHYS_MAP_EVAL",json.dumps(info),flush=True)
 good=(report["dynamic_z_rel_L2"]<best["dynamic_z_rel_L2"]-.001
    and report["r11_self"]<=baseline["r11_self"]*.98
    and report["r22_self"]<=baseline["r22_self"]*1.03
    and report["material_mean_rmse_K"]<=baseline["material_mean_rmse_K"]*1.03
    and report["current_rel_L2"]<=baseline["current_rel_L2"]*1.03
    and report["sea_power_rel_L2"]<=baseline["sea_power_rel_L2"]*1.05)
 if good:
  best=report;best_epoch=ep;best_weights=copy.deepcopy(skin.state_dict())
 if ep-best_epoch>=75:break
probe=copy.deepcopy(sc);probe["state"]=copy.deepcopy(skin.state_dict())
torch.save(probe,"runs/geomfrac_v8_txrx_skin_probe.pt")
em_last=em_val()
skin.load_state_dict(best_weights)
status="candidate" if best_epoch else "not_promoted"
if best_epoch:
 ckout=copy.deepcopy(sc);ckout["state"]=skin.state_dict()
 torch.save(ckout,"runs/geomfrac_v8_txrx_skin.pt")
report=dict(status=status,base=baseline,best=best,best_epoch=best_epoch,
 epochs_trained=ep,fd_base_ohm=base_fd,fd_final_ohm=fd_score(),history=history,
 em_base=em_base,em_probe=em_last,probe_checkpoint="runs/geomfrac_v8_txrx_skin_probe.pt",
 sample_count=len(g),fd_pairs=len(pairs),k_mix=.25,
 method="same SkinTan V3 full weights, physical true-thermal-sea target and BFZI FD sensitivity")
Path("runs/geomfrac_v8_txrx_skin.json").write_text(
 json.dumps(report,indent=2)+"\n",encoding="utf-8")
print("PHYS_MAP_RESULT",json.dumps({k:report[k] for k in
  ("status","base","best","best_epoch","epochs_trained","fd_final_ohm")}),flush=True)
