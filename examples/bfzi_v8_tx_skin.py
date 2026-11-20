"""Fit TX SkinTan V3 output layer to true-temperature impedance response."""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_canon import labels
from sdfmpneo_bfzi.geom_free import rollout,geometry,circuit
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_skintan import SkinTanV3,encode_geom_v3,restore_rank48,IDX
from sdfmpneo_bfzi.geom_memory import geom_port
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_heat import pack

torch.set_num_threads(2)
def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))
cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
pl=read("runs/geomfrac_v8_full_plan.json")
hp=read("runs/geomfrac_v8_tx_hot_plan.json")
pl["cases"].update(hp["cases"])
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
skin_path="runs/geomfrac_rank48_skin_tan_v3.pt"
base_path="runs/geomfrac_et_v9_ztune.pt"
sc=torch.load(skin_path,map_location="cpu",weights_only=True)
sn=SkinTanV3(ng=int(sc["ng"]),sa=np.asarray(sc["sa"]),
             ss=np.asarray(sc["ss"]))
sn.load_state_dict(sc["state"]);sn.eval()
tan=restore_rank48(skin_path,base_path)
par=ParZ48.load("runs/geomfrac_v8_par48.pt")
sy=np.asarray(sc["sy"])[:2]
shape=float(cfg["shape_mix"])
out_path=Path("runs/geomfrac_v8_tx_skin.pt")

def hidden(g,a):
    with torch.no_grad():
        G=torch.as_tensor(np.broadcast_to(g,(len(a),45)).copy(),dtype=torch.float32)
        A=torch.as_tensor(a,dtype=torch.float32)
        fg=encode_geom_v3(G)
        z=torch.cat((fg,torch.zeros_like(A[:,IDX]/sn.sa),
                     torch.zeros_like(A[:,32:40]/sn.ss[:8])),dim=1)
        x=torch.cat((fg,A[:,IDX]/sn.sa,A[:,32:40]/sn.ss[:8]),dim=1)
        return (sn.tx[:-1](x)-sn.tx[:-1](z)).numpy().astype(float)

def collect(specs,split):
    rows=[]; names=[]
    for spec in specs:
        n=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
        case=pl["cases"][n]
        init=thermal_mix(atlas,src.apply(case,cold.apply(case,atlas.predict(case))),.25)
        g=geometry(case,init)[1]
        rows.append((n,case,init,prepare(spec,48),g))
        names.append(n)
        if len(names)%12==0:
            print("SKIN_COLLECT",split,len(names),flush=True)
    return rows,names

train,tn=collect(mf["train"],"train")
val,vn=collect(mf["val"],"val")
if len(train)!=52 or len(val)!=15 or set(tn)&set(vn):
    raise RuntimeError("invalid V8 TX skin training partition")
X=[];Y=[];S=[];F=[]
with torch.no_grad():
 for name,case,init,hists,g in train:
    port=geom_port(geometry(case,init)[0])
    scale=np.sqrt(init["cap"][labels(48)])
    gt=torch.as_tensor(g[None],dtype=torch.float32)
    for row in hists:
        a=np.asarray(row["a"],float)
        def extra(state,tm,vi):
            return par(gt,state,
              torch.tensor([[tm]],dtype=torch.float32),
              torch.tensor([[vi]],dtype=torch.float32))[0].detach().numpy()
        zz=[]
        for state,t,v in zip(a,row["time"],row["voltage"]):
            z,_,_,_,_=circuit(state*scale,init,port,state*scale,
                              tan,gt,scale,float(v),293.15,extra,float(t))
            zz.append(pack(z))
        zz=np.asarray(zz)
        target=(row["z6"]-row["z6"][0])-(zz-zz[0])
        feat=hidden(g,a)
        X.append(feat)
        Y.append(target[:,:2])
        den=max(float(np.linalg.norm(row["z6"]-row["z6"][0])),1e-9)
        w=(2. if name.startswith("V8X") else 1.)/(den*np.sqrt(len(a)))
        S.extend([w]*len(a))
        F.append(name)
    if len(F)%36==0: print("SKIN_TARGET",len(F),flush=True)
X=np.concatenate(X)
Y=np.concatenate(Y)
S=np.asarray(S)
# Scale to output-unit normalizations; regularize around frozen original.
Xs=X*S[:,None]
T=Y*S[:,None]/sy[None,:]
M=Xs.T@Xs
B=Xs.T@T
diag=float(np.trace(M)/len(M))
print("SKIN_FIT",len(X),diag,flush=True)
keys=("dynamic_z_rel_L2","R11_dynamic_error_L2_over_total_delta",
      "X11_dynamic_error_L2_over_total_delta","current_rel_L2",
      "material_mean_rmse_K","sea_power_rel_L2","copper_power_rel_L2")

def evaluate(path):
    t=restore_rank48(path,base_path)
    acc={k:[] for k in keys}
    with torch.no_grad():
        for n,case,init,hists,g in val:
            for row in hists:
                p=rollout(case,atlas,net,ck,row["time"],row["voltage"],
                    t,init_override=init,extra=par,shape_mix=shape)
                m=metric(p,row)
                for k in keys:acc[k].append(float(m[k]))
    return {k:float(np.mean(v)) for k,v in acc.items()}
baseline=evaluate(skin_path)
print("SKIN_BASE",json.dumps(baseline),flush=True)
best=baseline;option=None
trials=[]
for lam in (0.1,1.,10.):
    beta=np.linalg.solve(M+np.eye(M.shape[0])*diag*lam,B)
    for channels,amp in [([0],.5),([0],1.),([0,1],.5)]:
        cand=copy.deepcopy(sc)
        for j in channels:
            cand["state"]["tx.4.weight"][j]+=torch.as_tensor(
                beta[:,j]*amp,dtype=torch.float32)
        temp=Path("runs/geomfrac_v8_tx_skin_trial.pt")
        torch.save(cand,temp)
        res=evaluate(temp)
        entry=dict(lam=lam,channels=channels,amp=amp,**res)
        trials.append(entry)
        print("SKIN_TRIAL",json.dumps(entry),flush=True)
        if (res["dynamic_z_rel_L2"]<best["dynamic_z_rel_L2"]-.001
            and res["R11_dynamic_error_L2_over_total_delta"] <
              baseline["R11_dynamic_error_L2_over_total_delta"]*.99
            and res["current_rel_L2"]<=baseline["current_rel_L2"]*1.03
            and res["sea_power_rel_L2"]<=baseline["sea_power_rel_L2"]*1.05
            and res["material_mean_rmse_K"]<=baseline["material_mean_rmse_K"]*1.03):
            best=res;option=(lam,channels,amp)
            torch.save(cand,out_path)
report=dict(status="candidate" if option else "not_promoted",
  train_geometries=48,train_geometry_cases=52,train_histories=140,
  val_geometries=15,val_histories=41,
  baseline=baseline,selected=best,choice=option,trials=trials,
  checkpoint=str(out_path) if option else None,
  method="Teacher-thermal-state TX SkinTan frozen-trunk linear calibration")
Path("runs/geomfrac_v8_tx_skin.json").write_text(
    json.dumps(report,indent=2)+"\n",encoding="utf-8")
print("SKIN_RESULT",json.dumps({k:report[k] for k in
       ("status","choice","baseline","selected")}),flush=True)
