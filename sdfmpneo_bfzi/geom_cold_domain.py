"""Small, explicit applicability gate for the V7 core inference domain."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .geom_coldnet import feat

BOUNDS=[
 ("alpha1",.70,1.),("alpha2",.70,1.),
 ("core1_mm",0.,25.),("core2_mm",0.,25.),
 ("shift_x",-16.,16.),("shift_y",80.,116.),("shift_z",-12.,12.),
 ("tilt_deg",-20.,20.),("yaw_deg",-20.,20.),("pitch_deg",-18.,18.),
 ("box1_mm",210.,242.),("box2_mm",210.,242.)]
FLOOR=np.array([.02,.02,1,1,2,2,2,2,2,2,2,2,.01,.1,.01,.1])
def vector(g):
    return np.array([g["alpha1"],g["alpha2"],g["core1_mm"],g["core2_mm"],
        *g["shift_mm"],g["tilt_deg"],g["yaw_deg"],g["pitch_deg"],
        g["box1_mm"],g["box2_mm"]],float)
def in_core(g):
    x=vector(g)
    return all((v-lo)/(hi-lo)>=.05 and (v-lo)/(hi-lo)<=.95
               for v,(_,lo,hi) in zip(x,BOUNDS))
def fit(train_rows,val_rows):
    tr=[r for r in train_rows if in_core(r["geometry"])]
    va=[r for r in val_rows if in_core(r["geometry"])]
    if len(tr)<20 or len(va)<5:raise ValueError("insufficient core samples to calibrate domain")
    x=np.stack([feat(r["geometry"]) for r in tr])
    xv=np.stack([feat(r["geometry"]) for r in va])
    mu=x.mean(0);sd=np.maximum(x.std(0),FLOOR)
    dist=np.linalg.norm((xv[:,None]-x[None,:])/sd,axis=-1).min(1)
    threshold=float(np.quantile(dist,.95))
    return dict(version=1,method="core bounds + nearest training geometry",
        bounds={n:[lo,hi] for n,lo,hi in BOUNDS},core_margin=.05,
        mean=mu.tolist(),scale=sd.tolist(),max_nn_distance=threshold,
        train_core=len(tr),val_core=len(va),
        val_nn_p50=float(np.quantile(dist,.5)),
        val_nn_p95=threshold)
def check(g,model):
    if not in_core(g):
        return dict(accepted=False,reason="outside_core_bounds",distance=None)
    x=(feat(g)-np.asarray(model["mean"]))/np.asarray(model["scale"])
    tr=np.asarray(model["train_features"])
    d=float(np.linalg.norm((tr-x),axis=1).min())
    ok=d<=float(model["max_nn_distance"])
    return dict(accepted=ok,reason="within_domain" if ok else "sparse_training_coverage",
                distance=d,threshold=float(model["max_nn_distance"]))
def save(model,train_rows,path):
    x=np.stack([feat(r["geometry"]) for r in train_rows if in_core(r["geometry"])])
    x=(x-np.asarray(model["mean"]))/np.asarray(model["scale"])
    model=dict(model);model["train_features"]=x.tolist()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(model,indent=2)+"\n",encoding="utf-8")

def load(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))
