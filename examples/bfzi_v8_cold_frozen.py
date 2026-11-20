"""Extend frozen V7 ColdNet with a four-output copper head using ridge fitting.

V7 shared trunk and nine cold-port outputs are preserved. Copper labels
come from the original V150 cold reference, with no new BFZI solves.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_cold_v7_train import read,circuits,evaluate
from sdfmpneo_bfzi.geom_coldnet import ColdNet,copper_target,feat


@torch.no_grad()
def hidden(net,rows):
    x=np.stack([feat(r["geometry"]) for r in rows])
    t=torch.as_tensor(x,dtype=torch.float32)
    h=net.net[:-1]((t-net.xm)/net.xs).numpy()
    return np.c_[h,np.ones(len(h))].astype(np.float64)


def train(a):
    torch.set_num_threads(2)
    data=read(a.data)
    old=ColdNet.load(a.base)
    tr=[r for r in data["train"] if copper_target(r) is not None]
    va=[r for r in data["val"] if copper_target(r) is not None]
    if len(tr)<100 or len(va)<20:
        raise ValueError("insufficient labeled copper geometries")
    cu=np.stack([copper_target(r) for r in tr])
    ym=cu.mean(axis=0)
    ys=np.maximum(cu.std(axis=0),.0005)
    net=ColdNet(old.xm,old.xs,
                np.r_[old.ym.detach().numpy(),ym],
                np.r_[old.ys.detach().numpy(),ys],
                hidden=old.net[0].out_features)
    with torch.no_grad():
        for i in (0,2,4):
            net.net[i].load_state_dict(old.net[i].state_dict())
        net.net[6].weight[:9].copy_(old.net[6].weight)
        net.net[6].bias[:9].copy_(old.net[6].bias)
    x=hidden(old,tr)
    y=(cu-ym)/ys
    xtx=x.T@x
    xty=x.T@y
    exts=circuits(read(a.circuits))
    results={}
    best=float("inf")
    selected=None
    for lam in (.0001,.01,1.,10.):
        A=xtx+lam*np.diag([1.]*(x.shape[1]-1)+[0.])
        w=np.linalg.solve(A,xty)
        with torch.no_grad():
            net.net[6].weight[9:].copy_(torch.as_tensor(w[:-1].T,dtype=torch.float32))
            net.net[6].bias[9:].copy_(torch.as_tensor(w[-1],dtype=torch.float32))
        stats=evaluate(net,va,exts)
        results[str(lam)]={k:stats[k] for k in
            ("cold_Z_pct","air_L_pct","current_pct",
             "cold_copper_pct","cold_sea_power_pct")}
        score=stats["cold_sea_power_pct"]
        print("V8_FROZEN_VAL",lam,round(score,6),
              "copper",round(stats["cold_copper_pct"],6),flush=True)
        if score<best:
            best=score
            selected=lam
            saved=(net.net[6].weight[9:].clone(),net.net[6].bias[9:].clone())
    with torch.no_grad():
        net.net[6].weight[9:].copy_(saved[0])
        net.net[6].bias[9:].copy_(saved[1])
    z0=old.predict(data["val"][0]["geometry"])[0]
    z1=net.predict(data["val"][0]["geometry"])[0]
    if np.linalg.norm(z0-z1)>1e-5:
        raise RuntimeError("frozen cold-port predictions changed")
    test=evaluate(net,data["test"],exts)
    baseline=evaluate(old,data["test"],exts)
    test={k:test[k] for k in
          ("cold_Z_pct","air_L_pct","current_pct",
           "cold_copper_pct","cold_sea_power_pct")}
    baseline={k:baseline[k] for k in
              ("cold_Z_pct","air_L_pct","current_pct")}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    net.save(a.out)
    result=dict(status="candidate",source=str(a.base),
        train_copper_geometries=len(tr),validation_copper_geometries=len(va),
        selected_ridge=selected,validation=results,
        frozen_v7_test=baseline,candidate_test=test,
        checkpoint=str(a.out))
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V8_FROZEN_FINAL",json.dumps({
        "selected_ridge":selected,"frozen_v7_test":baseline,
        "candidate_test":test}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("runs/geomfrac_cold_v7/data_merged.json"))
    p.add_argument("--circuits",type=Path,default=Path("runs/geomfrac_cold_ports_v1.json"))
    p.add_argument("--base",type=Path,default=Path("runs/geomfrac_cold_v7.pt"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_cold_v8_frozen.json"))
    a=p.parse_args()
    train(a)


if __name__=="__main__":
    main()
