"""Fit cold-geometry copper/package contact modes without dynamic teacher data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_canon import labels,make_rom
from sdfmpneo_bfzi.geom_init import GeomInit,_rbf
from sdfmpneo_bfzi.geom_culink import CuLink,xport


def contact(spec,case):
    head=GeomHeat.load(spec["anchor"])
    rom=make_rom(head,32)
    cap=np.array([head.cap[:head.ns].sum(),
        head.cap[head.ns:head.ns+head.npkg].sum(),
        head.cap[head.ns+head.npkg:head.cu.start].sum(),
        *head.cap[head.cu]])
    m=np.sqrt(cap[labels(32)])
    h=rom.K*m[:,None]*m[None,:]
    return np.stack([-h[3,18:25]/h[3,3],
                     -h[4,25:32]/h[4,4]])


def data(specs,cases):
    out=[]
    for spec in specs:
        name=Path(spec["anchor"]).stem.split("_")[2]
        out.append((cases[name],contact(spec,cases[name])))
    return out


def tune(train,dev):
    x=np.stack([xport(c) for c,_ in train],axis=1)
    y=np.stack([v for _,v in train],axis=1)
    xx=np.stack([xport(c) for c,_ in dev],axis=1)
    yy=np.stack([v for _,v in dev],axis=1)
    params=[]
    for j in (0,1):
        mu=x[j].mean(0)
        std=np.maximum(x[j].std(0),[.01,1.,2.])
        tr=(x[j]-mu)/std
        va=(xx[j]-mu)/std
        mean=y[j].mean(0)
        best=(float("inf"),None,None)
        for ell in (.4,.7,1.,1.5,2.,3.,5.,8.):
            k=_rbf(tr,tr,ell)
            kv=_rbf(va,tr,ell)
            for lam in (1e-5,1e-4,.001,.01,.05,.2):
                b=np.linalg.solve(k+lam*np.eye(len(tr)),y[j]-mean)
                pred=mean+kv@b
                err=float(np.sqrt(np.mean((pred-yy[j])**2)))
                if err<best[0]:
                    best=(err,ell,lam)
        params.append(best)
    return params


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,default=Path("runs/geomfrac_s5_v4_plan.json"))
    p.add_argument("--extra-plan",type=Path,default=Path("runs/geomfrac_s5_free8_plan.json"))
    p.add_argument("--manifest",type=Path,default=Path("runs/geomfrac_s5_v5_36g_manifest.json"))
    p.add_argument("--atlas",type=Path,default=Path("runs/geomfrac_init_v1.npz"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_culink_v1.npz"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_culink_v1_dev.json"))
    a=p.parse_args()
    plan=json.loads(a.plan.read_text(encoding="utf-8-sig"))
    ext=json.loads(a.extra_plan.read_text(encoding="utf-8-sig"))
    mf=json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    cases={**plan["cases"],**ext["cases"]}
    tr=data(mf["train"],cases)
    dev=data(mf["val"],cases)
    if set(map(lambda r:r["anchor"],mf["train"])) & set(
            map(lambda r:r["anchor"],mf["val"])):
        raise ValueError("train and development geometry must be disjoint")
    params=tune(tr,dev)
    model=CuLink.fit(tr,ell=[x[1] for x in params],
                        lam=[x[2] for x in params])
    atlas=GeomInit.load(a.atlas)
    err_new=[]
    err_old=[]
    details=[]
    lab=labels(32)
    for case,ref in dev:
        got=model.predict(case)
        init=atlas.predict(case)
        m=np.sqrt(init["cap"][lab])
        h=init["k"]*m[:,None]*m[None,:]
        old=np.stack((-h[3,18:25]/h[3,3],-h[4,25:32]/h[4,4]))
        err_new.append(np.sqrt(np.mean((got-ref)**2,axis=1)))
        err_old.append(np.sqrt(np.mean((old-ref)**2,axis=1)))
        details.append(dict(geometry=case["geometry"],
                            old_rmse=np.sqrt(np.mean((old-ref)**2,axis=1)).tolist(),
                            new_rmse=np.sqrt(np.mean((got-ref)**2,axis=1)).tolist()))
    result=dict(status="development_selected_copper_contact_profile",
        train_geometries=len(tr),dev_geometries=len(dev),
        train_history_dependency=False,
        port_params=[dict(rmse=x[0],ell=x[1],lam=x[2]) for x in params],
        atlas_dev_rmse=np.mean(err_old,axis=0).tolist(),
        learned_dev_rmse=np.mean(err_new,axis=0).tolist(),
        details=details)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    model.save(a.out)
    a.report.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("CU_LINK_DEV",result["atlas_dev_rmse"],
          "->",result["learned_dev_rmse"],flush=True)


if __name__=="__main__":
    main()
