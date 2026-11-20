"""Train V9 hot-state impedance residual with geometry-disjoint validation."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sdfmpneo_bfzi.geom_hotz import HotResidual
from sdfmpneo_bfzi.geom_skintan import restore_rank48


def rel(pred,true):
    return float(np.linalg.norm(pred-true)/max(np.linalg.norm(true),1e-12))


def metrics(pred,true,ids):
    names=("R11","X11","R12","X12","R22","X22")
    out={"all":rel(pred,true),"macro_geometry":None,"channels":{},"geometry":{}}
    for j,n in enumerate(names):
        out["channels"][n]=rel(pred[:,j],true[:,j])
    for name in np.unique(ids):
        m=ids==name
        out["geometry"][str(name)]={"count":int(m.sum()),"rel_L2":rel(pred[m],true[m])}
    out["macro_geometry"]=float(np.mean([v["rel_L2"] for v in out["geometry"].values()]))
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,nargs="+",default=[Path("runs/geomfrac_v9_hotstates_multi.npz")])
    p.add_argument("--skin-tangent",type=Path,default=Path("runs/geomfrac_rank48_skin_tan_v3.pt"))
    p.add_argument("--skin-base",type=Path,default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_v9_hotz.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_v9_hotz_train.json"))
    p.add_argument("--epochs",type=int,default=600)
    p.add_argument("--patience",type=int,default=80)
    p.add_argument("--hidden",type=int,default=96)
    p.add_argument("--seed",type=int,default=1709)
    a=p.parse_args()
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    torch.set_num_threads(2)
    parts=[]
    for path in a.data:
        with np.load(path) as d:
            parts.append((np.asarray(d["g"],np.float32),
                          np.asarray(d["a"],np.float32),
                          np.asarray(d["sea"],np.float32),
                          np.asarray(d["name"]).astype(str)))
    g=np.concatenate([q[0] for q in parts])
    x=np.concatenate([q[1] for q in parts])
    y=np.concatenate([q[2] for q in parts])
    ids=np.concatenate([q[3] for q in parts])
    if (g.shape!=(len(ids),45) or x.shape!=(len(ids),48)
            or y.shape!=(len(ids),6) or len(ids)<300
            or not np.isfinite(g).all() or not np.isfinite(x).all()
            or not np.isfinite(y).all()):
        raise ValueError("invalid or incomplete V9 hot-state dataset")
    alias={"V8X0001":"V8T0001","V8X0002":"V8T0004",
           "V8X0003":"V8T0007","V8X0004":"V8T0010"}
    groups=np.asarray([alias.get(str(n),str(n)) for n in ids])
    names=np.unique(groups)
    fixed={"V8T0001","V8T0004","V8T0007","V8T0010"}&set(names)
    candidates=np.asarray(sorted(set(names)-fixed))
    if len(candidates)<5:
        raise ValueError("geometry-disjoint V9 validation requires at least five non-aliased geometries")
    nval=max(1,int(round(.2*len(candidates))))
    val_geoms=[str(n) for n in sorted(rng.choice(candidates,size=nval,replace=False).tolist())]
    train_geoms=[str(n) for n in sorted(set(names)-set(val_geoms))]
    tr=np.flatnonzero(np.isin(groups,train_geoms))
    va=np.flatnonzero(np.isin(groups,val_geoms))
    if len(va)<20 or len(tr)<200:
        raise ValueError("geometry-disjoint split is too small")
    base=restore_rank48(a.skin_tangent,a.skin_base)
    with torch.no_grad():
        bp=[]
        for lo in range(0,len(ids),128):
            bp.append(base(torch.as_tensor(g[lo:lo+128]),
                           torch.as_tensor(x[lo:lo+128])).cpu().numpy())
    bp=np.concatenate(bp)
    res=y-bp
    gm=g[tr].mean(0)
    gs=np.maximum(g[tr].std(0),1e-4)
    ass=np.maximum(x[tr].std(0),1e-2)
    ys=np.maximum(y[tr].std(0),1e-7)
    net=HotResidual(gm,gs,ass,ys,a.hidden)
    opt=torch.optim.Adam(net.parameters(),lr=1e-3,weight_decay=1e-4)
    gt=torch.as_tensor(g);xt=torch.as_tensor(x)
    rt=torch.as_tensor(res/ys)
    counts={n:int(np.sum(groups[tr]==n)) for n in train_geoms}
    wt=np.asarray([1./counts[groups[i]] for i in tr],np.float32)
    wt*=len(wt)/wt.sum()
    w=torch.as_tensor(wt)
    best=float("inf");saved=None;best_ep=0;wait=0;history=[]
    for ep in range(1,a.epochs+1):
        net.train()
        order=rng.permutation(len(tr))
        for lo in range(0,len(order),64):
            pos=order[lo:lo+64];ix=tr[pos]
            opt.zero_grad(set_to_none=True)
            pred=net(gt[ix],xt[ix])/torch.as_tensor(ys)
            each=((pred-rt[ix])**2).mean(dim=1)
            loss=(each*w[pos]).sum()/w[pos].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv=net(gt[va],xt[va]).cpu().numpy()
            each=np.mean(((pv-res[va])/ys)**2,axis=1)
            v=float(np.mean([each[ids[va]==n].mean() for n in val_geoms]))
        history.append({"epoch":ep,"val_macro_mse":v})
        if v<best-1e-7:
            best=v;best_ep=ep;wait=0
            saved={k:v.detach().clone() for k,v in net.state_dict().items()}
        else:
            wait+=1
        if ep==1 or ep%20==0:
            print("V9_TRAIN",ep,"val",round(v,7),"best",best_ep,flush=True)
        if wait>=a.patience:
            break
    net.load_state_dict(saved);net.eval()
    with torch.no_grad():
        corr=net(gt,xt).cpu().numpy()
    base_val=bp[va];new_val=base_val+corr[va]
    bm=metrics(base_val,y[va],groups[va])
    nm=metrics(new_val,y[va],groups[va])
    ck=dict(state=net.state_dict(),gm=net.gm.detach().cpu(),
        gs=net.gs.detach().cpu(),ass=net.ass.detach().cpu(),
        ys=net.ys.detach().cpu(),hidden=a.hidden,
        skin_tangent=str(a.skin_tangent),skin_base=str(a.skin_base),
        sample_count=len(ids),train_count=len(tr),validation_count=len(va),
        geometries=names.tolist(),train_geometries=train_geoms,
        validation_geometries=val_geoms,seed=a.seed)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck,a.out)
    result=dict(status="trained",samples=len(ids),geometries=len(names),
        train_count=len(tr),validation_count=len(va),train_geometries=train_geoms,
        validation_geometries=val_geoms,best_epoch=best_ep,
        best_val_macro_mse=best,baseline_metrics=bm,
        corrected_metrics=nm,history=history,checkpoint=str(a.out),
        data_files=[str(v) for v in a.data])
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V9_TRAIN_RESULT",json.dumps({k:result[k] for k in
        ("samples","geometries","train_count","validation_count","train_geometries",
         "validation_geometries","best_epoch","baseline_metrics","corrected_metrics")}),flush=True)


if __name__=="__main__":
    main()
