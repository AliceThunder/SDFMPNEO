"""Train a geometry-shared hot-state sea-impedance residual from BFZI labels."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sdfmpneo_bfzi.geom_hotz import HotResidual
from sdfmpneo_bfzi.geom_skintan import restore_rank48


def rel(pred,true):
    return float(np.linalg.norm(pred-true)/max(np.linalg.norm(true),1e-12))


def report(pred,true,ids):
    out={"all":rel(pred,true),"channels":{},"geometry":{}}
    names=("R11","X11","R12","X12","R22","X22")
    for j,n in enumerate(names):
        out["channels"][n]=rel(pred[:,j],true[:,j])
    for name in np.unique(ids):
        m=ids==name
        out["geometry"][str(name)]={"count":int(m.sum()),
            "baseline":rel(pred[m],true[m])}
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,default=Path("runs/geomfrac_v9_hotstates.npz"))
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
    with np.load(a.data) as d:
        g=np.asarray(d["g"],np.float32)
        x=np.asarray(d["a"],np.float32)
        y=np.asarray(d["sea"],np.float32)
        ids=np.asarray(d["name"]).astype(str)
    if (g.shape!=(len(ids),45) or x.shape!=(len(ids),48)
            or y.shape!=(len(ids),6) or len(ids)<100
            or not np.isfinite(g).all() or not np.isfinite(x).all()
            or not np.isfinite(y).all()):
        raise ValueError("invalid or incomplete V9 hot-state dataset")
    tr=[];va=[]
    for name in np.unique(ids):
        ix=np.flatnonzero(ids==name)
        ix=rng.permutation(ix)
        n=max(1,int(round(.2*len(ix))))
        va.extend(ix[:n]);tr.extend(ix[n:])
    tr=np.asarray(sorted(tr));va=np.asarray(sorted(va))
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
    xt=torch.as_tensor(x);gt=torch.as_tensor(g)
    rt=torch.as_tensor(res/ys)
    best=float("inf");saved=None;best_ep=0;wait=0;history=[]
    for ep in range(1,a.epochs+1):
        net.train()
        order=rng.permutation(tr)
        for lo in range(0,len(order),64):
            ix=order[lo:lo+64]
            opt.zero_grad(set_to_none=True)
            pred=net(gt[ix],xt[ix])/torch.as_tensor(ys)
            loss=((pred-rt[ix])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv=net(gt[va],xt[va]).cpu().numpy()
            v=float(np.mean(((pv-res[va])/ys)**2))
        history.append({"epoch":ep,"val_mse":v})
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
    metrics={"baseline":report(base_val,y[va],ids[va]),
             "corrected":report(new_val,y[va],ids[va])}
    for name in np.unique(ids[va]):
        m=ids[va]==name
        metrics["baseline"]["geometry"][str(name)]["corrected"]=rel(new_val[m],y[va][m])
    ck=dict(state=net.state_dict(),gm=net.gm.detach().cpu(),\n        gs=net.gs.detach().cpu(),ass=net.ass.detach().cpu(),\n        ys=net.ys.detach().cpu(),
        hidden=a.hidden,skin_tangent=str(a.skin_tangent),skin_base=str(a.skin_base),
        sample_count=len(ids),train_count=len(tr),validation_count=len(va),
        geometries=np.unique(ids).tolist(),seed=a.seed)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck,a.out)
    result=dict(status="trained",samples=len(ids),geometries=len(np.unique(ids)),
        train_count=len(tr),validation_count=len(va),best_epoch=best_ep,
        best_val_mse=best,baseline_metrics=metrics["baseline"],
        corrected_metrics=metrics["corrected"],history=history,
        checkpoint=str(a.out))
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V9_TRAIN_RESULT",json.dumps({k:result[k] for k in
        ("samples","geometries","train_count","validation_count","best_epoch",
         "baseline_metrics","corrected_metrics")}),flush=True)


if __name__=="__main__":
    main()
