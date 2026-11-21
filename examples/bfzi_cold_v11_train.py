"""V11 balanced BFZI ColdNet: port currents, self impedances and Joule power.

Only the pretrained ColdNet parameters are changed. Select on the original
cold validation geometries; independent full-physics acceptances stay sealed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_cold_v7_train import circuits, evaluate, mat, read, zmat
from sdfmpneo_bfzi.geom_coldnet import ColdNet, copper_target, feat, target


def pwr(y, curr):
    """Cold sea and TX/RX copper power for unit TX port voltage."""
    zz=zmat(y)
    cu=y[:, [9, 11]]
    sea=zz.real-torch.diag_embed(cu)
    ps=.5*torch.einsum("bci,bij,bcj->bc",
        curr.conj(),sea.to(torch.complex64),curr).real
    pc=.5*cu[:,None,:]*curr.abs().square()
    return torch.cat((ps[...,None],pc),dim=-1)


def port_err(pred, true):
    """TX and RX current errors with a small coupling-independent floor."""
    floor=(.05*true[..., :1].abs()).square()
    return (pred-true).abs().square()/(true.abs().square()+floor).clamp_min(1e-12)


def err_tail(err):
    flat=err.reshape(-1)
    return flat.topk(max(1,(len(flat)+3)//4)).values.mean()


def loss_terms(net, x, y, ext, ir, scale, mask):
    pr=net(x)
    zm=zmat(pr)
    rhs=torch.zeros((len(x),len(ext),2),dtype=torch.complex64)
    rhs[...,0]=1
    ip=torch.linalg.solve(zm[:,None]+ext[None],rhs)
    zloss=(((pr[:,:9]-y[:,:9])/net.ys[:9])**2).mean()
    culoss=(((pr[mask,9:]-y[mask,9:])/net.ys[9:])**2).mean() if mask.any() else zloss.new_zeros(())
    ce=port_err(ip,ir)
    curr=.5*ce.mean()+.25*ce[...,1].mean()+.25*err_tail(ce[...,1])
    ys=torch.stack((pr[:,0],pr[:,1]+pr[:,6],
                    pr[:,4],pr[:,5]+pr[:,8]),-1)
    yt=torch.stack((y[:,0],y[:,1]+y[:,6],
                    y[:,4],y[:,5]+y[:,8]),-1)
    er=((ys-yt)/scale).square()
    selfz=er.mean()+.25*er[:,2].mean()
    if mask.any():
        pp=pwr(pr[mask],ip[mask])
        pt=pwr(y[mask],ir[mask])
        floor=.03*pt.abs().amax((1,2),keepdim=True).clamp_min(1e-10)
        perr=((pp-pt)/(pt.abs()+floor)).square()
        powloss=perr.mean()+.25*err_tail(perr[...,0])
    else:
        powloss=zloss.new_zeros(())
    zr=zmat(y)
    grad=torch.linalg.solve(zr[:,None]+ext[None],
                            ((zm-zr)[:,None]@ir[...,None]))[...,0]
    sens=(grad.abs().square().sum(-1)/
          ir.abs().square().sum(-1).clamp_min(1e-12))
    return dict(z=zloss,cu=culoss,curr=curr,selfz=selfz,
                power=powloss,sens=.5*(sens.mean()+err_tail(sens)))


def measure(net,rows,ext):
    """Evaluation measures include RX self and physical port power."""
    base=evaluate(net,rows,ext)
    base.pop("per_geometry",None)
    rr={"current_tx_pct":[],"current_rx_pct":[],
        "R11_cold_pct":[],"R22_cold_pct":[],
        "X11_cold_pct":[],"X22_cold_pct":[],
        "cold_copper_power_pct":[]}
    for row in rows:
        z,air,cu=net.predict_parts(row["geometry"])
        zr=np.asarray(row["z0"],float)
        for idx,name in ((0,"R11"),(4,"R22"),(1,"X11"),(5,"X22")):
            rr[name+"_cold_pct"].append(
                abs(z[idx]-zr[idx])/max(abs(zr[idx]),1e-8))
        rc=copper_target(row)
        if rc is None or cu is None:
            continue
        rcu=rc.reshape(2,2)
        rcu=rcu[:,0]+1j*rcu[:,1]
        for e in ext:
            ip=np.linalg.solve(mat(z)+e,np.array([1.,0],complex))
            ir=np.linalg.solve(mat(zr)+e,np.array([1.,0],complex))
            for ix,name in ((0,"current_tx_pct"),(1,"current_rx_pct")):
                rr[name].append(abs(ip[ix]-ir[ix])/max(abs(ir[ix]),1e-12))
            qpr=.5*cu.real*abs(ip)**2
            qref=.5*rcu.real*abs(ir)**2
            rr["cold_copper_power_pct"].append(
                np.mean(abs(qpr-qref)/np.maximum(abs(qref),1e-12)))
    for name,vals in rr.items():
        base[name]=100*float(np.mean(vals))
    return base


KEYS={"cold_Z_pct":.6,"current_pct":.2,"current_tx_pct":.1,
      "current_rx_pct":.2,"current_p90_pct":.15,
      "R22_cold_pct":.2,"cold_sea_power_pct":.2,
      "cold_copper_power_pct":.1}
GUARD={"cold_Z_pct":1.015,"current_pct":1.015,
       "current_tx_pct":1.015,"current_rx_pct":1.015,
       "current_p90_pct":1.02,"R22_cold_pct":1.015,
       "cold_sea_power_pct":1.015,"cold_copper_power_pct":1.02}


def select(d,base):
    score=sum(w*d[k]/max(base[k],1e-12) for k,w in KEYS.items())
    ok=all(d[k]<=lim*base[k] for k,lim in GUARD.items())
    return score,ok


def train(a):
    torch.manual_seed(34017)
    torch.set_num_threads(2)
    rng=np.random.default_rng(34017)
    data=read(a.data)
    ext=circuits(read(a.circuits))
    tr,va,te=(data[k] for k in ("train","val","test"))
    geo=[{json.dumps(r["geometry"],sort_keys=True) for r in rows}
         for rows in (tr,va,te)]
    if any(geo[i]&geo[j] for i,j in ((0,1),(0,2),(1,2))):
        raise ValueError("training, validation and test geometries must be disjoint")
    net=ColdNet.load(a.init)
    if len(net.ym)!=13:
        raise ValueError("V11 needs 13-output physical copper ColdNet")
    xt=torch.as_tensor(np.asarray([feat(r["geometry"]) for r in tr]),
                       dtype=torch.float32)
    copper=[copper_target(r) for r in tr]
    valid=np.asarray([v is not None for v in copper],bool)
    if valid.sum()<10:
        raise ValueError("V11 copper/power loss requires physical copper labels")
    yy=[np.r_[target(r),v if v is not None else np.zeros(4)]
        for r,v in zip(tr,copper)]
    yt=torch.as_tensor(np.asarray(yy),dtype=torch.float32)
    mask=torch.as_tensor(valid,dtype=torch.bool)
    ex=torch.as_tensor(ext,dtype=torch.complex64)
    rhs=torch.zeros((len(tr),len(ext),2),dtype=torch.complex64)
    rhs[...,0]=1.
    with torch.no_grad():
        ir=torch.linalg.solve(zmat(yt[:,:9])[:,None]+ex[None],rhs)
    zz=np.asarray([r["z0"] for r in tr])
    selfscale=torch.as_tensor(np.maximum(zz[:,[0,1,4,5]].std(0),
                                   [.01,.05,.01,.05]),dtype=torch.float32)
    net.eval()
    base=measure(net,va,ext)
    best_score,_=select(base,base)
    best_state=copy.deepcopy(net.state_dict())
    best_ep=0
    opt=torch.optim.AdamW(net.parameters(),lr=a.lr,weight_decay=1e-4)
    loss_w=dict(z=1.,cu=a.cu,curr=a.port,selfz=a.selfz,
                power=a.power,sens=a.sens)
    hist=[]
    for ep in range(1,a.epochs+1):
        net.train()
        order=rng.permutation(len(tr))
        for start in range(0,len(tr),a.batch):
            ix=order[start:start+a.batch]
            terms=loss_terms(net,xt[ix],yt[ix],ex,ir[ix],selfscale,mask[ix])
            loss=sum(v*loss_w[k] for k,v in terms.items())
            if not torch.isfinite(loss):
                raise RuntimeError("invalid balanced cold objective")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),2.)
            opt.step()
        if ep%5 and ep!=a.epochs:
            continue
        net.eval()
        val=measure(net,va,ext)
        score,ok=select(val,base)
        hist.append(dict(epoch=ep,valid=ok,score=float(score),
                         Z=val["cold_Z_pct"],R22=val["R22_cold_pct"],
                         I=val["current_pct"],Irx=val["current_rx_pct"],
                         Psea=val["cold_sea_power_pct"],
                         Pcu=val["cold_copper_power_pct"]))
        if ok and score < best_score-.002:
            best_score=score
            best_ep=ep
            best_state=copy.deepcopy(net.state_dict())
        print("V11_COLD",json.dumps(hist[-1]),flush=True)
    net.load_state_dict(best_state)
    net.eval()
    fin=measure(net,va,ext)
    test=measure(net,te,ext)
    if best_ep:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        net.save(a.out)
    result=dict(status="development_candidate" if best_ep else "not_promoted",
        model=str(a.out) if best_ep else None,init=str(a.init),
        best_epoch=best_ep,train_geometries=len(tr),val_geometries=len(va),
        test_geometries=len(te),copper_label_geometries=int(valid.sum()),
        weights=loss_w,base_val=base,
        val=fin,test=test,history=hist)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print("V11_RESULT",json.dumps({k:result[k] for k in
          ("status","best_epoch","base_val","val","test")}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,
        default=Path("runs/geomfrac_cold_v7/data_merged.json"))
    p.add_argument("--circuits",type=Path,
        default=Path("runs/geomfrac_cold_ports_v1.json"))
    p.add_argument("--init",type=Path,
        default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--out",type=Path,
        default=Path("runs/geomfrac_v11_cold.pt"))
    p.add_argument("--report",type=Path,
        default=Path("runs/geomfrac_v11_cold.json"))
    p.add_argument("--epochs",type=int,default=80)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--lr",type=float,default=4e-5)
    p.add_argument("--port",type=float,default=1.)
    p.add_argument("--selfz",type=float,default=.35)
    p.add_argument("--power",type=float,default=.5)
    p.add_argument("--sens",type=float,default=.15)
    p.add_argument("--cu",type=float,default=.2)
    a=p.parse_args()
    train(a)


if __name__=="__main__":
    main()
