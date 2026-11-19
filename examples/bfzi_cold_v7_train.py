"""Train the unified V7 cold-port network with a circuit-current objective."""
from __future__ import annotations
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from sdfmpneo_bfzi.geom_coldnet import ColdNet,feat,target

def read(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def mat(z):
    a,b,c,d,e,f=z
    return np.array([[a+1j*b,c+1j*d],[c+1j*d,e+1j*f]])
def zmat(y):
    a=torch.complex(y[:,0],y[:,1]+y[:,6])
    b=torch.complex(y[:,2],y[:,3]+y[:,7])
    c=torch.complex(y[:,4],y[:,5]+y[:,8])
    return torch.stack((torch.stack((a,b),-1),
                        torch.stack((b,c),-1)),-2)
def circuits(data):
    out={}
    for row in data.get("train",[])+data.get("dev",[]):
        if "ext_re" not in row or "ext_im" not in row:continue
        ext=np.asarray(row["ext_re"],float)+1j*np.asarray(row["ext_im"],float)
        name=row.get("donor",json.dumps([ext.real.tolist(),ext.imag.tolist()]))
        out[name]=ext
    if not out:raise ValueError("no external circuit families in legacy cold data")
    return np.stack([out[k] for k in sorted(out)])
def evaluate(net,rows,exts):
    if not rows:return None
    ze=[];ae=[];ie=[];per={}
    for row in rows:
        z,air=net.predict(row["geometry"])
        zr=np.asarray(row["z0"],float)
        ar=np.asarray(row["air"],float)
        if ar.shape==(2,2):ar=ar[[0,0,1],[0,1,1]]
        ze.append(float(np.linalg.norm(z-zr)/max(np.linalg.norm(zr),1e-12)))
        ae.append(float(np.linalg.norm(air-ar)/max(np.linalg.norm(ar),1e-12)))
        for ext in exts:
            ref=np.linalg.solve(mat(zr)+ext,[1.,0.])
            pred=np.linalg.solve(mat(z)+ext,[1.,0.])
            ie.append(float(np.linalg.norm(pred-ref)/max(np.linalg.norm(ref),1e-12)))
        per[row["name"]]=100*ze[-1]
    return dict(geometries=len(rows),cold_Z_pct=100*float(np.mean(ze)),
                air_L_pct=100*float(np.mean(ae)),
                current_pct=100*float(np.mean(ie)),
                per_geometry=per)
def by_stratum(net,rows,exts):
    return {s:evaluate(net,[r for r in rows if r.get("stratum")==s],exts)
            for s in ("core","boundary","extreme")}
def train(data,cir,*,out,epochs=600,patience=100,lr=6e-4,
          batch=64,current_weight=.5,seed=72027):
    torch.manual_seed(seed);torch.set_num_threads(2)
    tr=list(data["train"]);dev=list(data["val"]);test=list(data["test"])
    ids=[{json.dumps(r["geometry"],sort_keys=True) for r in x}
         for x in (tr,dev,test)]
    if ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:
        raise ValueError("V7 train/val/test geometry overlap")
    ext=circuits(cir)
    x=np.stack([feat(r["geometry"]) for r in tr])
    y=np.stack([target(r) for r in tr])
    xm=x.mean(0)
    xs=np.maximum(x.std(0),np.array([.02,.02,1,1,2,2,2,2,2,2,2,2,
                                    .01,.1,.01,.1],dtype=np.float32))
    ym=y.mean(0)
    ys=np.maximum(y.std(0),np.array([.01,.003,.003,.003,.01,.003,
                                    .25,.025,.25],dtype=np.float32))
    net=ColdNet(xm,xs,ym,ys)
    xt=torch.as_tensor(x,dtype=torch.float32)
    yt=torch.as_tensor(y,dtype=torch.float32)
    ys_t=net.ys
    ext_t=torch.as_tensor(ext,dtype=torch.complex64)
    ir=[]
    for row in tr:
        ir.append([np.linalg.solve(mat(row["z0"])+e,[1.,0.]) for e in ext])
    ir_t=torch.as_tensor(np.asarray(ir),dtype=torch.complex64)
    opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4)
    best=float("inf");best_ep=0;best_state=copy.deepcopy(net.state_dict())
    history=[];rng=np.random.default_rng(seed)
    for ep in range(1,epochs+1):
        net.train();order=rng.permutation(len(tr))
        for start in range(0,len(tr),batch):
            ix=order[start:start+batch]
            pred=net(xt[ix])
            loss_z=(((pred-yt[ix])/ys_t)**2).mean()
            zm=zmat(pred)
            rhs=torch.zeros((len(ix),len(ext),2),dtype=torch.complex64)
            rhs[...,0]=1.
            ip=torch.linalg.solve(zm[:,None]+ext_t[None],rhs)
            ref=ir_t[ix]
            loss_i=((ip-ref).abs().square().sum(-1)/
                    ref.abs().square().sum(-1).clamp_min(1e-12)).mean()
            loss=loss_z+current_weight*loss_i
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if ep%5:continue
        net.eval();d=evaluate(net,dev,ext)
        score=d["cold_Z_pct"]+.1*d["current_pct"]+.05*d["air_L_pct"]
        history.append(dict(epoch=ep,loss=float(loss.detach()),
                            val_Z_pct=d["cold_Z_pct"],
                            val_current_pct=d["current_pct"],
                            val_air_pct=d["air_L_pct"]))
        if score<best:
            best=score;best_ep=ep;best_state=copy.deepcopy(net.state_dict())
        if ep%25==0:
            print("V7_COLD_TRAIN",ep,"Z",round(d["cold_Z_pct"],4),
                  "I",round(d["current_pct"],4),"best",best_ep,flush=True)
        if ep-best_ep>=patience:break
    net.load_state_dict(best_state);net.eval()
    Path(out).parent.mkdir(parents=True,exist_ok=True);net.save(out)
    return dict(status="candidate" if best_ep else "baseline",
        checkpoint=str(out),best_epoch=best_ep,epochs_trained=ep,
        train_geometries=len(tr),val_geometries=len(dev),
        test_geometries=len(test),train_strata=by_stratum(net,tr,ext),
        val=evaluate(net,dev,ext),val_strata=by_stratum(net,dev,ext),
        test=evaluate(net,test,ext),test_strata=by_stratum(net,test,ext),
        current_loss_weight=current_weight,external_network_families=len(ext),
        history=history)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("runs/geomfrac_cold_v7/data_merged.json"))
    p.add_argument("--circuits",type=Path,default=Path("runs/geomfrac_cold_ports_v1.json"))
    p.add_argument("--out",type=Path,default=Path("runs/geomfrac_cold_v7.pt"))
    p.add_argument("--report",type=Path,default=Path("runs/geomfrac_cold_v7.json"))
    p.add_argument("--epochs",type=int,default=600)
    p.add_argument("--batch",type=int,default=64)
    p.add_argument("--current-weight",type=float,default=.5)
    a=p.parse_args()
    res=train(read(a.data),read(a.circuits),out=a.out,epochs=a.epochs,
              batch=a.batch,current_weight=a.current_weight)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(res,indent=2)+"\n",encoding="utf-8")
    print("V7_COLD_RESULT",json.dumps({k:res[k] for k in
          ("best_epoch","train_geometries","val_geometries","test_geometries",
           "val","test")}),flush=True)
if __name__=="__main__":main()
