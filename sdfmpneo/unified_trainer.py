"""Physical-residual training for the unified neural Maxwell initial guess."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import numpy as np
from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp


@dataclass(frozen=True)
class MaxwellTrainingConfig:
    epochs:int=1000
    batch_size:int=128
    learning_rate:float=1e-3
    weight_decay:float=1e-6
    patience:int=150
    validation_interval:int=5
    seed:int=17
    dtype:str="float32"
    def __post_init__(self):
        if min(self.epochs,self.batch_size,self.patience,self.validation_interval)<1: raise ValueError("training counts must be positive")
        if self.learning_rate<=0 or self.weight_decay<0: raise ValueError("invalid optimizer settings")
        if self.dtype not in {"float32","float64"}: raise ValueError("dtype must be float32 or float64")
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MaxwellTrainingReport:
    epochs_completed:int
    best_epoch:int
    best_validation_residual:float
    train_residual:float
    validation_residual:float
    test_residual:float
    stopped_early:bool
    device:str
    training_config:dict
    network_config:dict


def residual_loss(torch,y,base,scale,Q,S,N,p,r):
    c=base+scale.unsqueeze(-1)*y.reshape((-1,p,2*r))
    quad=torch.einsum("bpi,bij,bpj->bp",c,Q,c)
    linear=torch.einsum("bpi,bpi->bp",c,S)
    residual=(N-2.0*linear+quad).clamp_min(0.0)
    return torch.mean(residual/N.clamp_min(torch.finfo(c.dtype).tiny))


def train_maxwell_accelerator(dataset,*,network_settings=None,training_settings=None,device="cuda",monitor=None,checkpoint_path=None):
    try: import torch
    except ImportError as exc: raise ImportError("install sdfmpneo[neural] to train the unified model") from exc
    cfg=MaxwellTrainingConfig(**(training_settings or {})); r,p=dataset.reduced_rank,dataset.n_rhs
    train_ids=dataset.indices("train"); val_ids=dataset.indices("validation"); test_ids=dataset.indices("test")
    normalizer=FeatureNormalizer.fit(dataset.features[train_ids]); ns=dict(network_settings or {})
    netcfg=ResidualMLPConfig(input_dimension=dataset.features.shape[1],output_dimension=2*r*p,**ns)
    model=build_residual_mlp(netcfg,normalizer)
    actual=str(device)
    if actual.startswith("cuda") and not torch.cuda.is_available(): print("CUDA 不可用，自动退回 CPU。",flush=True); actual="cpu"
    dtype=torch.float32 if cfg.dtype=="float32" else torch.float64; model=model.to(device=actual,dtype=dtype)
    try: opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay,fused=actual.startswith("cuda"))
    except (TypeError,RuntimeError): opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    tensors=[torch.as_tensor(x,dtype=dtype,device=actual) for x in (dataset.features,dataset.baseline,dataset.coefficient_scale,dataset.residual_gram,dataset.residual_linear,dataset.rhs_norm2)]
    F,B0,CS,Q,S,N=tensors; checkpoint=None if checkpoint_path is None else Path(checkpoint_path); start=0; best=float("inf"); best_epoch=0; best_state=copy.deepcopy(model.state_dict())
    if checkpoint is not None and checkpoint.is_file():
        try:
            try: q=torch.load(checkpoint,map_location=actual,weights_only=False)
            except TypeError: q=torch.load(checkpoint,map_location=actual)
            if q.get("network_config")!=netcfg.to_dict(): raise ValueError("network config changed")
            model.load_state_dict(q["network"],strict=True); opt.load_state_dict(q["optimizer"]); start=int(q["epoch"]); best=float(q["best"]); best_epoch=int(q["best_epoch"]); best_state=q["best_state"]
            print(f"恢复 Maxwell residual 训练：epoch={start}",flush=True)
        except Exception as exc:
            print(f"训练检查点与当前统一模型不一致，重新训练网络：{exc}",flush=True); checkpoint.unlink(missing_ok=True)
    rng=np.random.default_rng(cfg.seed)
    def evaluate(ids):
        model.eval(); total=0.; count=0; batch=max(cfg.batch_size,256)
        with torch.no_grad():
            for st in range(0,len(ids),batch):
                index=torch.as_tensor(ids[st:st+batch],dtype=torch.long,device=actual); loss=residual_loss(torch,model(F[index]),B0[index],CS[index],Q[index],S[index],N[index],p,r); total+=float(loss.cpu())*len(index); count+=len(index)
        return total/max(count,1)
    last_train=last_val=float("inf"); stale=0; completed=start
    for epoch in range(start,cfg.epochs):
        if monitor is not None: monitor.checkpoint()
        model.train(); order=rng.permutation(train_ids); total=0.; count=0
        for st in range(0,len(order),cfg.batch_size):
            index=torch.as_tensor(order[st:st+cfg.batch_size],dtype=torch.long,device=actual); opt.zero_grad(set_to_none=True); loss=residual_loss(torch,model(F[index]),B0[index],CS[index],Q[index],S[index],N[index],p,r); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); opt.step(); total+=float(loss.detach().cpu())*len(index); count+=len(index)
        last_train=total/max(count,1); completed=epoch+1
        validate=completed==1 or completed%cfg.validation_interval==0 or completed==cfg.epochs
        if validate:
            last_val=evaluate(val_ids)
            threshold=best-1e-10*max(1.0,abs(best)) if np.isfinite(best) else float("inf")
            if not np.isfinite(best) or last_val<threshold: best=last_val; best_epoch=completed; best_state=copy.deepcopy(model.state_dict()); stale=0
            else: stale+=cfg.validation_interval
        print(f"训练神经网络……{100*completed/cfg.epochs:5.1f}%  epoch={completed}/{cfg.epochs}  train={last_train:.5g}  val={last_val:.5g}",flush=True)
        if monitor is not None:
            with monitor._lock: monitor.data.update(phase="neural_training",epoch=completed,epoch_total=cfg.epochs,train_loss=last_train,validation_loss=None if not np.isfinite(last_val) else last_val)
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True,exist_ok=True); temporary=checkpoint.with_suffix(checkpoint.suffix+".tmp"); torch.save({"network_config":netcfg.to_dict(),"epoch":completed,"network":model.state_dict(),"optimizer":opt.state_dict(),"best":best,"best_epoch":best_epoch,"best_state":best_state},temporary); temporary.replace(checkpoint)
        if stale>=cfg.patience: break
    model.load_state_dict(best_state); model.eval(); test=evaluate(test_ids)
    return model,MaxwellTrainingReport(completed,best_epoch,float(best),float(last_train),float(last_val),float(test),completed<cfg.epochs,actual,cfg.to_dict(),netcfg.to_dict())

__all__=["MaxwellTrainingConfig","MaxwellTrainingReport","residual_loss","train_maxwell_accelerator"]
