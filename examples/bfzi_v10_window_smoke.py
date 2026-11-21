"""One actual BFZI history smoke test of differentiable SOE short windows."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from examples.bfzi_geom_multi import prepare
from examples.bfzi_v10_window_train import load_heat
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_init import GeomInit,thermal_mix
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix
from sdfmpneo_bfzi.geom_window import HeatWindow,pack


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def main():
    torch.set_num_threads(2)
    cfg=read("runs/geomfrac_free_rank48_v8_selection.json")
    plan=read("runs/geomfrac_v8_full_plan.json")
    mf=read("runs/geomfrac_s5_v8_full_manifest.json")
    spec=mf["train"][0]
    name=Path(spec["anchor"].replace("\\","/")).stem.split("_")[2]
    case=plan["cases"][name]
    atlas=GeomInit.load(cfg["atlas"])
    cold=ColdNet.load("runs/geomfrac_cold_v8_frozen.pt")
    src=SrcFix.load(cfg["source_model"])
    init=thermal_mix(atlas,src.apply(case,cold.apply(
        case,atlas.predict(case))),.25)
    net,ck,ec=load_heat("runs/geomfrac_et_v8_heat_tune.pt")
    tan=restore_rank48("runs/geomfrac_rank48_skin_tan_v3.pt",
                       "runs/geomfrac_et_v9_ztune.pt")
    par=ParZ48.load("runs/geomfrac_v8_par48.pt")
    for p in par.parameters():
        p.requires_grad_(False)
    histories=prepare(spec,48)
    selected=[(row,i) for row in histories
              for i in range(2,len(row["time"])-1)
              if abs(float(row["voltage"][i]))>1]
    if not selected:
        raise ValueError("smoke test requires nonzero physical excitation")
    row,start=selected[0]
    win=HeatWindow(case,init,net,ec,tan,par,float(cfg["shape_mix"]))
    h=win.warm(row,start)
    old=torch.as_tensor(row["a"][start],dtype=torch.float32)*win.scale
    win.anchor(old)
    z0,_,_,_=win.circuit(old,float(row["voltage"][start]),float(row["time"][start]))
    dt=float(row["time"][start+1]-row["time"][start])
    h,pred=win.step(h,old,dt,float(row["voltage"][start]),
                    float(row["time"][start+1]))
    z,I,ps,pc=win.circuit(pred,float(row["voltage"][start+1]),
                           float(row["time"][start+1]))
    temp=((pred/win.scale-torch.as_tensor(row["a"][start+1],
                                          dtype=torch.float32))/win.sa).square().mean()
    dz=pack(z)-pack(z0)
    truth=torch.as_tensor(row["z6"][start+1]-row["z6"][start],dtype=dz.dtype)
    loss=temp+((dz-truth)**2).mean()
    loss.backward()
    grad=net.heat_head[2].weight.grad
    if (grad is None or not bool(torch.isfinite(grad).all())
            or float(grad.abs().sum())<=0 or not bool(torch.isfinite(loss))):
        raise AssertionError("differentiable SOE heat head gradients are invalid")
    print("V10_WINDOW_SMOKE",json.dumps(dict(name=name,
        dt=dt,drive_v=float(row["voltage"][start]),
        loss=float(loss.detach()),grad_norm=float(grad.norm()),
        source_w=float(ps.detach()),copper_w=pc.detach().tolist(),
        current_A=[float(I[0].real.detach()),
                   float(I[1].real.detach())])),flush=True)


if __name__=="__main__":
    main()
