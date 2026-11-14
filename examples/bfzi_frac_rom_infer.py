"""Independent BFZI-free inference from a saved 32-state ROM and memory net."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from sdfmpneo_bfzi.frac_memory import FracNet
from sdfmpneo_bfzi.frac_rom import FracROM
from sdfmpneo_bfzi.heat_rom import HeatROM

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--pose",choices=("P03","P07"),default="P03")
    p.add_argument("--mode",choices=("frac","exp","physics"),default="frac")
    p.add_argument("--voltage",type=float,default=75.)
    p.add_argument("--dt",type=float,default=15.)
    p.add_argument("--duration",type=float,default=600.)
    p.add_argument("--off",action="append",default=[],
                   help="off interval START:END in seconds; repeatable")
    p.add_argument("--history",type=Path,help="CSV with time_s,voltage_v")
    p.add_argument("--out",type=Path,default=Path("runs/bfzi_frac_infer.npz"))
    a=p.parse_args()
    key=a.pose.lower()
    if a.history is None:
        if a.dt<=0 or a.duration<=0:
            p.error("positive duration and timestep required")
        t=np.arange(0.,a.duration+.1*a.dt,a.dt)
        v=np.full_like(t,a.voltage)
        for item in a.off:
            x,y=map(float,item.split(":"))
            if not 0<=x<y<=a.duration:
                p.error("off intervals must lie within the time history")
            v[(t>=x)&(t<y)]=0.
    else:
        x=np.loadtxt(a.history,delimiter=",",ndmin=2)
        if x.shape[1]!=2:
            p.error("history CSV requires exactly two columns")
        t,v=x[:,0],x[:,1]
    fr=FracROM.load(a.runs/f"bfzi_frac_tangent_{key}.npz")
    rom=HeatROM.load(a.runs/f"bfzi_heat_rom_{key}.npz")
    net=ck=None
    if a.mode!="physics":
        ck=torch.load(a.runs/f"bfzi_frac_rom_{key}_{a.mode}.pt",
                      map_location="cpu",weights_only=False)
        net=FracNet(nout=6,mode=a.mode,hidden=ck["hidden"])
        net.load_state_dict(ck["state"])
        net.eval()
    torch.set_num_threads(1)
    out=fr.rollout(rom,t,v,net,ck)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,**out)
    print("SAVED",a.out,"steps",len(t),"max_total_budget_W",
          float(max(out["total_budget_W"])),flush=True)

if __name__=="__main__":
    main()
