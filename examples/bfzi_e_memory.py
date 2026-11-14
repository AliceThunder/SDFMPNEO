"""Train source-normalized SOE/exp residuals on G1/G2 donor geometries only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch

from examples.bfzi_frac_rom_eval import train
from sdfmpneo_bfzi.frac_rom import FracROM
from sdfmpneo_bfzi.heat_rom import HeatROM

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("runs"))
    p.add_argument("--epochs",type=int,default=650)
    a=p.parse_args()
    torch.set_num_threads(1)
    out={}
    for key in ("g1","g2"):
        rom=HeatROM.load(a.runs/f"bfzi_d_{key}_rom.npz")
        fr=FracROM.load(a.runs/f"bfzi_d_{key}_tangent.npz")
        with np.load(a.runs/f"bfzi_a2_{key}_extra.npz") as d:
            t=d["time_s"].copy();v=d["voltage_v"].copy()
            temp=d["temp_K"].copy();z=d["z6"].copy()
        state=(temp-rom.amb)@(rom.cap[:,None]*rom.basis)
        np.savez_compressed(a.runs/f"bfzi_frac_rom_{key}.npz",
                            time_s=t,voltage_v=v,
                            thermal_state=state,
                            delta_z6=z-z[0],z0=z[0])
        out[key]=train(a.runs,key.upper(),fr,rom,a.epochs,13)
    (a.runs/"bfzi_e_memory.json").write_text(json.dumps(out,indent=2))
    print("SAVED",a.runs/"bfzi_e_memory.json",flush=True)

if __name__=="__main__":
    main()
