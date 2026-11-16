"""Standalone GeomFracET trial: no online 8-mode tangent or implicit heat ROM."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.geom_rom import GeomHeatROM
from sdfmpneo_bfzi.geom_canon import make_rom
from sdfmpneo_bfzi.geom_neural import GeomFracET, rollout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--rom-file", type=Path, default=None)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/geomfrac_et_infer.npz"))
    p.add_argument("--voltage", type=float, default=100.)
    p.add_argument("--heat", type=float, default=240.)
    p.add_argument("--cool", type=float, default=180.)
    p.add_argument("--step", type=float, default=15.)
    p.add_argument("--horizon", type=float, default=600.)
    a = p.parse_args()
    if (a.step <= 0 or a.horizon <= 0 or a.heat < 0 or a.cool < 0
            or a.horizon/a.step != round(a.horizon/a.step)):
        p.error("expected positive aligned horizon/step and nonnegative heat/cool")
    head = GeomHeat.load(a.anchor)
    ck = torch.load(a.model, map_location="cpu", weights_only=True)
    if ck.get('canonical',False):
        rom = make_rom(head,rank=int(ck['rank']))
    else:
        if a.rom_file is None:
            p.error('--rom-file is required for the original POD model')
        rom = GeomHeatROM.load(head, a.rom_file)
    net = GeomFracET(ng=int(ck["ng"]), rank=int(ck["rank"]),
                     soe_quad=ck.get("soe_quad","legacy"),
                     heat_dt=float(ck.get("heat_dt",0.)),heat_z=bool(ck.get("heat_z",False)))
    net.load_state_dict(ck["state"])
    net.eval()
    t = np.linspace(0., a.horizon, int(round(a.horizon/a.step))+1)
    v = np.where((t < a.heat) | (t >= a.heat+a.cool), a.voltage, 0.)
    tic = time.perf_counter()
    pred = rollout(rom, net, ck, v, t)
    online_s = time.perf_counter()-tic
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, time_s=t, voltage_v=v,
                        **{k: v for k, v in pred.items() if isinstance(v, np.ndarray)})
    summary = dict(output=str(a.out), nsteps=len(t), rank=rom.rank,
                   online_seconds=float(online_s),
                   max_copper_K=float(np.max(pred["temp_K"][:, rom.cu])),
                   max_material_budget_W=float(np.max(abs(pred["material_budget_W"]))),
                   max_fv_residual_W=float(np.max(pred["fv_node_residual_W"])),
                   max_dynamic_Z_ohm=float(np.max(abs(pred["z6"]-pred["z6"][0]))),
                   neural_heat=True, em_directional_solves_online=0,
                   heat_implicit_solves_online=0)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
