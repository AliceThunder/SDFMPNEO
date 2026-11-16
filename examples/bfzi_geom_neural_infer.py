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
    p.add_argument("--tangent-model", type=Path, default=None)
    p.add_argument("--z-fix-model",type=Path,default=None)
    p.add_argument("--lift-model", type=Path, default=None,
                   help="optional 48-mode sea-skin reconstruction")
    p.add_argument("--pkg-model", type=Path, default=None)
    p.add_argument("--lift-feedback", action="store_true",
                   help="incorporate sea-skin lift in coupled heat balance")
    p.add_argument("--lift-gain", type=float, default=1.)
    p.add_argument("--skin-model", type=Path, default=None)
    p.add_argument("--skin-gain", type=float, default=1.)
    p.add_argument("--skin-output-model", type=Path, default=None)
    p.add_argument("--bulk-output-model", type=Path, default=None)
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
    tangent=None
    if a.tangent_model:
        from examples.bfzi_z_tangent import restore
        tangent=restore(a.tangent_model)
    if a.z_fix_model:
        from sdfmpneo_bfzi.geom_ztune import restore
        tangent=restore(a.z_fix_model)
    pkg=None
    if a.pkg_model:
        from sdfmpneo_bfzi.geom_pkg import restore
        pkg=restore(a.pkg_model)
    lift=None
    if a.lift_feedback:
        if a.lift_model is None:
            p.error("--lift-feedback requires --lift-model")
        with np.load(a.lift_model) as data:
            lift=data["w"]*a.lift_gain
    skin=None
    if a.skin_model:
        from sdfmpneo_bfzi.geom_skin import restore
        skin=restore(a.skin_model)
    pred = rollout(rom, net, ck, v, t, tangent=tangent, pkg=pkg,
                   lift=lift, skin=skin, skin_gain=a.skin_gain)
    if a.skin_output_model:
        from sdfmpneo_bfzi.geom_skin import reconstruct,restore
        fix=restore(a.skin_output_model)
        pred["temp_skin_K"]=reconstruct(rom,pred["temp_K"],
            pred["heat_state"],t,v,fix)
    if a.bulk_output_model:
        if not a.skin_output_model:
            p.error("--bulk-output-model requires --skin-output-model")
        from sdfmpneo_bfzi.geom_skin import reconstruct,restore
        fix=restore(a.bulk_output_model)
        pred["temp_bulk_K"]=reconstruct(rom,pred["temp_skin_K"],
            pred["heat_state"],t,v,fix,rank=64)
    if a.lift_model and not a.lift_feedback:
        from sdfmpneo_bfzi.geom_lift import reconstruct
        with np.load(a.lift_model) as lift:
            pred["temp_lift_K"] = reconstruct(head,pred["temp_K"],
                                               pred["heat_state"],lift["w"])
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
                   lift_model=str(a.lift_model) if a.lift_model else None,
                   z_fix_model=str(a.z_fix_model) if a.z_fix_model else None,
                   pkg_model=str(a.pkg_model) if a.pkg_model else None,
                   lift_feedback=a.lift_feedback,
                   lift_gain=a.lift_gain,
                   skin_model=str(a.skin_model) if a.skin_model else None,
                   skin_gain=a.skin_gain,
                   skin_output_model=str(a.skin_output_model)
                       if a.skin_output_model else None,
                   bulk_output_model=str(a.bulk_output_model)
                       if a.bulk_output_model else None,
                   neural_heat=True, em_directional_solves_online=0,
                   heat_implicit_solves_online=0)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
