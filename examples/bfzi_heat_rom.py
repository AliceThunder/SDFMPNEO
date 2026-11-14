"""Build and measure a 32-mode thermal ROM on real BFZI V150 trajectories.

This stage tests projection and heat-operator integration; neural EM rollout
and held-out drive prediction are separate B acceptance gates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_layer_study import GRIDS
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D
from sdfmpneo_bfzi.heat_rom import HeatROM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--poses", nargs="+", choices=("P03", "P07"),
                   default=("P03", "P07"))
    p.add_argument("--level", choices=tuple(GRIDS), default="extra")
    p.add_argument("--rank", type=int, default=32)
    a = p.parse_args()
    bf, v150 = originals(a.bfzi_root)
    old = json.loads((a.runs / "bfzi_obb_physical_study.json").read_text())
    report = {}
    for pose in a.poses:
        stem = f"bfzi_pose_{pose.lower()}"
        meta = json.loads((a.runs / f"{stem}_anchor.json").read_text())
        geo = build_args(meta)
        geo["shift_mm"] = np.asarray(
            old["geometries"][pose]["new_shift_mm"], float)
        prev = GeomHeat.load(a.runs / f"{stem}_disjoint_anchor.npz")
        xz, y = GRIDS[a.level]
        t0 = time.perf_counter()
        data, _ = create(bf, v150, "M4", **geo,
                         sea_void="oriented", physical_box_only=True,
                         grid_xz=xz, grid_y=y)
        em = BFZIV148Adapter(data, bf, heat_mode="port-weak",
                             fv_faces="exact", source_mask="subcell")
        et = BFZIV148Skin3D(
            em, external=prev.ext, pkg_shape=(4, 2, 4), pkg_k=.25,
            skin_depth=.002, skin_layers=8, skin_ratio=1.4,
            skin_quad=2, skin_near=4, skin_map=2, pkg_sea_h=20.,
            skin_heat="source")
        src = a.runs / f"{stem}_skin_source_subcell_{a.level}.npz"
        with np.load(src) as d:
            snap = d["temp_K"].copy()
            ts = d["time_s"].copy()
        if snap.shape[1] != et.n:
            raise ValueError(f"{pose}: physical snapshot/thermal operator mismatch")
        rom = HeatROM(et.cap, et.H, snap, rank=a.rank, ambient=et.amb)
        # The POD error measures thermal state representability only.
        pred = np.array([rom.decode(rom.encode(row)) for row in snap])
        err = abs(pred - snap)
        pkg = slice(et.count, et.count + 2*et.pkg_n)
        cu = et.cu
        sea = slice(0, et.count)
        result = dict(
            rank=rom.rank, nodes=et.n, snapshots=len(ts),
            retained=rom.retained,
            pkg_max_abs_K=float(np.max(err[:, pkg])),
            copper_max_abs_K=float(np.max(err[:, cu])),
            sea_mean_max_abs_K=float(np.max(abs(
                (pred[:, sea] - snap[:, sea]) @
                (em.vol / em.vol.sum())))),
            build_sec=time.perf_counter() - t0,
            model_file=str(a.runs / f"bfzi_heat_rom_{pose.lower()}.npz"),
            reference_file=str(src))
        rom.save(result["model_file"])
        report[pose] = result
        print(pose, json.dumps(result), flush=True)
    out = a.runs / "bfzi_heat_rom.json"
    out.write_text(json.dumps(report, indent=2))
    print("SAVED", out, flush=True)


if __name__ == "__main__":
    main()
