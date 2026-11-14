"""A2: six fixed independent BFZI V150 electrothermal validation histories.

Uses the locked original BFZI subcell + source-skin physics.  Two new
non-overlapping geometries each run at medium/extra; two additional
independent drive/material histories use the extra grid.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from examples.bfzi_param import create, originals
from examples.bfzi_native_sea_study import build_args
from examples.bfzi_obb_refine import load
from examples.bfzi_obb_layer_study import GRIDS, compare
from examples.bfzi_obb_subcell_study import measure
from sdfmpneo_bfzi.bfzi_v148_adapter import BFZIV148Adapter
from sdfmpneo_bfzi.geom_heat import GeomHeat
from sdfmpneo_bfzi.sea_layer import BFZIV148Skin3D


CASES = (
    ("G1", "P03", "medium", 20., 1., 1.),
    ("G1", "P03", "extra", 20., 1., 1.),
    ("G2", "P07", "medium", 25., .5, 1.),
    ("G2", "P07", "extra", 25., .5, 1.),
    ("G1_drive", "P03", "extra", 20., .65, 1.),
    ("G2_mat", "P07", "extra", 25., .5, .8),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bfzi-root", type=Path, required=True)
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("runs/bfzi_a2.json"))
    p.add_argument("--cases", nargs="+", default=None)\n    a = p.parse_args()
    bf, v150 = originals(a.bfzi_root)
    old = json.loads((a.runs / "bfzi_obb_physical_study.json").read_text())
    out = (json.loads(a.out.read_text()) if a.out.exists()\n           else dict(method="BFZI V150 A2 independent physics", cases={}, cross_grid={}))
    saved = {}
    for name, pose, level, dy, gain, sig in CASES:
        if a.cases is not None and name not in a.cases:
            continue
        stem = f"bfzi_pose_{pose.lower()}"
        meta = json.loads((a.runs / f"{stem}_anchor.json").read_text())
        geo = build_args(meta)
        shift = np.asarray(old["geometries"][pose]["new_shift_mm"], float).copy()
        shift[1] += dy
        geo["shift_mm"] = shift
        prev = GeomHeat.load(a.runs / f"{stem}_disjoint_anchor.npz")
        prog = load(a.runs / f"{stem}_disjoint_oriented_medium.npz")
        t = np.asarray(prog["time_s"], float)
        v = np.asarray(prog["voltage_v"], float) * gain
        if name == "G1_drive":
            # A second shutdown interval probes heat-memory evolution.
            v = v.copy()
            v[(t >= 240.) & (t < 330.)] = 0.
        xz, y = GRIDS[level]
        start = time.perf_counter()
        data, _ = create(bf, v150, "M4", **geo, sea_void="oriented",
                         physical_box_only=True, grid_xz=xz, grid_y=y)
        em = BFZIV148Adapter(data, bf, heat_mode="port-weak",
                             fv_faces="exact", source_mask="subcell")
        if sig != 1.:
            # Vary the temperature coefficient without changing the cold
            # Green operator or its reference conductivity.
            em.alpha *= sig
        et = BFZIV148Skin3D(
            em, external=prev.ext, pkg_shape=(4, 2, 4), pkg_k=.25,
            skin_depth=.002, skin_layers=8, skin_ratio=1.4,
            skin_quad=2, skin_near=4, skin_map=2, pkg_sea_h=20.,
            skin_heat="source")
        setup = time.perf_counter() - start
        res = measure(et, t, v)
        path = a.runs / f"bfzi_a2_{name.lower()}_{level}.npz"
        np.savez_compressed(
            path, time_s=t, voltage_v=v, temp_K=res["temp_K"],
            z6=res["z6"], sea_T_K=res["sea_T_K"],
            Psea_W=res["Psea_W"], Pcu_W=res["Pcu_W"],
            sea_xyz=res["sea_xyz"], sea_vol=res["sea_vol"],
            sea_q=res["sea_q"], sea_J=res["sea_J"],
            field_times_s=res["field_times_s"], nsea=et.count,
            n_pkg=et.pkg_n)
        saved[name, level] = res
        out["cases"][name + "_" + level] = dict(
            anchor=pose, extra_shift_y_mm=dy, voltage_scale=gain,
            sea_alpha_scale=sig, wet_cells=et.count,
            setup_sec=setup, direct_sec=res["direct_sec"],
            max_port_weak_rel=res["max_port_weak_rel"],
            max_node_budget_W=res["max_node_budget_W"],
            file=str(path))
        for g in ("G1", "G2"):
            if (g, "medium") in saved and (g, "extra") in saved:
                m, e = saved[g, "medium"], saved[g, "extra"]
                out["cross_grid"][g] = dict(
                    **compare(m, e),
                    max_sea_mean_abs_K=float(np.max(np.abs(
                        m["sea_T_K"] @ m["sea_vol"] / m["sea_vol"].sum()
                        - e["sea_T_K"] @ e["sea_vol"] / e["sea_vol"].sum()))))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))
        print(name, level, et.count, "power", res["max_port_weak_rel"],
              "heat", res["max_node_budget_W"], flush=True)
    print("SAVED", a.out, flush=True)


if __name__ == "__main__":
    main()
