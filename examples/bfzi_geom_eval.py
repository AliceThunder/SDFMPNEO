"""Independent autonomous evaluation of a trained cross-geometry surrogate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_neural import eval_rollout
from sdfmpneo_bfzi.geom_neural import GeomFracET


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--tangent-model",type=Path,default=None)
    p.add_argument("--z-fix-model",type=Path,default=None,
                   help="geometry/thermal-state nonlinear dynamic-Z residual")
    p.add_argument("--pkg-model", type=Path, default=None)
    p.add_argument("--lift-model", type=Path, default=None,
                   help="optional self-consistent sea-skin thermal feedback")
    p.add_argument("--lift-gain", type=float, default=1.,
                   help="fraction of learned sea-skin feedback to apply")
    p.add_argument("--skin-model", type=Path, default=None)
    p.add_argument("--skin-gain", type=float, default=1.)
    p.add_argument("--skin-output-model", type=Path, default=None,
                   help="postprocessed V7 seawater-skin field, without feedback")
    p.add_argument("--bulk-output-model", type=Path, default=None,
                   help="V8 extra 16-mode bulk-sea temperature output")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test"), default="val")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    ck = torch.load(a.model, map_location="cpu", weights_only=True)
    cfg = json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    specs = cfg.get(a.split, [])
    if not specs:
        p.error(f"no geometry in split {a.split}")
    net = GeomFracET(
        ng=int(ck["ng"]), rank=int(ck["rank"]),
        soe_quad=ck.get("soe_quad", "trap"),
        heat_dt=float(ck.get("heat_dt", 1.875)),
        heat_z=bool(ck.get("heat_z", False)))
    net.load_state_dict(ck["state"])
    rows = [r for spec in specs for r in prepare(spec, int(ck["rank"]))]
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
    if a.lift_model:
        with np.load(a.lift_model) as data:
            lift=data['w']*a.lift_gain
    skin=None
    if a.skin_model:
        from sdfmpneo_bfzi.geom_skin import restore
        skin=restore(a.skin_model)
    skin_out=None
    if a.skin_output_model:
        from sdfmpneo_bfzi.geom_skin import restore
        skin_out=restore(a.skin_output_model)
    bulk_out=None
    if a.bulk_output_model:
        from sdfmpneo_bfzi.geom_skin import restore
        bulk_out=restore(a.bulk_output_model)
        if skin_out is None:
            p.error("--bulk-output-model requires --skin-output-model")
    vals = eval_rollout(None, net, ck, rows, tangent=tangent, pkg=pkg,
                        lift=lift, skin=skin, skin_gain=a.skin_gain,
                        skin_out=skin_out, bulk_out=bulk_out)
    out = dict(
        model=str(a.model),tangent_model=str(a.tangent_model) if a.tangent_model else None,
        z_fix_model=str(a.z_fix_model) if a.z_fix_model else None,
        pkg_model=str(a.pkg_model) if a.pkg_model else None,
        lift_model=str(a.lift_model) if a.lift_model else None,
        lift_gain=a.lift_gain,
        skin_model=str(a.skin_model) if a.skin_model else None,
        skin_gain=a.skin_gain,
        skin_output_model=str(a.skin_output_model) if a.skin_output_model else None,
        bulk_output_model=str(a.bulk_output_model) if a.bulk_output_model else None,
        manifest=str(a.manifest), split=a.split,
        geometries=len(specs), histories=len(vals),
        dynamic_Z_mean_rel_L2=float(np.mean(
            [r["dynamic_Z_rel_L2"] for r in vals])),
        results=vals)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("EVAL", a.out, "geometries", len(specs), "histories",
          len(vals), "dynamic_Z_mean_rel_L2",
          round(out["dynamic_Z_mean_rel_L2"], 6), flush=True)


if __name__ == "__main__":
    main()
