"""Evaluate the frozen V8 BFZI-free model on validation or sealed holdout geometries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.bfzi_geom_multi import prepare
from examples.bfzi_geom_free_eval import metric
from sdfmpneo_bfzi.geom_coldnet import ColdNet
from sdfmpneo_bfzi.geom_free import rollout
from sdfmpneo_bfzi.geom_init import GeomInit, thermal_mix
from sdfmpneo_bfzi.geom_neural import GeomFracET
from sdfmpneo_bfzi.geom_parallel import ParZ
from sdfmpneo_bfzi.geom_par48 import ParZ48
from sdfmpneo_bfzi.geom_weakz import WeakSea
from sdfmpneo_bfzi.geom_skintan import restore_rank48
from sdfmpneo_bfzi.geom_srcfix import SrcFix


def avg(rows, key):
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return float(np.mean(vals)) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--atlas", type=Path, default=Path("runs/geomfrac_init_v8_expand.npz"))
    p.add_argument("--cold-model", type=Path, default=Path("runs/geomfrac_cold_v8_frozen.pt"))
    p.add_argument("--config", type=Path, default=Path("runs/geomfrac_free_rank48_selection.json"))
    p.add_argument("--skin-tangent", type=Path, default=Path("runs/geomfrac_rank48_skin_tan_v3.pt"))
    p.add_argument("--skin-base", type=Path, default=Path("runs/geomfrac_et_v9_ztune.pt"))
    p.add_argument("--parallel-model", type=Path, default=None)
    p.add_argument("--k-mix", type=float, default=0.0)
    p.add_argument("--weak-sea-mix", type=float, default=0.0)
    p.add_argument("--skin-base-model", type=Path, default=None)
    p.add_argument("--skin-mix", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()

    if not 0.0 <= a.weak_sea_mix <= 1.0:
        p.error("--weak-sea-mix must be in [0,1]")
    torch.set_num_threads(2)
    plan = json.loads(a.plan.read_text(encoding="utf-8-sig"))
    mf = json.loads(a.manifest.read_text(encoding="utf-8-sig"))
    cfg = json.loads(a.config.read_text(encoding="utf-8-sig"))
    specs = mf.get(a.split, [])
    names = [Path(s["anchor"].replace("\\", "/")).stem.split("_")[2]
             for s in specs]
    if not specs or sorted(names) != sorted(plan.get(a.split, [])):
        raise ValueError("manifest geometries do not match the frozen plan")
    if len(set(names)) != len(names):
        raise ValueError("duplicate geometry in acceptance manifest")

    ck = torch.load(a.model, map_location="cpu", weights_only=True)
    if int(ck["rank"]) != 48:
        raise ValueError("V8 acceptance requires the rank48 heat model")
    net = GeomFracET(ng=int(ck["ng"]), rank=48,
        soe_quad=ck.get("soe_quad", "trap"),
        heat_dt=float(ck.get("heat_dt", 1.875)),
        heat_z=bool(ck.get("heat_z", False)))
    net.load_state_dict(ck["state"])
    net.eval()
    eck = dict(src_scale=ck["src_scale"], y_scale=ck["y_scale"],
               a_scale=ck["a_scale"], canonical=True, port_sea=True)

    atlas = GeomInit.load(a.atlas)
    cold = ColdNet.load(a.cold_model)
    src = SrcFix.load(cfg["source_model"])
    tan_new = restore_rank48(a.skin_tangent, a.skin_base)
    if a.skin_base_model:
        if not 0.0 <= a.skin_mix <= 1.0:
            p.error("--skin-mix must be in [0,1]")
        tan_old = restore_rank48(a.skin_base_model, a.skin_base)
        alpha = float(a.skin_mix)
        def tan(g, state):
            return (1.0-alpha)*tan_old(g, state) + alpha*tan_new(g, state)
        tan.time_conditioned = False
    else:
        tan = tan_new
        alpha = 1.0
    if a.parallel_model:
        pck = torch.load(a.parallel_model, map_location="cpu", weights_only=True)
        extra = (ParZ48.load(a.parallel_model) if "am" in pck["state"]
                 else ParZ.load(a.parallel_model))
    else:
        extra = None
    mix = float(cfg["shape_mix"])

    rows = []
    with torch.no_grad():
        for spec, name in zip(specs, names):
            case = plan["cases"][name]
            init = atlas.predict(case)
            init = cold.apply(case, init)
            init = src.apply(case, init)
            init = thermal_mix(atlas, init, a.k_mix)
            ex = (WeakSea(tan, extra, init, case["alpha_per_K"], a.weak_sea_mix)
                  if a.weak_sea_mix else extra)
            for row in prepare(spec, 48):
                pred = rollout(case, atlas, net, eck, row["time"], row["voltage"],
                               tan, init_override=init, extra=ex, shape_mix=mix)
                item = metric(pred, row)
                item["geometry"] = name
                rows.append(item)
            count = sum(r["geometry"] == name for r in rows)
            print("V8_ACCEPT", name, count, flush=True)

    expected = sum(len(s["teacher"]) if isinstance(s["teacher"], list) else 1
                   for s in specs)
    if len(rows) != expected:
        raise ValueError(f"expected {expected} histories, evaluated {len(rows)}")
    keys = ("cold_z_rel_L2", "dynamic_z_rel_L2", "current_rel_L2",
            "material_mean_rmse_K", "material_mean_max_K",
            "sea_power_rel_L2", "copper_power_rel_L2",
            "R11_dynamic_rel_L2_self", "X11_dynamic_rel_L2_self",
            "R22_dynamic_rel_L2_self", "X22_dynamic_rel_L2_self")
    overall = {k: avg(rows, k) for k in keys}
    overall["energy_budget_max_W"] = float(max(r["energy_budget_W"] for r in rows))
    per = {}
    for name in names:
        part = [r for r in rows if r["geometry"] == name]
        per[name] = {k: avg(part, k) for k in keys}
        per[name]["energy_budget_max_W"] = float(
            max(r["energy_budget_W"] for r in part))
    nums = [v for r in rows for v in r.values()
            if isinstance(v, (int, float, np.number))]
    if not np.isfinite(nums).all():
        raise ValueError("non-finite V8 acceptance metric")

    out = dict(status="complete", split=a.split, geometry_count=len(names),
        history_count=len(rows), model=str(a.model), atlas=str(a.atlas),
        cold_model=str(a.cold_model), parallel_model=str(a.parallel_model) if a.parallel_model else None,
        skin_mix=alpha, skin_base_model=str(a.skin_base_model) if a.skin_base_model else None,
        shape_mix=mix, k_mix=a.k_mix, weak_sea_mix=a.weak_sea_mix,
        online_BFZI_solves=0, geometry_specific_heat_basis_builds=0,
        overall=overall, per_geometry=per, rows=rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("V8_ACCEPT_SUMMARY", json.dumps(overall), flush=True)


if __name__ == "__main__":
    main()
