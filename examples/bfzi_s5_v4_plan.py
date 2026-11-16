"""Deterministic, geometry-disjoint V4 teacher-design space.

The first 3 training geometries and T04 development geometry retain their
original physical definitions. New cases sample the same validated V150 domain.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def drives(v):
    return [
        {"voltage": round(v * 0.78, 2), "off": [], "duration": 60},
        {"voltage": round(v * 1.08, 2),
         "off": [[45, 105], [195, 255]], "duration": 300},
        {"voltage": round(v, 2),
         "off": [[120, 210], [300, 390], [480, 540]],
         "duration": 600},
    ]


def design(seed=1504):
    rng = np.random.default_rng(seed)
    src = json.loads(Path("runs/geomfrac_s5_plan.json").read_text(encoding="utf-8"))
    cases = {k: src["cases"][k] for k in ("T01", "T02", "T03", "T04")}
    ids = ([f"N{i:02d}" for i in range(1, 28)]
           + [f"D{i:02d}" for i in range(1, 5)]
           + [f"E{i:02d}" for i in range(1, 6)])
    for i, key in enumerate(ids):
        a1 = round(float(rng.uniform(.69, .98)), 3)
        a2 = round(float(rng.uniform(.69, .98)), 3)
        g = {
            "alpha1": a1,
            "alpha2": a2,
            "core1_mm": int(rng.integers(8, 25)),
            "core2_mm": int(rng.integers(8, 25)),
            "shift_mm": [int(rng.integers(-9, 10)),
                         int(rng.integers(82, 109)),
                         int(rng.integers(-8, 9))],
            "tilt_deg": int(rng.integers(6, 18)),
            "yaw_deg": int(rng.integers(-9, 10)),
            "pitch_deg": int(rng.integers(-8, 9)),
            "box1_mm": int(rng.integers(217, 237)),
            "box2_mm": int(rng.integers(217, 237)),
        }
        if key == "N01":
            g.update(alpha1=.91, alpha2=.84, core1_mm=12, core2_mm=17,
                     shift_mm=[5, 89, -6], tilt_deg=11, yaw_deg=-2,
                     pitch_deg=5, box1_mm=223, box2_mm=226)
        if key == "N02":
            g.update(alpha1=.76, alpha2=.93, core1_mm=22, core2_mm=10,
                     shift_mm=[-7, 100, 6], tilt_deg=14, yaw_deg=4,
                     pitch_deg=-7, box1_mm=230, box2_mm=224)
        # M4 has 10 turns, outer radius 88.85 mm, and a nominal 5 mm
        # inward turn offset. Its BFZI-native pitch is computed from
        # (outer - inner_radius) / turns and depends on alpha and core.
        # Enforce both the inner straight length and the native
        # minimum bend radius (1.25 times conductor radius).
        for p in (1, 2):
            a = g[f"alpha{p}"]
            key_core = f"core{p}_mm"
            low = int(np.ceil((1. - a) * 50. + 1.))
            high = int(np.floor(
                (88.85 - 50. * a*a - 1.25 * .790569415 - 1.)
                / (1. + a)))
            if low > high:
                raise ValueError("V4 geometry has no valid M4 core")
            g[key_core] = min(high, max(low, g[key_core]))
        th = {
            "ins_k": round(float(rng.uniform(.17, .36)), 4),
            "ins_t": round(float(rng.uniform(.00033, .00047)), 6),
            "pkg_k": round(float(rng.uniform(.18, .36)), 4),
            "pkg_sea_h": round(float(rng.uniform(14, 36)), 3),
            "skin_m": .002,
        }
        cases[key] = {
            "family": "continuous",
            "donor": "g1" if i % 2 == 0 else "g2",
            "geometry": g,
            "alpha_per_K": round(float(rng.uniform(.016, .027)), 5),
            "thermal": th,
            "drives": drives(float(rng.uniform(65, 95))),
        }
    return {
        "description": "BFZI V150 S5 GeomFracET V4; independent geometry-level split",
        "train": ["T01", "T02", "T03"] + ids[:27],
        "val": ["T04"] + ids[27:31],
        "test": ids[31:],
        "cases": cases,
        "seed": seed,
        "grid": "ultra",
        "inner_step_s": 1.875,
    }


if __name__ == "__main__":
    path = Path("runs/geomfrac_s5_v4_plan.json")
    data = design()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("V4_PLAN", path, len(data["train"]), len(data["val"]),
          len(data["test"]))
