"""Same-physical-time spatial waveform convergence on actual TEAM-7 SaveLine fields.

Only official 200 Hz h10 BDF2 64/128/256 files; no experimental data.
Elemental B/J SaveLine fields are sampled diagnostics, not raw Whitney curls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from examples.bfzi_d2_team7_time import STEPS, sif_info

OBS = (
    (0, (22,), "A1_Bz"),
    (0, (20, 21, 22), "A1_Bxyz"),
    (1, (22,), "A2_Bz"),
    (2, (24,), "Jy_top"),
    (3, (24,), "Jy_bottom"),
    (2, (23, 24, 25), "Jxyz_top"),
    (3, (23, 24, 25), "Jxyz_bottom"),
)


def read(root, n, points=101):
    base = root / "h10"
    dt, cfg = sif_info(base / f"case_ene200s{n}.sif", n)
    file = base / f"resene200s{n}" / "f.dat"
    names = file.with_name("f.dat.names").read_text(encoding="utf-8")
    if "magnetic flux density e 3" not in names or "current density e 2" not in names:
        raise ValueError(f"Wrong Elmer fields: {file}")
    a = np.loadtxt(file)
    if a.shape != (n * 4 * points, 27) or not np.isfinite(a).all():
        raise ValueError(f"Wrong SaveLine field shape: {file}, {a.shape}")
    a = a.reshape(n, 4, points, 27)
    if not np.array_equal(a[:, 0, 0, 0], np.arange(1, n+1)):
        raise ValueError("SaveLine step order differs")
    if not np.array_equal(a[0, :, 0, 2], np.arange(1, 5)):
        raise ValueError("SaveLine line order differs")
    if not np.allclose(a[:, :, :, 4:7], a[0, :, :, 4:7], atol=1e-10, rtol=0):
        raise ValueError("Coordinates change with timestep")
    return a, cfg


def pair(data):
    u, v, w = data
    x = np.linalg.norm(u-v)
    y = np.linalg.norm(v-w)
    if np.linalg.norm(v) == 0 or np.linalg.norm(w) == 0:
        raise ValueError("Zero reference field")
    return {
        "rel64_128": float(x/np.linalg.norm(v)),
        "rel128_256": float(y/np.linalg.norm(w)),
        "order": float(np.log2(x/y)) if x > 0 and y > 0 else None,
    }


def calc(root, points=101):
    inp = {n:read(root, n, points) for n in STEPS}
    cfg = [inp[n][1] for n in STEPS]
    if cfg[0] != cfg[1] or cfg[1] != cfg[2]:
        raise ValueError("Elmer SIF physics mismatch")
    base = inp[64][0][0, :, :, 4:7]
    for n in (128,256):
        if not np.allclose(inp[n][0][0, :, :, 4:7], base, atol=1e-10, rtol=0):
            raise ValueError("Interlevel probe coordinates mismatch")
    results = {}
    for line, cols, name in OBS:
        vals = []
        for n in STEPS:
            step = n//64
            idx = np.arange(3*n//4+step-1, n, step)
            rows = inp[n][0][idx, line]
            vals.append(rows[..., list(cols)])
        end = [inp[n][0][-1, line, :, list(cols)] for n in STEPS]
        results[name] = {**pair(vals), "endpoint": pair(end)}
    return {
        "frequency_hz": 200, "mesh": "h10", "steps": list(STEPS),
        "sample_times_s": [float(k*0.02/64) for k in range(49,65)],
        "points_per_line": points, "observables": results,
        "source": "Elmer SaveLine elemental B/J, fixed 3D points; no experimental fitting",
        "status": "D2 OPEN",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = calc(a.root)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
