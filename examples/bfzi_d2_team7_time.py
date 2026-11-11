"""Three-level 200 Hz TEAM-7 BDF2 time convergence of native Elmer integrals.

Read only completed original 3D solves. Check same SIF physics and true
physical endpoint before computing a measured order or extrapolation.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

STEPS = (64, 128, 256)
T_END = 0.02


def sif_info(path, steps):
    s = path.read_text(encoding="utf-8-sig")
    def get(pat):
        m = re.search(pat, s, flags=re.I | re.M)
        if not m:
            raise ValueError(f"Missing SIF setting {pat}: {path}")
        return m.group(1)
    n = int(get(r"^\s*TimeStep Intervals\(1\)\s*=\s*(\d+)\s*$"))
    dt = float(get(r"^\s*Timestep Sizes\(1\)\s*=\s*Real\s+([\d.eE+-]+)\s*$"))
    order = int(get(r"^\s*BDF Order\s*=\s*(\d+)\s*$"))
    out = get(r'^\s*Results Directory\s+"([^"]+)"\s*$')
    if (n != steps or order != 2 or not math.isclose(dt*n, T_END, rel_tol=1e-12)
            or not math.isclose(dt, T_END/steps, rel_tol=1e-12)
            or out != f"resene200s{steps}"):
        raise ValueError(f"Time discretization mismatch: {path}: n={n} dt={dt} BDF={order}")
    if "cos(2*3.14*200*tx[0])" not in s:
        raise ValueError(f"Unexpected original TEAM-7 excitation: {path}")
    norm = re.sub(r'(^\s*Results Directory\s+)"[^"]+"', r'\1"<OUT>"', s, flags=re.I | re.M)
    norm = re.sub(r'(^\s*TimeStep Intervals\(1\)\s*=\s*)\d+', r'\g<1><N>', norm, flags=re.I | re.M)
    norm = re.sub(r'(^\s*Timestep Sizes\(1\)\s*=\s*Real\s+)[\d.eE+-]+', r'\g<1><DT>', norm, flags=re.I | re.M)
    return dt, norm


def read_run(root, steps):
    d = root / "h10"
    dt, cfg = sif_info(d / f"case_ene200s{steps}.sif", steps)
    res = d / f"resene200s{steps}"
    f = res / "phys.dat"
    names = (res / "phys.dat.names").read_text(encoding="utf-8")
    if ("1: res: eddy current power" not in names
            or "2: res: electromagnetic field energy" not in names):
        raise ValueError(f"Native Elmer column names changed: {f}")
    log = (d / f"ene200s{steps}.log").read_text(encoding="utf-8", errors="replace")
    if "Elmer Solver: ALL DONE" not in log:
        raise ValueError(f"Elmer solve not complete: {steps} time steps")
    a = np.atleast_2d(np.loadtxt(f))
    if a.shape != (steps, 2) or not np.isfinite(a).all():
        raise ValueError(f"Incomplete or invalid native Elmer integrals: {f}, shape={a.shape}")
    k = steps // 4
    row = dict(
        steps=steps, dt_s=dt, time_s=steps*dt,
        eddy_W=float(a[-1, 0]), energy_J=float(a[-1, 1]),
        power_period_rel=float(abs(a[-1, 0] - a[-1-k, 0])/abs(a[-1, 0])),
        energy_period_rel=float(abs(a[-1, 1] - a[-1-k, 1])/abs(a[-1, 1])),
    )
    return row, cfg


def compare(rows, field):
    u, v, w = [z[field] for z in rows]
    x, y = u-v, v-w
    p = math.log2(abs(x/y)) if x*y > 0 else None
    lim = w + (w-v)/(2**p-1) if p is not None and p > 0 else None
    return dict(
        rel64_128=abs(x)/abs(v), rel128_256=abs(y)/abs(w),
        order=p, extrapolated=lim,
        remaining_relative=abs(lim-w)/abs(lim) if lim is not None and lim else None,
    )


def calc(root):
    read = [read_run(root, n) for n in STEPS]
    rows, configs = zip(*read)
    if configs[0] != configs[1] or configs[1] != configs[2]:
        raise ValueError("Input SIF physics differ beyond dt, steps and result directory")
    res = {field: compare(rows, field) for field in ("eddy_W", "energy_J")}
    return dict(
        frequency_hz=200, mesh="h10", R_m=0.2,
        scheme="BDF2", source="CSC TEAM7 original three-component cos(2*3.14*f*t)",
        measurement_used=False, rows=list(rows), comparison=res, gate="OPEN",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = calc(a.root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
