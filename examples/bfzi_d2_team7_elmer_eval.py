"""Evaluate REAL Elmer TEAM-7 SaveLine fields against fixed measured Bz.

Input is ElmerSolver's res/f.dat from the verified CSC TEAM7.sif,
not a result synthesized by this repository. The first data columns
are timestep, time index, polyline number and x-sample number.
Reference: TEAM-7 A1-B1, omega*t=0 deg, 50/200 Hz, peak 2742 A.turn.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REF = (Path(__file__).resolve().parents[1] /
       "docs/research/bfzi_uwpt_electrothermal/evidence/"
       "team7_a1b1_measured_bz_0deg.csv")


def sample(data, step):
    rows = data[(data[:, 0] == step) & (data[:, 2] == 1)]
    if len(rows) != 101:
        raise ValueError(f"A1-B1 at step {step}: expected 101 samples, got {len(rows)}")
    rows = rows[np.argsort(rows[:, 4])]
    if not np.all(np.diff(rows[:, 4]) > 0):
        raise ValueError("A1-B1 x coordinates must increase")
    if not np.all(np.isfinite(rows)):
        raise ValueError("Elmer probe data contain nonfinite values")
    return rows


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def evaluate(file, freq, dt, step=None, ref=REF):
    data = np.loadtxt(file)
    if data.ndim != 2 or data.shape[1] < 26:
        raise ValueError("Expected original Elmer TEAM7 SaveLine f.dat with >=26 columns")
    n = int(np.max(data[:, 0])) if step is None else step
    npd = round(1.0 / (freq * dt))
    if abs(npd * freq * dt - 1.0) > 1e-8:
        raise ValueError("A period must contain an integer number of steps")
    if n % npd:
        raise ValueError("The requested step must end at excitation phase 0 degrees")

    row = sample(data, n)
    refdata = np.genfromtxt(ref, delimiter=",", names=True)
    x = np.asarray(refdata["x_m"])
    y = np.asarray(refdata[f"Bz_{freq}Hz_T"])
    if x[0] < row[0, 4] - 1e-9 or x[-1] > row[-1, 4] + 1e-9:
        raise ValueError("Measurement points exceed Elmer A1-B1 sample range")
    pred = np.interp(x, row[:, 4], row[:, 22])
    out = {
        "source": str(Path(file)),
        "reference": str(ref),
        "freq_hz": freq,
        "dt_s": dt,
        "step": n,
        "time_s": n * dt,
        "reference_phase_deg": 0,
        "points": len(x),
        "Bz_rel_L2_to_measurement": rel(pred, y),
        "Bz_max_abs_diff_T": float(np.max(np.abs(pred - y))),
        "Bz_0_T": float(row[0, 22]),
        "Bz_max_abs_on_line_T": float(np.max(np.abs(row[:, 22]))),
    }
    if n > npd:
        prev = sample(data, n - npd)
        if not np.allclose(row[:, 4:7], prev[:, 4:7], rtol=0, atol=1e-9):
            raise ValueError("Period-to-period probes have different coordinates")
        out["B_xyz_previous_period_rel_L2"] = rel(
            prev[:, 20:23], row[:, 20:23])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--freq", type=int, choices=[50, 200], required=True)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--step", type=int)
    p.add_argument("--ref", type=Path, default=REF)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    out = evaluate(args.data, args.freq, args.dt, args.step, args.ref)
    txt = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(txt, encoding="utf-8")
    print(txt, end="")


if __name__ == "__main__":
    main()
