"""Compare TEAM-7 50/200 Hz h/R scans on frozen physical Bz/Jy lines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from examples.bfzi_d2_team7_elmer_eval import evaluate, sample


def field(path):
    data = np.loadtxt(path)
    step = int(data[:, 0].max())
    b = sample(data, step)[:, [4, 22]]
    j = sample_line(data, step, 3)
    return b, j[:, [4, 24]], data, step


def sample_line(data, step, line):
    v = data[(data[:, 0] == step) & (data[:, 2] == line)]
    if len(v) != 101:
        raise ValueError(f"Expected 101 points on line {line}, got {len(v)}")
    return v[np.argsort(v[:, 4])]


def diff(x, y):
    if not np.allclose(x[:, 0], y[:, 0], atol=1e-10, rtol=0):
        raise ValueError("Probe locations differ")
    error = x[:, 1] - y[:, 1]
    norm = np.linalg.norm(y[:, 1])
    return {"abs_rms": float(np.sqrt(np.mean(error**2))),
            "abs_max": float(np.max(np.abs(error))),
            "rel_l2": float(np.linalg.norm(error)/norm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("runs/team7_hr"))
    ap.add_argument("--freq", type=int, choices=(50, 200), required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    names = ("h15", "h10", "h075", "r15", "r30")
    fields = {}
    for name in names:
        p = args.root / name / f"f{args.freq}" / "res" / "f.dat"
        if p.exists():
            fields[name] = field(p)
    if not fields:
        raise ValueError("No completed f.dat")
    rows = []
    for name, (b, j, data, step) in fields.items():
        row = {"case": name, "freq_hz": args.freq, "step": step,
               "mesh_ref": "h075" if name.startswith("h") else "r30"}
        dt = .00125*50/args.freq
        if step == 64:
            row["measurement"] = evaluate(
                args.root/name/f"f{args.freq}"/"res"/"f.dat", args.freq, dt,
                step=step)
        ref = fields.get(row["mesh_ref"])
        if ref:
            row["Bz_ref_diff"] = diff(b, ref[0])
            row["Jy_ref_diff"] = diff(j, ref[1])
        rows.append(row)
    output = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(output)
    print(output)


if __name__ == "__main__":
    main()

