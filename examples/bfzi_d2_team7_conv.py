"""TEAM-7 numerical-only h/R convergence of fixed physical field observables.

Uses independent Elmer 101-point SaveLine fields; no experimental samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SERIES = {
    "h": ("h15", "h10", "h075"),
    "R": ("r15", "h10", "r30"),
    "air": ("a40", "h10", "a25"),
    "probe": ("probe10", "probe120", "probe240"),
}
OBS = ((1, (22,), "A1_Bz"), (1, (20, 21, 22), "A1_Bxyz"),
       (2, (22,), "A2_Bz"), (2, (20, 21, 22), "A2_Bxyz"),
       (3, (24,), "Jy_top"), (4, (24,), "Jy_bottom"))


def load(path, step=64):
    """Read a finished elemental SaveLine with four fixed 101-point lines."""
    if not path.exists():
        return None
    names = path.with_name(path.name + ".names")
    if names.exists():
        s = names.read_text(encoding="utf-8")
        if "magnetic flux density e 3" not in s or "current density e 2" not in s:
            raise ValueError(f"Expected elemental B/J fields: {path}")
    a = np.loadtxt(path)
    if a.ndim != 2 or a.shape[1] != 27 or int(a[:, 0].max()) != step:
        raise ValueError(f"Unexpected field layout or final step in {path}")
    lines = {}
    for k in (1, 2, 3, 4):
        q = a[(a[:, 0] == step) & (a[:, 2] == k)]
        if len(q) != 101:
            raise ValueError(f"Expected 101 final-step nodes: {path}, line {k}")
        lines[k] = q[np.argsort(q[:, 4])]
    return lines


def cmp(a, b):
    """Compare two fields; relative errors normalized to the second field."""
    out = {}
    for k, cols, key in OBS:
        x = a[k][:, 4]
        if not np.allclose(x, b[k][:, 4], rtol=0, atol=1e-10):
            raise ValueError(f"Probe coordinates mismatch on line {k}")
        u, v = a[k][:, cols], b[k][:, cols]
        den = np.linalg.norm(v)
        if den <= 0:
            raise ValueError(f"Zero reference field: {key}")
        out[key + "_rel"] = float(np.linalg.norm(u-v) / den)
        out[key + "_rms_delta"] = float(np.sqrt(np.mean((u-v)**2)))
        if key in ("A1_Bz", "A2_Bz"):
            z, w = u[:, 0], v[:, 0]
            length = x[-1]-x[0]
            avg = lambda y: float(np.trapezoid(y, x)/length)
            rms = lambda y: float(np.sqrt(avg(y*y)))
            ref = rms(w)
            out[key + "_mean_delta_T"] = avg(z-w)
            out[key + "_mean_delta_over_rms"] = abs(avg(z-w))/ref
            out[key + "_rms_rel"] = abs(rms(z)-rms(w))/ref
    return out


def scan(root, hz):
    result = {}
    for kind, cases in SERIES.items():
        vals = {n: load(root/n/f"res{hz}"/"f.dat") for n in cases}
        rows = []
        for n, nxt in zip(cases[:-1], cases[1:]):
            if vals[n] is None or vals[nxt] is None:
                continue
            rows.append({"from": n, "to": nxt, **cmp(vals[n], vals[nxt])})
        if rows:
            result[kind] = rows
    return result


def markdown(data):
    cols = ("A1_Bz_rel", "A1_Bxyz_rel", "A1_Bz_mean_delta_over_rms",
            "A1_Bz_rms_rel", "A2_Bz_rel", "Jy_top_rel", "Jy_bottom_rel")
    lines = [
        "# D2 TEAM-7 numerical convergence (no experimental-data fitting)",
        "",
        "All comparisons use 101 fixed physical points at the fourth-period "
        "end phase, original elemental B and Jy. Percentages use the second "
        "case as denominator. The signed Bz line-mean difference is normalized "
        "by the reference Bz line RMS to avoid cancellation or near-zero mean.",
        "",
        "| Frequency | Independent sweep | Pair | A1 Bz L2 | A1 Bxyz L2 | "
        "A1 mean Δ / RMS | A1 RMS Δ | A2 Bz L2 | Top Jy L2 | Bottom Jy L2 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for hz, studies in data["runs"].items():
        for typ, rows in studies.items():
            for row in rows:
                fmt = [f"{100*row[c]:.5f}%" for c in cols]
                lines.append(f"| {hz} Hz | {typ} | {row['from']} → {row['to']} | "
                             + " | ".join(fmt) + " |")
    lines += [
        "",
        "**Decision: OPEN.** These are numerical intermesh differences, not "
        "experimental errors or certified errors to a continuum solution. "
        "Keep full-field H(curl), outer-domain, source and time consistency "
        "gates active. Small line-integrated differences alone do not "
        "certify convergence of the spatial B/J field.",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/team7_hr"))
    p.add_argument("--json", type=Path)
    p.add_argument("--md", type=Path)
    args = p.parse_args()
    data = {
        "reference": "Elmer TEAM-7 elemental B and Jy, step 64, fixed physical lines",
        "measurements_used": False,
        "status": "OPEN",
        "runs": {str(hz): scan(args.root, hz) for hz in (50, 200)},
    }
    s = markdown(data)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(s, encoding="utf-8")
    print(s)


if __name__ == "__main__":
    main()

