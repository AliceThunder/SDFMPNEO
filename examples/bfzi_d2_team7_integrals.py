"""Numerical h/R convergence of Elmer-native TEAM-7 energy and eddy power."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

CASES = ("h15", "h10", "h075", "r15", "r30")
PAIRS = (("h15", "h10", "h"), ("h10", "h075", "h"),
         ("r15", "h10", "R"), ("h10", "r30", "R"))


def input_sif(src, hz, steps, out):
    s = src.read_text(encoding="utf-8-sig")
    assert f"cos(2*3.14*{hz}*tx[0])" in s
    assert "TimeStep Intervals(1) = 64" in s
    assert "Active Solvers(4) = 1 2 3 4" in s
    assert "Active Solvers(3) = 2 3 4" in s
    s = s.replace(f'Results Directory "res{hz}"', f'Results Directory "{out}"')
    s = s.replace("TimeStep Intervals(1) = 64", f"TimeStep Intervals(1) = {steps}")
    s = s.replace("Active Solvers(4) = 1 2 3 4", "Active Solvers(4) = 1 2 3 6")
    s = s.replace("Active Solvers(3) = 2 3 4", "Active Solvers(3) = 2 3 6")
    i = s.index("Equation 1\n")
    return s[:i] + """Solver 6
  Exec Solver = Always
  Equation = "SaveScalars"
  Procedure = "SaveData" "SaveScalars"
  Filename = "phys.dat"
  Show Norm Index = 1
End

""" + s[i:]


def read_phys(path):
    names = path.with_name("phys.dat.names").read_text(encoding="utf-8")
    assert "1: res: eddy current power" in names
    assert "2: res: electromagnetic field energy" in names
    a = np.atleast_2d(np.loadtxt(path))
    if a.shape[1] != 2:
        raise ValueError(f"Expected two native scalar quantities, got {a.shape}")
    result = {"eddy_power_W": float(a[-1, 0]),
              "em_energy_J": float(a[-1, 1]), "rows": len(a)}
    if len(a) >= 32 and len(a) % 16 == 0:
        result["power_period_rel"] = abs(a[-1, 0]-a[-17, 0])/abs(a[-1, 0])
        result["energy_period_rel"] = abs(a[-1, 1]-a[-17, 1])/abs(a[-1, 1])
    return result


def run(root, el, name, hz, steps):
    d = root / name
    out = f"resene{hz}s{steps}"
    sf = d / f"case_ene{hz}s{steps}.sif"
    sf.write_text(input_sif(d / f"TEAM7_{hz}.sif", hz, steps, out), encoding="utf-8")
    (d / out).mkdir(exist_ok=True)
    env = os.environ.copy()
    env["ELMER_HOME"] = str(el)
    env["PATH"] = str(el / "bin") + os.pathsep + env.get("PATH", "")
    t = time.perf_counter()
    with (d / f"ene{hz}s{steps}.log").open("w") as log:
        proc = subprocess.run([str(el / "bin" / "ElmerSolver.exe"), sf.name],
                              cwd=d, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError(f"Elmer {name}/{hz} failed: {proc.returncode}")
    res = read_phys(d / out / "phys.dat")
    if res["rows"] != steps:
        raise ValueError(f"Unexpected SaveScalars steps: {name}/{hz}: {res['rows']}")
    return {"case": name, "hz": hz, "seconds": round(time.perf_counter()-t, 2), **res}


def collect(root, hz, steps):
    result = {}
    for name in CASES:
        out = root/name/f"resene{hz}s{steps}"/"phys.dat"
        # Preserve original pilot output directory naming.
        if steps == 1 and not out.exists():
            out = root/name/f"resene{hz}"/"phys.dat"
        if not out.exists():
            raise FileNotFoundError(out)
        item = read_phys(out)
        if item["rows"] != steps:
            raise ValueError(f"Step mismatch for {out}: {item['rows']}")
        result[name] = item
    return result


def compare(data):
    pairs = []
    for a, b, axis in PAIRS:
        row = {"axis": axis, "from": a, "to": b}
        for field in ("em_energy_J", "eddy_power_W"):
            v = data[b][field]
            row[field + "_rel"] = abs(data[a][field]-v)/abs(v)
        pairs.append(row)
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--elmer", type=Path)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    if args.solve:
        if not args.elmer:
            raise ValueError("--solve requires --elmer")
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            jobs = [pool.submit(run, args.root, args.elmer, name, hz, args.steps)
                    for hz in (50, 200) for name in CASES]
            for f in cf.as_completed(jobs):
                print("COMPLETED", f.result(), flush=True)
    runs = {}
    for hz in (50, 200):
        data = collect(args.root, hz, args.steps)
        runs[str(hz)] = {"cases": data, "pairs": compare(data)}
    out = {"kind": "Elmer native integrated electromagnetic field energy and eddy power",
           "source": "TEAM-7 3D physical body coil",
           "measurements_used": False, "steps": args.steps,
           "phase": "first time step" if args.steps == 1 else "BDF2 time integration",
           "status": "OPEN", "runs": runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for hz, data in runs.items():
        for x in data["pairs"]:
            print(hz, x["axis"], x["from"], "vs", x["to"],
                  f"energy={100*x['em_energy_J_rel']:.5f}%",
                  f"eddy={100*x['eddy_power_W_rel']:.5f}%")


if __name__ == "__main__":
    main()
