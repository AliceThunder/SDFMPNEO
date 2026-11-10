"""TEAM-7 independent Gmsh h/R scans using the CSC reference geometry and SIF.

Example:
  python -m examples.bfzi_d2_team7_hr --geo D:/work/Elmer-elmag-ref/TEAM7/gmsh/TEAM7.geo --sif D:/work/Elmer-elmag-ref/TEAM7/TEAM7.sif --out runs/team7_hr --meshes
  python -m examples.bfzi_d2_team7_hr --sif D:/work/Elmer-elmag-ref/TEAM7/TEAM7.sif --out runs/team7_hr --solve --freq 50
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import time

CASES = (("h15", 1.5, .20), ("h10", 1.0, .20), ("h075", .75, .20),
         ("r15", 1.0, .15), ("r30", 1.0, .30))
SIZE = (("lc_coil", 20), ("lc_plate", 10), ("lc_hole", 10), ("lc_air", 30))


def mesh(geo, out, eg):
    import gmsh

    raw = geo.read_text()
    for name, factor, radius in CASES:
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        src = raw
        for var, mm in SIZE:
            pattern = r"(?m)^" + var + r"\s*=.*?;"
            src, n = re.subn(pattern, f"{var} = {mm*factor:.8g}*mm;", src, count=1)
            if n != 1:
                raise ValueError(f"Missing {var} in CSC TEAM7.geo")
        src, n = re.subn(r"(?m)^boxt\s*=.*?;", f"boxt = {1000*radius:.8g}*mm;", src, count=1)
        if n != 1:
            raise ValueError("Missing boxt in CSC TEAM7.geo")
        src = src.replace("394*mm+boxt", "194*mm+2*boxt")
        path = folder / "TEAM7.geo"
        path.write_text(src)
        gmsh.initialize()
        try:
            gmsh.option.setNumber("General.Terminal", 0)
            gmsh.open(str(path))
            gmsh.model.mesh.generate(3)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.Binary", 0)
            msh = folder / "TEAM7.msh"
            gmsh.write(str(msh))
            tetra = sum(len(x) for x in gmsh.model.mesh.getElements(3)[1])
        finally:
            gmsh.finalize()
        with (folder / "grid.log").open("w") as log:
            subprocess.run([str(eg), "14", "2", str(msh), "-names",
                            "-out", str(folder / "TEAM7")], stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"{name}: {tetra} tetrahedra; radius={radius} m; h factor={factor}", flush=True)


def solve(sif, out, exe, freq):
    raw = sif.read_text()
    if "cos(2*3.14*50*tx[0])" not in raw:
        raise ValueError("Input SIF is not the CSC 50Hz AC TEAM7.sif")
    if freq == 200:
        raw = raw.replace("cos(2*3.14*50*tx[0])", "cos(2*3.14*200*tx[0])")
    dt = .00125 * 50 / freq
    raw = re.sub(r"BDF Order\s*=\s*\d+", "BDF Order = 2", raw)
    raw = re.sub(r"Timestep Sizes\(1\)\s*=\s*Real\s+[0-9.eE+-]+",
                 f"Timestep Sizes(1) = Real {dt:.9g}", raw)
    raw = re.sub(r"TimeStep Intervals\(1\)\s*=\s*\d+",
                 "TimeStep Intervals(1) = 64", raw)
    raw = re.sub(r"Output Intervals\(1\)\s*=\s*\d+",
                 "Output Intervals(1) = 8", raw)
    raw = raw.replace('Mesh DB "." "TEAM7"', 'Mesh DB ".." "TEAM7"')
    env = os.environ.copy()
    env["ELMER_HOME"] = str(exe.parent.parent)
    env["PATH"] = str(exe.parent) + os.pathsep + env.get("PATH", "")
    for name, _, _ in CASES:
        folder = out / name / f"f{freq}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "res").mkdir(exist_ok=True)
        (folder / "TEAM7_case.sif").write_text(raw)
        print(f"START {name} {freq}Hz", flush=True)
        start = time.perf_counter()
        with (folder / "solve.log").open("w") as log:
            p = subprocess.run([str(exe), "TEAM7_case.sif"], cwd=folder, env=env,
                               stdout=log, stderr=subprocess.STDOUT)
        print(f"END {name} exit={p.returncode} seconds={time.perf_counter()-start:.2f}",
              flush=True)
        if p.returncode:
            raise RuntimeError(f"Elmer failed: {folder / 'solve.log'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--geo", type=Path)
    ap.add_argument("--sif", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs/team7_hr"))
    ap.add_argument("--elmerbin", type=Path, required=True)
    ap.add_argument("--meshes", action="store_true")
    ap.add_argument("--solve", action="store_true")
    ap.add_argument("--freq", type=int, choices=(50, 200), default=50)
    args = ap.parse_args()
    if args.meshes:
        if args.geo is None:
            ap.error("--meshes requires --geo")
        mesh(args.geo, args.out, args.elmerbin / "ElmerGrid.exe")
    if args.solve:
        if args.sif is None:
            ap.error("--solve requires --sif")
        solve(args.sif, args.out, args.elmerbin / "ElmerSolver.exe", args.freq)


if __name__ == "__main__":
    main()



