"""TEAM-7 air-gap 3-D h-refinement, original CSC solids and fixed outer domain.

The Box/Restrict field changes air tetrahedra near the physical gap, rather
than embedded probe-line segmentation. Run the original Elmer BDF2 case.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time

GAPS = {"gap12": .012, "gap08": .008}
FIELD = """
Field[90] = Box;
Field[90].VIn = {h:.9g};
Field[90].VOut = 0.030;
Field[90].XMin = -0.02;
Field[90].XMax = 0.314;
Field[90].YMin = -0.02;
Field[90].YMax = 0.314;
Field[90].ZMin = 0.020;
Field[90].ZMax = 0.044;
Field[90].Thickness = 0.002;
Field[91] = Restrict;
Field[91].InField = 90;
Field[91].VolumesList = {{AirVolume[0]}};
Background Field = 91;
"""


def make(root, name, bin_dir):
    import gmsh
    h = GAPS[name]
    base = root / "h10" / "TEAM7.geo"
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    geo = base.read_text(encoding="utf-8-sig")
    if "AirVolume[0]" not in geo or "lc_air = 30*mm;" not in geo:
        raise ValueError("Original CSC TEAM-7 geo required")
    (out/"TEAM7.geo").write_text(geo + FIELD.format(h=h), encoding="utf-8")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(out/"TEAM7.geo"))
        gmsh.model.mesh.generate(3)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.write(str(out/"TEAM7.msh"))
        tetra = sum(len(a) for a in gmsh.model.mesh.getElements(3)[1])
    finally:
        gmsh.finalize()
    with (out/"grid.log").open("w") as f:
        subprocess.run([str(bin_dir/"ElmerGrid.exe"), "14", "2",
                        str(out/"TEAM7.msh"), "-names", "-out",
                        str(out/"TEAM7")], stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"{name}: {tetra} tetrahedra, air gap h={1000*h:g} mm", flush=True)


def solve(root, name, bin_dir, freq):
    if freq not in (50, 200):
        raise ValueError("Use TEAM-7 50 or 200 Hz")
    out = root / name
    if not (out/"TEAM7"/"mesh.elements").exists():
        raise ValueError("Generate Elmer mesh first")
    s = (root/"h10"/f"TEAM7_{freq}.sif").read_text(encoding="utf-8-sig")
    dt = .00125 * 50/freq
    if (f"cos(2*3.14*{freq}*tx[0])" not in s
            or "TimeStep Intervals(1) = 64" not in s
            or "BDF Order = 2" not in s
            or f"Timestep Sizes(1) = Real {dt}" not in s):
        raise ValueError("Unchanged CSC source / four-period BDF2 input required")
    (out/f"TEAM7_{freq}.sif").write_text(s, encoding="utf-8")
    (out/f"res{freq}").mkdir(exist_ok=True)
    env = os.environ.copy()
    env["ELMER_HOME"] = str(bin_dir.parent)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    start = time.perf_counter()
    with (out/f"solve_{freq}.log").open("w") as f:
        res = subprocess.run([str(bin_dir/"ElmerSolver.exe"),
                              f"TEAM7_{freq}.sif"], cwd=out, env=env,
                             stdout=f, stderr=subprocess.STDOUT)
    print(f"{name} {freq}Hz: exit={res.returncode}, elapsed={time.perf_counter()-start:.1f}s",
          flush=True)
    if res.returncode:
        raise RuntimeError(f"Elmer failed: {out/f'solve_{freq}.log'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--elmerbin", type=Path, required=True)
    p.add_argument("--case", choices=tuple(GAPS), required=True)
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--solve", action="store_true")
    p.add_argument("--freq", type=int, choices=(50, 200), default=200)
    a = p.parse_args()
    if a.mesh:
        make(a.root, a.case, a.elmerbin)
    if a.solve:
        solve(a.root, a.case, a.elmerbin, a.freq)


if __name__ == "__main__":
    main()
