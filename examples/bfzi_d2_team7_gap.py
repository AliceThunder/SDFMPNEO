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

GAPS = {"gap12": .012, "gap08": .008, "gap06": .006, "gap05": .005, "gap04": .004}
RADII = {"rgap15": .15, "rgap30": .30, "rgap40": .40}
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
    h = GAPS.get(name, .006)
    base = root / "h10" / "TEAM7.geo"
    out = root / name
    out.mkdir(parents=True, exist_ok=True)
    geo = base.read_text(encoding="utf-8-sig")
    if "AirVolume[0]" not in geo or "lc_air = 30*mm;" not in geo:
        raise ValueError("Original CSC TEAM-7 geo required")
    if name in RADII:
        if "boxt = 200*mm;" not in geo or "394*mm+boxt" not in geo:
            raise ValueError("Expected fixed CSC TEAM-7 air-box geometry")
        geo = geo.replace("boxt = 200*mm;", f"boxt = {RADII[name]*1000:g}*mm;")
        geo = geo.replace("394*mm+boxt", "194*mm+2*boxt")
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
    s = (root/"h10"/f"case_ene{freq}s64.sif").read_text(encoding="utf-8-sig")
    dt = .00125 * 50/freq
    if (f"cos(2*3.14*{freq}*tx[0])" not in s
            or "TimeStep Intervals(1) = 64" not in s
            or "BDF Order = 2" not in s
            or f"Timestep Sizes(1) = Real {dt}" not in s):
        raise ValueError("Unchanged CSC source / four-period BDF2 input required")
    (out/f"case_ene{freq}s64.sif").write_text(s, encoding="utf-8")
    (out/f"resene{freq}s64").mkdir(exist_ok=True)
    env = os.environ.copy()
    env["ELMER_HOME"] = str(bin_dir.parent)
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    start = time.perf_counter()
    with (out/f"solve_{freq}.log").open("w") as f:
        res = subprocess.run([str(bin_dir/"ElmerSolver.exe"),
                              f"case_ene{freq}s64.sif"], cwd=out, env=env,
                             stdout=f, stderr=subprocess.STDOUT)
    print(f"{name} {freq}Hz: exit={res.returncode}, elapsed={time.perf_counter()-start:.1f}s",
          flush=True)
    if res.returncode:
        raise RuntimeError(f"Elmer failed: {out/f'solve_{freq}.log'}")




def raw_run(root, name, bin_dir, freq=200, steps=1):
    """Save original Elmer AV edge DOFs at first step or fourth-period end."""
    out = root/name
    if not (out/"TEAM7"/"mesh.elements").is_file():
        raise ValueError("Original Elmer tetra mesh is required")
    s=(root/"h10"/f"case_ene{freq}s64.sif").read_text(encoding="utf-8-sig")
    if ("TimeStep Intervals(1) = 64" not in s or
            'Results Directory "resene'+str(freq)+'s64"' not in s):
        raise ValueError("Expected original TEAM-7 64-step source")
    if steps not in (1, 64):
        raise ValueError("Expected one or 64 time steps")
    if steps == 1:
        s=s.replace("TimeStep Intervals(1) = 64","TimeStep Intervals(1) = 1")
        s=s.replace("Output Intervals(1) = 64","Output Intervals(1) = 1")
    tag=f"raw{steps}"
    s=s.replace(f'Results Directory "resene{freq}s64"',
                f'Results Directory "res{tag}"')
    s=s.replace("Simulation\n", f'Simulation\n  Output File = "{tag}.result"\n', 1)
    (out/f"case_{tag}.sif").write_text(s, encoding="utf-8")
    (out/f"res{tag}").mkdir(exist_ok=True)
    env=os.environ.copy()
    env["ELMER_HOME"]=str(bin_dir.parent)
    env["PATH"]=str(bin_dir)+os.pathsep+env.get("PATH","")
    with (out/f"{tag}.log").open("w") as log:
        res=subprocess.run([str(bin_dir/"ElmerSolver.exe"),f"case_{tag}.sif"],
                           cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
    result=out/"TEAM7"/f"{tag}.result"
    if res.returncode != 0 or not result.is_file():
        raise RuntimeError(f"Elmer native edge run failed for {name}")
    print(f"{tag} {name} exit=0 result_bytes={result.stat().st_size}",flush=True)

def compare(root, freq, axis="h"):
    """Original elemental SaveLine B/J at fourth-period end on fixed 101 points."""
    from examples.bfzi_d2_team7_conv import load, cmp
    from examples.bfzi_d2_team7_integrals import read_phys
    cases = (("h10", "gap12", "gap08", "gap06", "gap05", "gap04")
             if axis == "h" else ("rgap15", "gap06", "rgap30", "rgap40"))
    vals = {n: load(root/n/f"resene{freq}s64"/"f.dat") for n in cases}
    native = {n: read_phys(root/n/f"resene{freq}s64"/"phys.dat") for n in cases}
    if any(v["rows"] != 64 for v in native.values()):
        raise ValueError("Incomplete four-period native Elmer energy/power")
    energy = [
        {"from": a, "to": b,
         **{q+"_rel": abs(native[a][q]-native[b][q])/abs(native[b][q])
            for q in ("em_energy_J", "eddy_power_W")}}
        for a,b in zip(cases[:-1], cases[1:])
    ]
    return {
        "frequency_hz": freq, "steps": 64, "phase_time_s": 4/freq,
        "axis": axis,
        "source": "same CSC TEAM-7 source, solids, and 6mm gap mesh target",
        "measurement_used": False,
        "cases": list(cases), "native": native, "energy_pairs": energy,
        "pairs": [{"from": a, "to": b, **cmp(vals[a], vals[b])}
                  for a, b in zip(cases[:-1], cases[1:])],
        "status": "D2 OPEN",
    }

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--elmerbin", type=Path, required=True)
    p.add_argument("--case", choices=tuple(GAPS)+tuple(RADII), required=True)
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--solve", action="store_true")
    p.add_argument("--raw-one", action="store_true")
    p.add_argument("--raw-64", action="store_true")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--axis", choices=("h", "R"), default="h")
    p.add_argument("--out", type=Path)
    p.add_argument("--freq", type=int, choices=(50, 200), default=200)
    a = p.parse_args()
    if a.mesh:
        make(a.root, a.case, a.elmerbin)
    if a.solve:
        solve(a.root, a.case, a.elmerbin, a.freq)
    if a.raw_one:
        raw_run(a.root, a.case, a.elmerbin, a.freq, 1)
    if a.raw_64:
        raw_run(a.root, a.case, a.elmerbin, a.freq, 64)
    if a.compare:
        import json
        report = compare(a.root, a.freq, a.axis)
        out = json.dumps(report, indent=2)
        if a.out:
            a.out.parent.mkdir(parents=True, exist_ok=True)
            a.out.write_text(out, encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
