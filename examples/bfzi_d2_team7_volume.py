"""Compare TEAM-7 3-D field observables on fixed physical integration grids.

Uses Elmer's exported VTU point field reconstruction; no experiment files.
The VTU B/J variables are nodal output of elemental edge-derived fields,
so these numbers are reconstruction-dependent and not a certification of
the raw Whitney-element curl field.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


BOXES = {
    "gap_air": ((.15, .26), (.05, .17), (.027, .043)),
    "upper_air": ((.10, .25), (.06, .20), (.17, .23)),
    "plate_bulk": ((.16, .26), (.045, .175), (.004, .014)),
}
SERIES = {
    "h": ("h15", "h10", "h075"),
    "R": ("r15", "h10", "r30"),
    "air": ("a40", "h10", "a25"),
    "probe": ("probe10", "probe120", "probe240"),
    "gap": ("h10", "gap12", "gap08", "gap06"),
}


def pts(bounds, n=11):
    x, y, z = (np.linspace(*v, n) for v in bounds)
    a, b, c = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([a.ravel(), b.ravel(), c.ravel()])


def fields(path, points):
    mesh = pv.read(path)
    if mesh.n_cells == 0:
        raise ValueError(f"Empty VTU {path}")
    wanted = ("magnetic flux density e", "current density e")
    if any(n not in mesh.point_data for n in wanted):
        raise ValueError(f"Missing recovered B/J data in {path}")
    q = pv.PolyData(points).sample(mesh, locator="static_cell")
    ok = np.asarray(q["vtkValidPointMask"]).astype(bool)
    b = np.asarray(q[wanted[0]], dtype=float)
    j = np.asarray(q[wanted[1]], dtype=float)
    if b.shape != points.shape or j.shape != points.shape:
        raise ValueError("Unexpected B/J vector field layout")
    return {"B": b, "J": j, "valid": ok}


def pair(a, b, box):
    mask = a["valid"] & b["valid"]
    count = int(mask.sum())
    if count != len(mask):
        raise ValueError(f"{box}: {count}/{len(mask)} shared samples valid")
    result = {"points": count}
    for name in ("B", "J"):
        if name == "J" and box != "plate_bulk":
            continue
        u, v = a[name][mask], b[name][mask]
        d = u-v
        den = np.linalg.norm(v)
        if den <= 0:
            raise ValueError(f"Zero reference {name} in {box}")
        result[name + "_L2_rel"] = float(np.linalg.norm(d)/den)
        result[name + "_rms_a"] = float(np.sqrt(np.mean(np.sum(u*u, axis=1))))
        result[name + "_rms_b"] = float(np.sqrt(np.mean(np.sum(v*v, axis=1))))
        result[name + "_rms_rel"] = abs(result[name+"_rms_a"]-result[name+"_rms_b"])/result[name+"_rms_b"]
        result[name + "_mean_d"] = [float(x) for x in np.mean(d, axis=0)]
    return result


def analyze(root, hz, n):
    out = {}
    for name, bounds in BOXES.items():
        xyz = pts(bounds, n)
        cache = {}
        for case in dict.fromkeys(q for cases in SERIES.values() for q in cases):
            f = root / case / f"res{hz}" / "transient_t0064.vtu"
            if not f.exists():
                f = root / case / f"resene{hz}s64" / "transient_t0064.vtu"
            if f.exists():
                cache[case] = fields(f, xyz)
        studies = {}
        for axis, cases in SERIES.items():
            rows = []
            for one, two in zip(cases[:-1], cases[1:]):
                if one in cache and two in cache:
                    rows.append({"from": one, "to": two, **pair(cache[one], cache[two], name)})
            if rows:
                studies[axis] = rows
        out[name] = {"bounds_m": bounds, "sampling": "uniform 3D Cartesian point grid",
                     "studies": studies}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=11)
    args = ap.parse_args()
    data = {"kind": "3D recovered VTU B/J on fixed physical regions",
            "measurements_used": False,
            "field_origin": "VTU nodal output interpolated from elemental fields",
            "status": "OPEN", "n": args.n,
            "runs": {str(f): analyze(args.root, f, args.n) for f in (50, 200)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for hz, regions in data["runs"].items():
        for region, v in regions.items():
            print(hz, region)
            for axis, rows in v["studies"].items():
                for row in rows:
                    key = "J" if region == "plate_bulk" else "B"
                    print(axis, row["from"], row["to"],
                          "L2", f"{row[key+'_L2_rel']*100:.4f}%",
                          "RMS", f"{row[key+'_rms_rel']*100:.4f}%")
    print("OUT", args.out)


if __name__ == "__main__":
    main()

