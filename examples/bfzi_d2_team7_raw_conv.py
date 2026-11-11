"""Compare original Elmer Whitney curl B directly on matched 3-D h/R samples.

This uses native one-step AV edge DOFs, no Elmer recovered B, no nodal
smoothing or DG export. Sampling a constant tetra-cell field is exact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv

from examples.bfzi_d2_team7_raw_b import raw_field
from examples.bfzi_d2_team7_volume import BOXES, pts

SERIES = {
    "h": ("h10", "gap06", "gap05", "gap04", "gap03", "gap025"),
    "R": ("gap06", "rgap30", "rgap40"),
    "Rfixed": ("gap06", "shell30", "shell40"),
}
MU = 1.256e-6


def probe(xyz, tet, b, points):
    cells = np.column_stack((np.full(len(tet), 4, dtype=np.intp), tet)).ravel()
    types = np.full(len(tet), int(pv.CellType.TETRA), dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, types, xyz)
    grid.cell_data["Braw"] = b
    res = pv.PolyData(points).sample(grid, locator="static_cell")
    good = np.asarray(res["vtkValidPointMask"], dtype=bool)
    if not np.all(good) or "Braw" not in res.point_data:
        raise ValueError(f"Raw Whitney B cell sampling failed: {good.sum()}/{len(points)}")
    return np.asarray(res["Braw"], dtype=float)


def run(root, n=11, steps=1):
    positions = {name: pts(bounds, n) for name, bounds in BOXES.items()}
    if steps not in (1, 64):
        raise ValueError("Expected 1 or 64 steps")
    series = (SERIES if steps == 1 else {
        "h": ("h10", "gap06", "gap04", "gap03", "gap025"),
        "R": ("gap06", "rgap30"),
        "Rfixed": ("gap06", "shell30", "shell40"),
    })
    cases = tuple(dict.fromkeys(v for vs in series.values() for v in vs))
    sample, energy, power = {}, {}, {}
    for name in cases:
        directory = root/name
        mesh = directory/"TEAM7"
        res = mesh/(f"edge{steps}.result" if name == "h10" and steps==64
                    else "edge.result" if name == "h10"
                    else f"raw{steps}.result")
        xyz, tet, b, vol, _, _ = raw_field(mesh, res)
        energy[name] = float(np.dot(vol, np.sum(b*b, axis=1))/(2*MU))
        phys = directory/(f"res_edge{steps}" if name == "h10" else f"resraw{steps}")/"phys.dat"
        scalars = np.atleast_2d(np.loadtxt(phys))[-1]
        native = float(scalars[1])
        power[name] = float(scalars[0])
        if abs(energy[name]-native)/abs(native) > 1e-7:
            raise ValueError(f"Original Whitney B energy differs from native: {name}")
        sample[name] = {region: probe(xyz, tet, b, pos)
                        for region, pos in positions.items()}
        print(name, "raw_energy_J", energy[name], flush=True)
    out = {
        "frequency_hz": 200,
        "time_s": steps*0.0003125,
        "steps": steps,
        "origin": "Original Elmer A edge DOFs / Whitney tetra curl, raw cell sampling",
        "material_mu0": MU,
        "measurements_used": False,
        "n_axis": n,
        "energy_J": energy,
        "native_power_W": power,
        "native_integral_pairs": {
            axis: [{"from": a, "to": b,
                    "energy_rel": float(abs(energy[a]-energy[b])/abs(energy[b])),
                    "power_rel": float(abs(power[a]-power[b])/abs(power[b]))}
                   for a, b in zip(names[:-1], names[1:])]
            for axis, names in series.items()
        },
        "regions": {},
        "status": "D2 OPEN",
    }
    for region in positions:
        studies = {}
        for axis, names in series.items():
            rows = []
            for left, right in zip(names[:-1], names[1:]):
                x, y = sample[left][region], sample[right][region]
                den = float(np.linalg.norm(y))
                if den == 0:
                    raise ValueError("Zero B reference in " + region)
                rows.append({
                    "from": left, "to": right, "count": len(x),
                    "B_L2_rel": float(np.linalg.norm(x-y)/den),
                    "B_rms_rel": float(abs(np.linalg.norm(x)-den)/den),
                    "B_mean_rel": float(np.linalg.norm(np.mean(x-y,axis=0)) /
                                        np.sqrt(np.mean(np.sum(y*y,axis=1)))),
                })
            studies[axis] = rows
        out["regions"][region] = {"bounds_m": BOXES[region],
                                   "studies": studies}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=11)
    p.add_argument("--steps", type=int, choices=(1,64), default=1)
    a=p.parse_args()
    result=run(a.root, a.n, a.steps)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for region, data in result["regions"].items():
        for axis, rows in data["studies"].items():
            for x in rows:
                print(region, axis, x["from"], x["to"],
                      f"{x['B_L2_rel']*100:.4f}%", flush=True)


if __name__ == "__main__":
    main()
