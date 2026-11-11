"""Probe DG-exported elemental B/J against Elmer's native integrated observables.

DG VTU uses disconnected tetra vertices; its linear vertex interpolation is not
the underlying Whitney H(curl) Gauss-point field. This is a diagnostic only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import meshio
import numpy as np

MU0 = 1.256e-6
SIGMA = 3.526e7


def mean_sq(v):
    """Exact tetra mean of |v|^2 for linear vertex interpolation."""
    if v.ndim != 3 or v.shape[1:] != (4, 3):
        raise ValueError("Expected (elements, four vertices, vector three)")
    return ((v*v).sum(axis=(1, 2)) + (v.sum(axis=1)**2).sum(axis=1))/20


def extract(path):
    m = meshio.read(path)
    if len(m.cells) != 1 or m.cells[0].type != "tetra":
        raise ValueError("Expected tetra-only mesh")
    t = m.cells[0].data
    if len(np.unique(t)) != 4*len(t):
        raise ValueError("VTU field is not element-discontinuous (shared vertices)")
    coords = m.points[t]
    mat = np.stack([coords[:,1]-coords[:,0],
                    coords[:,2]-coords[:,0],
                    coords[:,3]-coords[:,0]], axis=-1)
    vol = np.abs(np.linalg.det(mat))/6
    ids = m.cell_data["GeometryIds"][0]
    b = m.point_data["magnetic flux density e"][t]
    j = m.point_data["current density e"][t]
    e = float(np.sum(vol*mean_sq(b))/(2*MU0))
    p = float(np.sum(vol[ids==3]*mean_sq(j)[ids==3])/SIGMA)
    return {
        "tets": len(t), "energy_dg_J": e, "plate_power_dg_W": p,
        "regions_m3": {str(k):float(vol[ids==k].sum()) for k in (1,2,3)},
        "note": "DG vertex field interpolation, not raw Whitney curl Gauss integration",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--vtu", type=Path, required=True)
    ap.add_argument("--energy", type=float)
    ap.add_argument("--power", type=float)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    data = extract(a.vtu)
    for val, key, dkey in ((a.energy, "energy_dg_J", "energy_rel"),
                           (a.power, "plate_power_dg_W", "power_rel")):
        if val is not None:
            data[dkey] = abs(data[key]-val)/abs(val)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
