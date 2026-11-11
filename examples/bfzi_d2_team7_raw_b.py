"""Recover original TEAM-7 Whitney curl B from Elmer's native AV result.

The raw AV edge unknowns are read from the ASCII Elmer .result file.
No nodal smoothing, DG recovery or exported VTU B is used.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from sdfmpneo_bfzi.direct3d_aphi import MU0

# The order in Elmer MeshBasics.F90::FindMeshEdges3D (tetrahedra).
EL_EDGES = np.array([[0, 1], [1, 2], [2, 0],
                     [0, 3], [1, 3], [2, 3]], dtype=np.intp)


def edge_ids(tet):
    """Elmer's first-occurrence edge order and min-to-max orientation."""
    pair = tet[:, EL_EDGES].reshape(-1, 2)
    _, first, inv = np.unique(np.sort(pair, axis=1), axis=0,
                               return_index=True, return_inverse=True)
    order = np.argsort(first)
    number = np.empty(len(first), dtype=np.intp)
    number[order] = np.arange(len(first))
    idx = number[inv].reshape(len(tet), 6)
    sign = np.where(pair[:, 0] < pair[:, 1], 1., -1.).reshape(len(tet), 6)
    return idx, sign, len(first)


def read_av(path, n):
    """Read original global-DOF AV values; Perm is solver reordering metadata."""
    last = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() != "a":
                continue
            h = next(f).split()
            if len(h) != 3 or h[0] != "Perm:" or int(h[1]) != n:
                raise ValueError("Unexpected Elmer AV permutation header")
            for _ in range(n):
                next(f)
            last = np.fromiter((float(next(f)) for _ in range(n)),
                               dtype=float, count=n)
    if last is None:
        raise ValueError("Elmer result has no AV variable 'a'")
    return last


def calc(mesh, result, mu0=MU0):
    if not np.isfinite(mu0) or mu0 <= 0:
        raise ValueError("Expected positive vacuum permeability")
    node = np.loadtxt(mesh/"mesh.nodes")
    elem = np.loadtxt(mesh/"mesh.elements", dtype=np.intp)
    if not np.array_equal(node[:, 0].astype(np.intp), np.arange(1, len(node)+1)):
        raise ValueError("Elmer mesh nodes are not sequentially numbered")
    if not np.all(elem[:, 2] == 504):
        raise ValueError("Expected 4-node tetrahedral Elmer mesh")
    tet = elem[:, 3:7]-1
    idx, sign, ned = edge_ids(tet)
    av = read_av(result, len(node)+ned)
    xyz = node[tet, 2:5]
    mat = (xyz[:, 1:]-xyz[:, 0, None, :]).transpose(0, 2, 1)
    det = np.linalg.det(mat)
    if np.any(np.abs(det) < 1e-16):
        raise ValueError("Degenerate tetrahedron")
    grad = np.empty((len(tet), 4, 3))
    grad[:, 1:] = np.linalg.inv(mat)
    grad[:, 0] = -grad[:, 1:].sum(axis=1)
    rot = 2*np.stack([np.cross(grad[:, i], grad[:, j])
                      for i, j in EL_EDGES], axis=1)
    b = np.einsum("te,tec->tc", av[len(node):][idx]*sign, rot,
                  optimize=True)
    volume = np.abs(det)/6
    energy = float(np.dot(volume, np.sum(b*b, axis=1))/(2*mu0))
    return {"nodes": len(node), "tetrahedra": len(tet),
            "edge_dofs": ned, "av_dofs": len(av),
            "magnetic_energy_J": energy,
            "mu0_H_per_m": float(mu0),
            "B_rms_T": float(np.sqrt(np.dot(volume, np.sum(b*b, axis=1))/volume.sum()))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--phys", type=Path)
    ap.add_argument("--sif", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    mu0 = MU0
    if a.sif:
        sif = a.sif.read_text(encoding="utf-8-sig")
        m = re.search(r"^\s*Permeability of Vacuum\s*=\s*([\d.eE+-]+)",
                      sif, flags=re.I | re.M)
        if not m:
            raise ValueError("Missing Permeability of Vacuum in Elmer SIF")
        mu0 = float(m.group(1))
    data = calc(a.mesh, a.result, mu0=mu0)
    if a.phys:
        native = float(np.atleast_2d(np.loadtxt(a.phys))[-1, 1])
        data["native_energy_J"] = native
        data["energy_rel_diff"] = abs(data["magnetic_energy_J"]-native)/abs(native)
    data["field_source"] = "Elmer native AV global edge DOFs / original Whitney curl"
    data["measurement_used"] = False
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
