"""Extend TEAM-7 air domain while keeping every original inner tetra fixed.

The old physical boundary Inf becomes an internal conforming interface.
Only a tetrahedral air shell and its new physical Inf boundary are added.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from examples.bfzi_d2_team7_gap import solve, raw_run

RAD = {"shell30": .30, "shell40": .40}


def make(root, name, bin_dir):
    import subprocess

    src = root/"gap06"/"TEAM7.msh"
    out = root/name
    out.mkdir(parents=True, exist_ok=True)
    lines = src.read_text(encoding="utf-8").splitlines()
    ni = lines.index("$Nodes")
    n = int(lines[ni+1])
    nl = lines[ni+2:ni+2+n]
    xyz = np.array([[float(v) for v in row.split()[1:4]] for row in nl])
    ei = lines.index("$Elements")
    ne = int(lines[ei+1])
    rows = lines[ei+2:ei+2+ne]

    old = []
    inf = []
    for row in rows:
        s = row.split()
        if int(s[1]) == 2 and int(s[3]) == 6:
            inf.append(s)
        else:
            old.append(s)
    if not inf or not np.any(xyz[:, 0] < -.19):
        raise ValueError("Expected original R=0.20m TEAM-7 Inf boundary")
    tris = np.asarray([list(map(int, s[-3:])) for s in inf], dtype=np.intp)
    verts = np.unique(tris)
    coords = xyz[verts-1]
    low, high = coords.min(axis=0), coords.max(axis=0)
    center = (low + high)/2
    # Original exterior [-.2,.494]^2 x [-.2,.394]. Enlarge each side equally.
    r = RAD[name]
    steps = int(round((r-.2)/.025))
    if steps < 1 or not np.allclose([low[0], low[1], low[2]],
                                  [-.2]*3, atol=1e-10):
        raise ValueError("Unexpected original TEAM-7 exterior")
    new = []
    maps = []
    for k in range(1, steps+1):
        scale = (high-low + 2*(r-.2)*k/steps)/(high-low)
        pos = center + (coords-center)*scale
        ids = np.arange(n+len(verts)*(k-1)+1,
                        n+len(verts)*k+1, dtype=np.intp)
        maps.append(dict(zip(verts.tolist(), ids.tolist())))
        for idx, pt in zip(ids, pos):
            new.append(f"{idx} {pt[0]:.16g} {pt[1]:.16g} {pt[2]:.16g}")
    all_xyz = np.vstack((xyz, np.array([
        [float(x) for x in row.split()[1:]] for row in new
    ])))
    # Three tetrahedra per triangular prism. Ordered vertices prescribe
    # the same diagonal on a shared quadrilateral, across all neighbors.
    sh = []
    face = np.sort(tris, axis=1)
    for k in range(steps):
        bot = face if k == 0 else np.array([
            [maps[k-1][int(v)] for v in tri] for tri in face
        ], dtype=np.intp)
        top = np.array([[maps[k][int(v)] for v in tri] for tri in face],
                       dtype=np.intp)
        a, b, c = bot.T
        u, v, w = top.T
        sh.extend((np.stack((a, b, c, w), axis=1),
                   np.stack((a, b, v, w), axis=1),
                   np.stack((a, u, v, w), axis=1)))
    tet = np.vstack(sh)
    p = all_xyz[tet-1]
    det = np.linalg.det(p[:, 1:]-p[:, :1])
    if np.any(np.abs(det) < 1e-15):
        raise ValueError("Degenerate tetrahedra in outer shell")
    neg = det < 0
    tet[neg] = tet[neg][:, [1, 0, 2, 3]]
    shell_volume = float(np.sum(np.abs(det))/6)
    outer = (high-low+2*(r-.2)).prod()
    inner = (high-low).prod()
    if not np.isclose(shell_volume, outer-inner, rtol=2e-8):
        raise ValueError(f"Non-conformal shell volume {shell_volume}, expected {outer-inner}")

    # Mesh 2.2: retain physical groups and all original tetrahedra exactly.
    surf = np.asarray([[maps[-1][int(v)] for v in tri] for tri in tris])
    elem = [f"{i} {' '.join(s[1:])}" for i, s in enumerate(old, 1)]
    k = len(elem)
    for tri in tet:
        k += 1
        elem.append(f"{k} 4 2 3 278 {' '.join(map(str,tri))}")
    for orig, tri in zip(inf, surf):
        k += 1
        elem.append(f"{k} 2 2 6 {orig[4]} {' '.join(map(str,tri))}")
    header = lines[:ni+1]
    mid = ["$EndNodes", "$Elements", str(len(elem))]
    (out/"TEAM7.msh").write_text(
        "\n".join(header+[str(len(all_xyz))]+nl+new+mid+elem+
                  ["$EndElements", ""]), encoding="utf-8")
    with (out/"grid.log").open("w") as f:
        subprocess.run([str(bin_dir/"ElmerGrid.exe"), "14", "2",
                        str(out/"TEAM7.msh"), "-names", "-out",
                        str(out/"TEAM7")], check=True, stdout=f,
                       stderr=subprocess.STDOUT)
    print(name, "original_tets_kept", sum(int(s[1])==4 for s in old),
          "shell_tets", len(tet), "total_tets", len(tet)+sum(int(s[1])==4 for s in old),
          "shell_volume_m3", shell_volume, flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--elmerbin", type=Path, required=True)
    p.add_argument("--case", choices=tuple(RAD), required=True)
    p.add_argument("--mesh", action="store_true")
    p.add_argument("--solve", action="store_true")
    p.add_argument("--raw-one", action="store_true")
    p.add_argument("--raw-64", action="store_true")
    a=p.parse_args()
    if a.mesh:
        make(a.root, a.case, a.elmerbin)
    if a.solve:
        solve(a.root, a.case, a.elmerbin, 200)
    if a.raw_one:
        raw_run(a.root, a.case, a.elmerbin, 200, 1)
    if a.raw_64:
        raw_run(a.root, a.case, a.elmerbin, 200, 64)


if __name__ == "__main__":
    main()
