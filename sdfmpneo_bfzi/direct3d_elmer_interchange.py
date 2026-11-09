"""ElmerGrid tetrahedron index transfer for the BFZI D2 identical-source gate.

ElmerGrid is allowed to renumber BOTH nodes and tetrahedra.  Never match
an exported unit-port current field by array position.  This reads the
native serial *text* mesh.nodes/mesh.elements and restores Gmsh tetrahedron
indices by exact connectivity after tolerance-bounded node matching.

The input table here is an explicit 7-column two-unit-port export:
    ElmerElementID J1x J1y J1z J2x J2y J2z   (A/m^2, real basis).
An actual Elmer CoilSolver/WhitneyAVSolver run must supply this table;
Elmer does not create this custom table automatically.  Exported J must
still pass discrete current conservation, 1-A flux, direction, and source
space equivalence before physical comparison with SDFMPNEO.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def elmer_to_gmsh_tetra_indices(mesh, elmer_directory, *,
                                coordinate_tolerance_m=1e-9):
    """Map Elmer element ID -> original MSH2 tetra row, rejecting remeshing."""
    folder=Path(elmer_directory)
    tol=float(coordinate_tolerance_m)
    if not np.isfinite(tol) or tol<=0:
        raise ValueError("Positive finite node coordinate tolerance required")
    node_lines=(folder/"mesh.nodes").read_text(encoding="utf-8").splitlines()
    element_lines=(folder/"mesh.elements").read_text(
        encoding="utf-8").splitlines()
    elmer_nodes={}
    for row in node_lines:
        if not row.strip():
            continue
        fields=row.split()
        if len(fields)!=5:
            raise ValueError("Elmer mesh.nodes must have ID p x y z")
        node_id=int(fields[0])
        coords=np.asarray([float(x) for x in fields[2:]],float)
        if node_id in elmer_nodes or not np.isfinite(coords).all():
            raise ValueError("Elmer node IDs must be unique and finite")
        elmer_nodes[node_id]=coords
    if len(elmer_nodes)!=len(mesh.nodes):
        raise ValueError("ElmerGrid changed the number of tetra mesh nodes")
    node_ids=list(elmer_nodes)
    xyz=np.array([elmer_nodes[n] for n in node_ids])
    distances,original=cKDTree(mesh.nodes).query(xyz)
    if np.any(distances>tol) or len(set(map(int,original)))!=len(mesh.nodes):
        raise ValueError("ElmerGrid node coordinates cannot be mapped bijectively")
    id_to_orig=dict(zip(node_ids,map(int,original)))
    original_keys={
        tuple(sorted(map(int,tet))):index
        for index,tet in enumerate(mesh.tetrahedra)}
    if len(original_keys)!=len(mesh.tetrahedra):
        raise ValueError("Original tetrahedral mesh has duplicated cells")
    mapped={}
    used=set()
    for row in element_lines:
        if not row.strip():
            continue
        fields=row.split()
        if len(fields)!=7 or int(fields[2])!=504:
            raise ValueError("Expected only four-node Elmer tetrahedral 504 elements")
        eid=int(fields[0])
        if eid in mapped:
            raise ValueError("Duplicate Elmer element ID")
        try:
            key=tuple(sorted(id_to_orig[int(x)] for x in fields[3:]))
        except KeyError as exc:
            raise ValueError("Elmer element references an unknown node") from exc
        if key not in original_keys:
            raise ValueError("Elmer tet connectivity differs from original Gmsh mesh")
        index=original_keys[key]
        if index in used:
            raise ValueError("More than one Elmer tetra maps to original cell")
        mapped[eid]=index
        used.add(index)
    if len(mapped)!=len(mesh.tetrahedra):
        raise ValueError("Elmer tetra mapping does not cover every original cell")
    return mapped


def load_two_unit_element_currents(table_path, elmer_element_to_tetra,
                                   n_tetrahedra):
    """Reorder an explicit Elmer-ID two-port elemental J table to [2,tet,3]."""
    expected=set(elmer_element_to_tetra)
    seen=set()
    currents=np.zeros((2,int(n_tetrahedra),3),dtype=float)
    for row in Path(table_path).read_text(encoding="utf-8").splitlines():
        row=row.strip()
        if not row or row.startswith("#"):
            continue
        fields=row.split()
        if len(fields)!=7:
            raise ValueError("Expected ElmerID and six real J components")
        eid=int(fields[0])
        if eid not in expected or eid in seen:
            raise ValueError("Unknown or duplicated Elmer current element ID")
        components=np.array([float(x) for x in fields[1:]],float)
        if not np.isfinite(components).all():
            raise ValueError("Non-finite unit-port impressed current")
        currents[:,elmer_element_to_tetra[eid],:]=components.reshape(2,3)
        seen.add(eid)
    if seen!=expected:
        raise ValueError("Missing Elmer unit-current data for tetrahedra")
    if np.any(np.sum(np.abs(currents),axis=(1,2))==0):
        raise ValueError("Each of the two real unit-current ports must be nonzero")
    return currents
