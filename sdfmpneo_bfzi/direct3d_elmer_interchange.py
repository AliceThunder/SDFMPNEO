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



def verify_elmer_converted_bfzi_groups(mesh, original_physical_tags,
                                       elmer_directory, element_mapping,
                                       scene, *, coordinate_tolerance_m=1e-9):
    """Validate actual ElmerGrid body/boundary IDs, never assume Gmsh IDs.

    Requires `ElmerGrid ... -names`: three named bodies and named outer
    boundary. The mapped Elmer element bodies must equal original Gmsh
    physical 1/2/3 per tetra; every outer 303 triangle must be assigned
    the named Elmer boundary ID (which need not equal Gmsh 101).
    """
    from .direct3d_mesh import _verify_exterior_surface_101
    folder=Path(elmer_directory)
    labels={"bodies":{},"boundaries":{}}
    kind=None
    for line in (folder/"mesh.names").read_text(encoding="utf-8").splitlines():
        lowered=line.lower()
        if "names for bodies" in lowered:
            kind="bodies"
        elif "names for boundaries" in lowered:
            kind="boundaries"
        elif line.strip().startswith("$") and kind:
            field=line.strip()[1:].split("=")
            if len(field)!=2:
                raise ValueError("Invalid Elmer mesh.names group assignment")
            name=field[0].strip()
            tag=int(field[1].strip())
            if name in labels[kind]:
                raise ValueError("Duplicated Elmer mesh.names group")
            labels[kind][name]=tag
    bodies=labels["bodies"]
    exterior=labels["boundaries"].get("outer_A_tangential_zero")
    required=("seawater","transmitter_insulation","receiver_insulation")
    if not all(name in bodies for name in required) or exterior is None:
        raise ValueError(
            "ElmerGrid mesh.names is missing sea/tx/rx body or outer boundary")
    actual_tags=np.asarray(original_physical_tags,dtype=int)
    if actual_tags.shape!=(len(mesh.tetrahedra),):
        raise ValueError("Missing original Gmsh physical tags per tetra")
    expected={i+1:bodies[name] for i,name in enumerate(required)}
    for row in (folder/"mesh.elements").read_text(
            encoding="utf-8").splitlines():
        if not row.strip():
            continue
        fields=row.split()
        eid=int(fields[0])
        tet=element_mapping.get(eid)
        if tet is None or int(fields[1])!=expected.get(int(actual_tags[tet])):
            raise ValueError("ElmerGrid body assignment changed BFZI materials")

    # Recover native Elmer node IDs by exact position; the ElmerGrid
    # serialization may reorder and renumber both nodes and elements.
    elmer_nodes={}
    for row in (folder/"mesh.nodes").read_text(
            encoding="utf-8").splitlines():
        if row.strip():
            fields=row.split()
            elmer_nodes[int(fields[0])]=[float(v) for v in fields[2:]]
    ids=list(elmer_nodes)
    x=np.asarray([elmer_nodes[k] for k in ids],float)
    d,indices=cKDTree(mesh.nodes).query(x)
    if np.any(d>coordinate_tolerance_m) or len(set(indices))!=len(mesh.nodes):
        raise ValueError("ElmerGrid boundary node mapping is not bijective")
    back=dict(zip(ids,map(int,indices)))
    triangles=[]
    for row in (folder/"mesh.boundary").read_text(
            encoding="utf-8").splitlines():
        if not row.strip():
            continue
        fields=row.split()
        if int(fields[1])!=exterior:
            continue
        if int(fields[4])!=303 or len(fields)!=8:
            raise ValueError("BFZI exterior Elmer boundary must be 303 triangles")
        try:
            triangles.append([back[int(n)] for n in fields[5:]])
        except KeyError as exc:
            raise ValueError("Elmer boundary refers to unknown node") from exc
    _verify_exterior_surface_101(mesh,np.asarray(triangles,dtype=int).reshape(-1,3),
                                 np.asarray(scene.outer_center,float),
                                 float(scene.outer_radius))
    return {"verified_elmer_tetrahedra":len(element_mapping),
            "verified_outer_triangles":len(triangles),
            "elmer_body_ids":{name:bodies[name] for name in required},
            "elmer_outer_boundary_id":exterior}


def compare_final_elmer_whitney_source(
        mesh, elmer_directory, elmer_edge_load_table, expected_edge_load, *,
        relative_tolerance=1e-7, absolute_tolerance_A=1e-11):
    """Compare both FINAL Elmer edge RHS columns to independently assembled J.

    Four whitespace-separated columns, one record for EVERY edge including
    zero entries:
        ElmerNodeFrom ElmerNodeTo RHS_Port1_A RHS_Port2_A
    The RHS columns must be taken AFTER all Elmer source-current fixing and
    before the field linear solve, in the raw *oriented Whitney edge basis*.
    Values are real per-unit-current weak loads, not field solutions.
    This is an explicitly exported artifact: neither ElmerGrid nor an
    unmodified ElmerSolver produces this table automatically.

    The map is by actual node XYZ and unordered edge incidence; reversal
    of the Elmer-oriented edge flips its reported RHS sign. It is therefore
    independent of arbitrary ElmerGrid global node/edge numbering.
    """
    from .direct3d_aphi import _topology
    rtol=float(relative_tolerance)
    atol=float(absolute_tolerance_A)
    if (not np.isfinite(rtol) or not np.isfinite(atol)
            or rtol<0 or atol<0):
        raise ValueError("Finite nonnegative edge-source tolerances required")
    edges,*_=_topology(mesh)
    expectation=np.asarray(expected_edge_load,dtype=complex)
    if (expectation.shape!=(2,len(edges)) or
            not np.isfinite(expectation).all() or
            np.any(np.imag(expectation)!=0)):
        raise ValueError("Expected two finite real Whitney load rows")
    folder=Path(elmer_directory)
    node_records={}
    for line in (folder/"mesh.nodes").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields=line.split()
        if len(fields)!=5:
            raise ValueError("Invalid Elmer source node record")
        eid=int(fields[0])
        if eid in node_records:
            raise ValueError("Duplicate Elmer source node ID")
        node_records[eid]=[float(a) for a in fields[2:]]
    ids=list(node_records)
    coords=np.asarray([node_records[i] for i in ids],float)
    if (len(ids)!=len(mesh.nodes) or not np.isfinite(coords).all()):
        raise ValueError("Elmer source mesh nodes do not match original")
    distance,index=cKDTree(mesh.nodes).query(coords)
    if np.any(distance>1e-9) or len(set(index))!=len(mesh.nodes):
        raise ValueError("Elmer source node mapping is not bijective")
    elmer_to_original=dict(zip(ids,map(int,index)))
    observed=np.zeros((2,len(edges)),float)
    seen=set()
    for line in Path(elmer_edge_load_table).read_text(
            encoding="utf-8").splitlines():
        line=line.strip()
        if not line or line.startswith("#"):
            continue
        fields=line.split()
        if len(fields)!=4:
            raise ValueError("Expected two Elmer endpoint IDs and two source loads")
        try:
            a=elmer_to_original[int(fields[0])]
            b=elmer_to_original[int(fields[1])]
        except KeyError as exc:
            raise ValueError("Elmer source edge has an unknown node") from exc
        key=tuple(sorted((a,b)))
        if a==b or key not in edges:
            raise ValueError("Elmer source references a non-tetrahedral edge")
        e=edges[key]
        if e in seen:
            raise ValueError("Duplicate Elmer oriented edge RHS record")
        values=np.asarray([float(fields[2]),float(fields[3])])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite Elmer final source RHS")
        observed[:,e]=values if a<b else -values
        seen.add(e)
    if len(seen)!=len(edges):
        raise ValueError("Final Elmer RHS table must contain every mesh edge")
    difference=np.abs(observed-expectation.real)
    scale=np.max(np.abs(expectation),axis=1)
    threshold=atol+rtol*scale
    per_port=np.max(difference,axis=1)
    return {
        "matched":bool(np.all(per_port<=threshold)),
        "maximum_absolute_difference_A":per_port.tolist(),
        "maximum_relative_to_peak_source":(
            per_port/np.maximum(scale,1e-300)).tolist(),
        "threshold_A":threshold.tolist(),
        "verified_edge_count":len(edges),
    }
