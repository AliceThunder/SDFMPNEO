"""3D mesh interchange for TEAM-7 and real BFZI dual oriented boxes.

Gmsh is an OPTIONAL local CLI; this module never invokes GitHub Actions.
The D2 A-phi solver consumes tagged tetra meshes, not G1 current bases.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .direct3d_aphi import TetraMesh


def read_gmsh22(filename, conductivity_by_tag):
    """Read ASCII Gmsh MSH 2.2 linear tetra, sigma by physical volume id.

    For a third-party TEAM-7 mesh, use the verified plate region physical id.
    All other domains remain insulating. Higher-order tetra are rejected by
    having no tetra to read; avoid accidentally assigning conductivity to air.
    """
    if not isinstance(conductivity_by_tag, dict):
        raise ValueError('Pass physical-material conductivity map')
    lines=iter(Path(filename).read_text(encoding='utf-8').splitlines())
    node_ids={}; nodes=[]; tets=[]; sigma=[]; found=False
    for s in lines:
        if s=='$MeshFormat':
            fmt=next(lines).split()
            if len(fmt)<3 or fmt[0]!='2.2' or fmt[1]!='0':
                raise ValueError('Only ASCII Gmsh MSH 2.2 supported')
            found=True
        elif s=='$Nodes':
            for _ in range(int(next(lines))):
                row=next(lines).split()
                node_ids[int(row[0])]=len(nodes)
                nodes.append([float(x) for x in row[1:4]])
        elif s=='$Elements':
            for _ in range(int(next(lines))):
                row=list(map(int,next(lines).split()))
                if row[1]!=4:continue
                ntag=row[2]
                if ntag<1:
                    raise ValueError('Every linear tetra needs a physical region tag')
                tag=row[3]
                tets.append([node_ids[x] for x in row[3+ntag:3+ntag+4]])
                sig=float(conductivity_by_tag.get(tag,0.0))
                if not np.isfinite(sig) or sig<0:
                    raise ValueError('Invalid conductivity')
                sigma.append(sig)
    if not found or not nodes or not tets:
        raise ValueError('Missing mesh 2.2 header, nodes or linear tetrahedra')
    return TetraMesh(np.array(nodes),np.array(tets),np.array(sigma))


def gmsh_bfzi_two_boxes(scene, *, mesh_size_m: float) -> str:
    """OCC fragmenting full outer cube + both independently rotated OBBs.

    Input is existing BfziScene (validated disjoint enclosures). Classify the
    resulting conforming tetra by centroid after import with
    bfzi_mesh_from_gmsh22. Gmsh is run locally, not from this function.
    """
    if not np.isfinite(mesh_size_m) or mesh_size_m<=0:
        raise ValueError('mesh_size_m must be positive')
    center=np.asarray(scene.outer_center)
    radius=scene.outer_radius
    s=['SetFactory("OpenCASCADE");', 'Mesh.MshFileVersion = 2.2;',
       f'Mesh.MeshSizeMin = {mesh_size_m/3:.17g};',
       f'Mesh.MeshSizeMax = {mesh_size_m:.17g};',
       f'Box(1) = {{{center[0]-radius:.17g}, {center[1]-radius:.17g}, {center[2]-radius:.17g}, {2*radius:.17g}, {2*radius:.17g}, {2*radius:.17g}}};']
    for i,assembly in enumerate(scene.assemblies,2):
        h=np.asarray(assembly.enclosure.half_extents)
        t=assembly.pose.center
        rotvec=Rotation.from_matrix(assembly.pose.rotation).as_rotvec()
        size=2*h
        s.append(f'Box({i}) = {{{-h[0]:.17g}, {-h[1]:.17g}, {-h[2]:.17g}, {size[0]:.17g}, {size[1]:.17g}, {size[2]:.17g}}};')
        angle=np.linalg.norm(rotvec)
        if angle>1e-12:
            axis=rotvec/angle
            s.append(f'Rotate {{{{{axis[0]:.17g},{axis[1]:.17g},{axis[2]:.17g}}},{{0,0,0}},{angle:.17g}}} {{ Volume{{{i}}}; }}')
        s.append(f'Translate {{{t[0]:.17g},{t[1]:.17g},{t[2]:.17g}}} {{ Volume{{{i}}}; }}')
    s.append('v() = BooleanFragments{ Volume{1}; Delete; }{ Volume{2,3}; Delete; };')
    s.append('Physical Volume("all",1) = {v()};')
    s.append('Mesh 3;')
    return '\n'.join(s)+'\n'


def bfzi_mesh_from_gmsh22(filename, scene):
    """Mesh conforming to OBB surfaces, sea outside, insulators inside.

    Requires Gmsh .geo from gmsh_bfzi_two_boxes (all physical id=1).
    Outer domain here is a truncated CUBE, not the old artificial sphere
    and not exact infinite ocean. Must perform exterior-domain convergence.
    """
    mesh=read_gmsh22(filename,{1:scene.seawater.conductivity})
    x=mesh.nodes[mesh.tetrahedra].mean(axis=1)
    inside=np.zeros(len(x),dtype=bool)
    for assembly in scene.assemblies:
        inside |= assembly.enclosure.contains(x,assembly.pose)
    return TetraMesh(mesh.nodes,mesh.tetrahedra,
                     np.where(inside,0.0,scene.seawater.conductivity))
