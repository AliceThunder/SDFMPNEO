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


def gmsh_bfzi_two_boxes(scene, *, mesh_size_m: float,
                        near_mesh_size_m: float | None = None,
                        near_padding_m: float | None = None) -> str:
    """Gmsh OCC two independently posed insulating OBBs plus exterior cube.

    Physical material interfaces are explicitly Boolean-fragmented.
    Two independent world-axis-aligned Box background sizing fields are
    calculated from each OBB's TRUE SE(3) rotation, not from the old
    vNext superellipse geometry. They resolve each package plus adjacent
    seawater while keeping a coarser exterior. AABB envelopes are *sizing
    supports*, not a replacement for the actual oriented OBB surfaces.

    mesh_size_m is the exterior target; near_mesh_size_m defaults to
    mesh_size_m/3. near_padding_m defaults to 2*near_mesh_size_m.
    Must still independently verify h, material-interface and outer-domain
    convergence after Gmsh actually builds the tetra mesh.
    """
    far=float(mesh_size_m)
    near=far/3 if near_mesh_size_m is None else float(near_mesh_size_m)
    pad=2*near if near_padding_m is None else float(near_padding_m)
    if not all(np.isfinite(v) for v in (far, near, pad)):
        raise ValueError('Finite near/far mesh sizes and padding required')
    if not (0 < near <= far) or pad < 0:
        raise ValueError('Require 0<near_mesh_size_m<=mesh_size_m and nonnegative padding')
    center=np.asarray(scene.outer_center)
    radius=scene.outer_radius
    s=['SetFactory("OpenCASCADE");', 'Mesh.MshFileVersion = 2.2;',
       'Mesh.MeshSizeFromPoints = 0;',
       'Mesh.MeshSizeFromCurvature = 0;',
       'Mesh.MeshSizeExtendFromBoundary = 0;',
       f'Mesh.MeshSizeMin = {near/3:.17g};',
       f'Mesh.MeshSizeMax = {far:.17g};',
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
    # The true geometry remains rotated OBB surfaces, but each background
    # field uses its exact axis-aligned envelope |R|*half_extent in WORLD XYZ.
    # Box.Thickness blends near and far element sizes outside the envelope.
    for i,assembly in enumerate(scene.assemblies,1):
        h=np.asarray(assembly.enclosure.half_extents)
        extent=np.abs(assembly.pose.rotation)@h
        center=np.asarray(assembly.pose.center)
        lo=center-extent-pad
        hi=center+extent+pad
        s.extend([
            f'Field[{i}] = Box;',
            f'Field[{i}].VIn = {near:.17g};',
            f'Field[{i}].VOut = {far:.17g};',
            f'Field[{i}].XMin = {lo[0]:.17g};',
            f'Field[{i}].XMax = {hi[0]:.17g};',
            f'Field[{i}].YMin = {lo[1]:.17g};',
            f'Field[{i}].YMax = {hi[1]:.17g};',
            f'Field[{i}].ZMin = {lo[2]:.17g};',
            f'Field[{i}].ZMax = {hi[2]:.17g};',
            f'Field[{i}].Thickness = {max(pad, 2*near):.17g};',
        ])
    s.extend(['Field[3] = Min;', 'Field[3].FieldsList = {1,2};',
              'Background Field = 3;', 'Mesh 3;'])
    return '\n'.join(s)+'\n'


def bfzi_mesh_from_gmsh22(filename, scene):
    """Mesh conforming to OBB surfaces, sea outside, insulators inside.

    Requires Gmsh .geo from gmsh_bfzi_two_boxes (all physical id=1).
    Outer domain here is a truncated CUBE, not the old artificial sphere
    and not exact infinite ocean. Must perform exterior-domain convergence.
    """
    mesh=read_gmsh22(filename,{1:scene.seawater.conductivity})
    # An R-sweep must compare genuinely DIFFERENT outer domains.
    # Reject a mesh generated at a different radius instead of silently
    # claiming the larger requested boundary in the result metadata.
    center=np.asarray(scene.outer_center,dtype=float)
    radius=float(scene.outer_radius)
    lower=mesh.nodes.min(axis=0)
    upper=mesh.nodes.max(axis=0)
    tolerance=max(1e-9,1e-8*radius)
    if (not np.allclose(lower,center-radius,atol=tolerance,rtol=0)
            or not np.allclose(upper,center+radius,atol=tolerance,rtol=0)):
        raise ValueError(
            "Gmsh mesh exterior does not match scene outer radius/center; "
            "regenerate the mesh with identical --outer-radius for every R run")
    x=mesh.nodes[mesh.tetrahedra].mean(axis=1)
    inside=np.zeros(len(x),dtype=bool)
    for assembly in scene.assemblies:
        inside |= assembly.enclosure.contains(x,assembly.pose)
    return TetraMesh(mesh.nodes,mesh.tetrahedra,
                     np.where(inside,0.0,scene.seawater.conductivity))
