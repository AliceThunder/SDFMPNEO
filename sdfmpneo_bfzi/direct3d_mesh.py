"""3D mesh interchange for TEAM-7 and real BFZI dual oriented boxes.

Gmsh is an OPTIONAL local CLI; this module never invokes GitHub Actions.
The D2 A-phi solver consumes tagged tetra meshes, not G1 current bases.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .direct3d_aphi import TetraMesh


def read_gmsh22(filename, conductivity_by_tag, *, return_physical_tags=False,
                return_outer_faces=False):
    """Read ASCII Gmsh MSH 2.2 linear tetra, sigma by physical volume id.

    For a third-party TEAM-7 mesh, use the verified plate region physical id.
    All other domains remain insulating for legacy TEAM-7 import.
    Set return_physical_tags=True for canonical BFZI material verification;
    the resulting second value contains the physical volume ID of each tetra.
    Higher-order tetra are not interpreted as linear tetra.
    When both return flags are set, also return the node indices of
    physical surface 101 triangles for strict Elmer boundary verification.
    """
    if not isinstance(conductivity_by_tag, dict):
        raise ValueError('Pass physical-material conductivity map')
    if return_outer_faces and not return_physical_tags:
        raise ValueError('return_outer_faces requires return_physical_tags')
    lines=iter(Path(filename).read_text(encoding='utf-8').splitlines())
    node_ids={}; nodes=[]; tets=[]; sigma=[]; physical_tags=[]
    outer_triangles=[]; found=False
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
                if row[1]==2 and return_outer_faces:
                    ntag=row[2]
                    if ntag>=1 and row[3]==101:
                        outer_triangles.append(
                            [node_ids[x] for x in row[3+ntag:6+ntag]])
                    continue
                if row[1]!=4:continue
                ntag=row[2]
                if ntag<1:
                    raise ValueError('Every linear tetra needs a physical region tag')
                tag=row[3]
                physical_tags.append(tag)
                tets.append([node_ids[x] for x in row[3+ntag:3+ntag+4]])
                sig=float(conductivity_by_tag.get(tag,0.0))
                if not np.isfinite(sig) or sig<0:
                    raise ValueError('Invalid conductivity')
                sigma.append(sig)
    if not found or not nodes or not tets:
        raise ValueError('Missing mesh 2.2 header, nodes or linear tetrahedra')
    mesh=TetraMesh(np.array(nodes),np.array(tets),np.array(sigma))
    if return_physical_tags:
        tags=np.asarray(physical_tags,dtype=int)
        if return_outer_faces:
            return mesh,tags,np.asarray(outer_triangles,dtype=int).reshape(-1,3)
        return mesh,tags
    return mesh


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


def _verify_exterior_surface_101(mesh, triangles, center, radius):
    """Physical 101 must be exactly the exterior tetrahedral facets."""
    if not len(triangles):
        raise ValueError("MSH2 missing physical surface 101 on outer boundary")
    tets=mesh.tetrahedra
    all_faces=np.concatenate((
        tets[:,(0,1,2)],tets[:,(0,1,3)],
        tets[:,(0,2,3)],tets[:,(1,2,3)]),axis=0)
    unique,counts=np.unique(np.sort(all_faces,axis=1),
                            axis=0,return_counts=True)
    if np.any(counts>2):
        raise ValueError("MSH2 nonmanifold tetrahedral face incidence")
    outer=unique[counts==1]
    labelled,face_counts=np.unique(np.sort(triangles,axis=1),
                                   axis=0,return_counts=True)
    if np.any(face_counts!=1) or not np.array_equal(outer,labelled):
        raise ValueError(
            "MSH2 physical surface 101 must label every exterior tetra "
            "face exactly once and no dry/wet internal interface")
    points=mesh.nodes[outer]
    tolerance=max(1e-9,1e-8*radius)
    on_cube=np.zeros(len(outer),dtype=bool)
    for axis in range(3):
        for sign in (-1.,1.):
            on_cube |= np.all(np.abs(
                points[:,:,axis]-(center[axis]+sign*radius))
                <=tolerance,axis=1)
    if not np.all(on_cube):
        raise ValueError(
            "MSH2 exterior has faces away from finite outer cube boundary")


def bfzi_mesh_from_gmsh22(filename, scene, *, allow_legacy_single_tag=False):
    """Read true material-tagged BFZI OCC mesh and verify all three regions.

    Canonical physical volumes: 1=seawater, 2=transmitter insulation,
    3=receiver insulation. The three tags are assigned by OCC fragment's
    original-entity correspondence; material conductivity is read FROM
    THESE TAGS, not inferred from centroids. Cell centers check the CAD
    mapping only; they do not overwrite it. Every body's tetra volume
    must equal the true SE(3) box volume.

    The old .geo script has a single "all" volume tag 1. Explicitly pass
    allow_legacy_single_tag=True to classify that LEGACY diagnostic mesh
    by geometry; never silently use that mode for Elmer cross-validation.
    """
    mesh, tags, outer_faces=read_gmsh22(
        filename,{1:scene.seawater.conductivity,2:0.,3:0.},
        return_physical_tags=True,return_outer_faces=True)
    seen=set(int(i) for i in tags)
    legacy=seen=={1}
    if legacy and not allow_legacy_single_tag:
        raise ValueError(
            "Single-region legacy BFZI mesh has no physical insulation "
            "labels. Generate with --generate-msh or explicitly enable "
            "the legacy single-volume importer for diagnostics only.")
    if not legacy and seen!={1,2,3}:
        raise ValueError(
            "BFZI mesh needs exactly physical volumes 1=sea, 2=tx dry, "
            f"3=rx dry; found {sorted(seen)}")
    # An R sweep must really change the finite outer mesh.
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

    tetra_coords=mesh.nodes[mesh.tetrahedra]
    x=tetra_coords.mean(axis=1)
    tx=scene.transmitter.enclosure.contains(x,scene.transmitter.pose)
    rx=scene.receiver.enclosure.contains(x,scene.receiver.pose)
    if np.any(tx & rx):
        raise ValueError("BFZI insulating boxes overlap")
    if legacy:
        return TetraMesh(mesh.nodes,mesh.tetrahedra,
                         np.where(tx|rx,0.,scene.seawater.conductivity))
    _verify_exterior_surface_101(mesh,outer_faces,center,radius)

    expected=np.where(tx,2,np.where(rx,3,1))
    mismatch=np.flatnonzero(tags!=expected)
    if len(mismatch):
        raise ValueError(
            f"Gmsh physical material tags disagree with actual SE(3) "
            f"boxes in {len(mismatch)} tetrahedra; first cell {mismatch[0]} "
            f"has tag {tags[mismatch[0]]} but lies in region "
            f"{expected[mismatch[0]]}")
    # Material volumes have physical meaning in both solvers: a missing
    # cut/interface would otherwise silently change seawater Joule heat.
    v1=tetra_coords[:,1]-tetra_coords[:,0]
    v2=tetra_coords[:,2]-tetra_coords[:,0]
    v3=tetra_coords[:,3]-tetra_coords[:,0]
    tet_volumes=np.abs(np.einsum(
        'ij,ij->i',np.cross(v1,v2),v3))/6.
    expected_volumes={
        1:(2*radius)**3-sum(8*np.prod(a.enclosure.half_extents)
                           for a in scene.assemblies),
        2:8*np.prod(scene.transmitter.enclosure.half_extents),
        3:8*np.prod(scene.receiver.enclosure.half_extents)}
    for tag,volume in expected_volumes.items():
        actual=float(tet_volumes[tags==tag].sum())
        if not np.isclose(actual,volume,rtol=2e-7,atol=1e-10):
            raise ValueError(
                f"Gmsh physical region {tag} volume {actual:.10g} m3 "
                f"does not match BFZI geometry {volume:.10g} m3")
    return mesh
