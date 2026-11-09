"""Physical-tagged Gmsh OCC mesh for independent D2 BFZI/Elmer comparison.

Unlike the legacy all-tagged .geo smoke mesh, this uses OCC fragment's
documented original-to-result entity correspondence: seawater=1,
transmitter dry box=2, receiver dry box=3, exterior surface=101.
No geometry classification by a posteriori tetra centroids is needed.

Gmsh is a LOCAL optional dependency. Never run GitHub Actions.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def generate_physical_bfzi_msh22(
    scene, path, *, mesh_size_m, near_mesh_size_m=None,
    near_padding_m=None,
):
    """Write MSH 2.2 tetra mesh with independent physical material regions.

    Both packages retain independent arbitrary proper 3-D poses. True
    material surfaces are OCC BooleanFragments interfaces, NOT axis-aligned
    bounding boxes. The latter only support near-wall *mesh size* fields.

    Return dictionary of physical tags and geometric region volumes;
    physical output is not certified without a Gmsh/Elmer run and h/R
    convergence. Gmsh 4.x Python API is needed on the running workstation.
    """
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError(
            "Install the Gmsh Python API to generate physically tagged "
            "SE(3) meshes. The restricted axis-aligned solver does "
            "not substitute for arbitrary independently rotated boxes."
        ) from exc

    far=float(mesh_size_m)
    near=far/3 if near_mesh_size_m is None else float(near_mesh_size_m)
    pad=2*near if near_padding_m is None else float(near_padding_m)
    if not all(np.isfinite(v) for v in (far,near,pad)) or not (
            0 < near <= far and pad >= 0):
        raise ValueError("Require finite 0 < near <= far and padding >= 0")
    output=Path(path)
    if output.suffix.lower() != ".msh":
        raise ValueError("Output must be a .msh file")
    output.parent.mkdir(parents=True,exist_ok=True)

    gmsh.initialize()
    try:
        gmsh.model.add("BFZI_D2_two_sealed_SE3_ports")
        gmsh.option.setNumber("Mesh.MshFileVersion",2.2)
        gmsh.option.setNumber("Mesh.Binary",0)
        gmsh.option.setNumber("Mesh.SaveAll",0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints",0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature",0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary",0)
        gmsh.option.setNumber("Mesh.MeshSizeMin",near/3)
        gmsh.option.setNumber("Mesh.MeshSizeMax",far)

        center=np.asarray(scene.outer_center,float)
        radius=float(scene.outer_radius)
        lo=center-radius
        sea_box=gmsh.model.occ.addBox(
            *map(float,lo),2*radius,2*radius,2*radius)
        package_tags=[]
        for assembly in scene.assemblies:
            half=np.asarray(assembly.enclosure.half_extents,float)
            t=gmsh.model.occ.addBox(
                *map(float,-half),*map(float,2*half))
            rv=Rotation.from_matrix(
                assembly.pose.rotation).as_rotvec()
            angle=float(np.linalg.norm(rv))
            if angle>1e-12:
                axis=rv/angle
                gmsh.model.occ.rotate(
                    [(3,t)],0.,0.,0.,*map(float,axis),angle)
            gmsh.model.occ.translate(
                [(3,t)],*map(float,assembly.pose.center))
            package_tags.append(t)

        fragments,origin_map=gmsh.model.occ.fragment(
            [(3,sea_box)],[(3,package_tags[0]),(3,package_tags[1])],
            removeObject=True,removeTool=True)
        gmsh.model.occ.synchronize()

        # origin_map[0] is the full outer box and includes ALL partitioned
        # fragments. origin_map[1:3] identifies each dry solid exactly;
        # taking the set difference isolates physical seawater regardless
        # of the numeric entity tags Gmsh chooses after fragmentation.
        if len(origin_map)!=3:
            raise ValueError("OCC did not return the three input volume maps")
        all_volumes={int(t) for dim,t in fragments if dim==3}
        tx={int(t) for dim,t in origin_map[1] if dim==3}
        rx={int(t) for dim,t in origin_map[2] if dim==3}
        water=all_volumes-tx-rx
        if (not tx or not rx or not water or tx & rx
                or (tx|rx|water)!=all_volumes):
            raise ValueError("OCC material mapping is ambiguous")
        region_map={
            1:("seawater",water),
            2:("transmitter_insulation",tx),
            3:("receiver_insulation",rx),
        }
        for tag,(name,volumes) in region_map.items():
            gmsh.model.addPhysicalGroup(
                3,sorted(volumes),tag=tag,name=name)

        # External Dirichlet boundary is the six OUTER cube faces only.
        # Internal water/dry interfaces are NOT artificial outer surfaces.
        boundary=gmsh.model.getBoundary(
            [(3,t) for t in sorted(water)],
            combined=True,oriented=False,recursive=False)
        exterior=[]
        tolerance=max(1e-6,1e-6*radius)  # OCC face bbox carries geometric tolerance
        for dim,surface in boundary:
            if dim!=2:
                continue
            box=np.asarray(gmsh.model.getBoundingBox(2,surface),float)
            for axis in range(3):
                if (abs(box[axis+3]-box[axis])<tolerance
                        and (abs(box[axis]-(center[axis]-radius))<tolerance
                             or abs(box[axis]-(center[axis]+radius))<tolerance)):
                    exterior.append(int(surface))
                    break
        if len(set(exterior))!=6:
            raise ValueError(
                "Exterior cube surfaces were not uniquely recovered "
                "after OCC fragmentation")
        gmsh.model.addPhysicalGroup(
            2,sorted(set(exterior)),tag=101,name="outer_A_tangential_zero")

        field_tags=[]
        for assembly in scene.assemblies:
            half=np.asarray(assembly.enclosure.half_extents,float)
            extent=np.abs(assembly.pose.rotation)@half
            c=np.asarray(assembly.pose.center,float)
            lower=c-extent-pad
            upper=c+extent+pad
            field=gmsh.model.mesh.field.add("Box")
            gmsh.model.mesh.field.setNumber(field,"VIn",near)
            gmsh.model.mesh.field.setNumber(field,"VOut",far)
            for axis,name in enumerate("XYZ"):
                gmsh.model.mesh.field.setNumber(
                    field,name+"Min",float(lower[axis]))
                gmsh.model.mesh.field.setNumber(
                    field,name+"Max",float(upper[axis]))
            gmsh.model.mesh.field.setNumber(
                field,"Thickness",max(pad,2*near))
            field_tags.append(field)
        minimum=gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            minimum,"FieldsList",field_tags)
        gmsh.model.mesh.field.setAsBackgroundMesh(minimum)

        gmsh.model.mesh.generate(3)
        gmsh.write(str(output))
        volumes={
            name:float(sum(gmsh.model.occ.getMass(3,t)
                           for t in volume_tags))
            for _,(name,volume_tags) in region_map.items()
        }
        return {
            "mesh":str(output),
            "physical_volume_tags":{
                "seawater":1,"transmitter_insulation":2,
                "receiver_insulation":3},
            "physical_surface_tags":{"outer_A_tangential_zero":101},
            "geometric_volume_m3":volumes,
        }
    finally:
        gmsh.finalize()
