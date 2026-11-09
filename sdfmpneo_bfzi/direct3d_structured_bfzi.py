"""Geometry-conforming structured *axis-aligned* BFZI D2 tetra mesh.

This is a genuine two-port Maxwell-MQS smoke/continuation mesh usable
without Gmsh. Its conforming material faces are exact for two individually
axis-aligned (signed-permutation) insulating boxes. Arbitrary rotated SE(3)
boxes MUST use the Gmsh BooleanFragments implementation; silently
voxelizing rotated boxes would invalidate interface currents and Joule heat.
"""
from __future__ import annotations

import numpy as np

from .direct3d_aphi import TetraMesh


def _signed_axis_aligned(rotation):
    a=np.abs(np.asarray(rotation,float))
    return bool(np.all(np.minimum(a,1-a)<1e-9))


def _axis_coordinates(scene, axis, pad, far_step, layers, near_step, core_radius):
    center=float(scene.outer_center[axis])
    radius=float(scene.outer_radius)
    low, high=center-radius,center+radius
    knots=[low,high]
    if core_radius is not None:
        # Keep identical near-field grid planes across an outer-R sweep.
        knots.extend((center-core_radius, center+core_radius))
    near_intervals=[]
    for assembly in scene.assemblies:
        h=np.asarray(assembly.enclosure.half_extents,float)
        extent=np.abs(assembly.pose.rotation)@h
        c=float(assembly.pose.center[axis])
        a,b=c-extent[axis],c+extent[axis]
        if not low<a<b<high:
            raise ValueError("Insulating box intersects exterior truncation")
        # The oriented box is actually axis-aligned for this restricted mesh.
        # Hence *all* its physical material faces are mesh coordinate planes.
        knots.extend(np.linspace(a,b,layers+1))
        knots.extend([max(low,a-pad),min(high,b+pad)])
        near_intervals.append((max(low,a-pad),min(high,b+pad)))
    raw=np.sort(np.asarray(knots,float))
    unique=[float(raw[0])]
    for k in raw[1:]:
        if float(k)-unique[-1]>1e-11:
            unique.append(float(k))
    pts=[unique[0]]
    for left,right in zip(unique[:-1],unique[1:]):
        # This interval lies entirely inside or outside the padded
        # material coordinate slab because both endpoints are knots.
        midpoint=.5*(left+right)
        near=any(a-1e-11 <= midpoint <= b+1e-11
                 for a,b in near_intervals)
        step=near_step if near and near_step is not None else far_step
        count=max(1,int(np.ceil((right-left)/step)))
        pts.extend(np.linspace(left,right,count+1)[1:])
    return np.asarray(pts,float)


def axis_aligned_bfzi_tetra_mesh(scene, *,
                                near_padding_m: float = .025,
                                far_step_m: float = .22,
                                box_layers: int = 2,
                                near_step_m: float | None = None,
                                inner_core_radius_m: float | None = None) -> TetraMesh:
    """Structured 3D conforming tetrahedra around both genuine BFZI boxes.

    No hard-coded demo geometry. Both enclosures may have independent
    centers, dimensions, and right-handed axis-permutation orientations.
    Reject any genuinely oblique pose rather than approximating its water
    interface with stair-step voxels.

    inner_core_radius_m preserves the same inner coordinate cutting planes
    when enlarging R; exterior shells then attach to a fixed inner cube.
    It must enclose both padded boxes and stay strictly inside outer R.

    near_step_m independently resolves conducting seawater slabs just
    OUTSIDE the insulating walls (as well as the nearby dry coordinate
    intervals). Refining only box_layers would otherwise change the dry
    mesh without resolving wet wall gradients. This is a TENSOR-PRODUCT
    mesh: cuts extend across the full other axes, so use moderate values
    and prefer OCC/Gmsh for economical full SE(3) adaptive refinement.

    The entire domain is a finite CUBE, as in direct3d_mesh.py.
    The near/far grid is deliberately low order and unverified; use
    this for direct physical BFZI end-to-end smoke and then the real
    Gmsh SE(3) path for high-fidelity / open-boundary certification.
    """
    if (not np.isfinite(near_padding_m) or near_padding_m<0
            or not np.isfinite(far_step_m) or far_step_m<=0
            or isinstance(box_layers,bool) or not isinstance(box_layers,int)
            or box_layers<1):
        raise ValueError("Expected nonnegative near padding, positive far step and box_layers>=1")
    if (near_step_m is not None
            and (not np.isfinite(near_step_m)
                 or not 0 < near_step_m <= far_step_m)):
        raise ValueError("near_step_m must be positive and <= far_step_m")
    if near_step_m is not None and near_padding_m == 0:
        raise ValueError("near_step_m needs near_padding_m > 0 to refine wet seawater")
    if inner_core_radius_m is not None:
        core=float(inner_core_radius_m)
        if not np.isfinite(core) or not 0 < core < scene.outer_radius:
            raise ValueError("inner_core_radius_m must be between 0 and outer radius")
        for assembly in scene.assemblies:
            extent=np.abs(assembly.pose.rotation)@np.asarray(
                assembly.enclosure.half_extents,float)
            outer=np.abs(np.asarray(assembly.pose.center,float)-
                         np.asarray(scene.outer_center,float))
            if np.any(outer+extent+near_padding_m >= core):
                raise ValueError("inner_core_radius_m must enclose both padded OBBs")
    for assembly in scene.assemblies:
        if not _signed_axis_aligned(assembly.pose.rotation):
            raise ValueError(
                "Oblique SE(3) box requires Gmsh conforming BooleanFragments mesh; "
                "structured voxel approximation is forbidden")
    axes=[_axis_coordinates(scene,k,near_padding_m,far_step_m,
                            box_layers,near_step_m,inner_core_radius_m)
          for k in range(3)]
    sizes=[len(axis)-1 for axis in axes]
    x=np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)
    def index(i,j,k):
        return (i*(sizes[1]+1)+j)*(sizes[2]+1)+k
    cube=((0,1,3,7),(0,3,2,7),(0,2,6,7),
          (0,6,4,7),(0,4,5,7),(0,5,1,7))
    tetra=[]
    sigma=[]
    for i in range(sizes[0]):
        for j in range(sizes[1]):
            for k in range(sizes[2]):
                nodes=(index(i,j,k),index(i+1,j,k),
                       index(i,j+1,k),index(i+1,j+1,k),
                       index(i,j,k+1),index(i+1,j,k+1),
                       index(i,j+1,k+1),index(i+1,j+1,k+1))
                midpoint=(x[nodes[0]]+x[nodes[-1]])/2
                dry=any(bool(assembly.enclosure.contains(
                    midpoint,assembly.pose))
                    for assembly in scene.assemblies)
                conductivity=0. if dry else float(scene.seawater.conductivity)
                for cell in cube:
                    tetra.append([nodes[v] for v in cell])
                    sigma.append(conductivity)
    return TetraMesh(x,np.asarray(tetra,int),np.asarray(sigma,float))
