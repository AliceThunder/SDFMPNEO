"""P1-C: genuine five-mode original 1/r Green L on the REAL BFZI wet domain.

The 2026-11-05 run found 23.6% y-axis change and an 80.9% difference
versus Taylor: the previous symmetric-difference representation suffered
98% cancellation on a coarse origin-centred ray grid. Rather than
tuning this cancellation away, assemble the Green potential directly
in x-centred spherical coordinates, where (r**2 / r) dr = r dr.
Exact wet ray intervals intersect both independently posed insulating
boxes and the outer ball. The outer x integration uses already available
face/edge-aligned positive box-shell cubature (not a volumetric mesh).

The five-rule x/inner experiment plus one corner-projected angular pilot
is diagnostic until near-wall and near-edge convergence is established.
No neural teacher or G1 certificate is produced.
"""
from __future__ import annotations

from dataclasses import replace
import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.sea_basis import SeaCurrentBasis, build_sea_basis
from sdfmpneo_bfzi.green_coulomb import (
    assemble_basis_coulomb_taylor,
    assemble_basis_coulomb_difference,
    assemble_basis_coulomb_radial,
    assemble_coulomb_radial,
    _bfzi_wet_radial_segments,
    _bfzi_angular_face_frame,
    _bfzi_projected_box_azimuth_rule,
)
from sdfmpneo_bfzi.coulomb_geometry_moments import sea_coulomb_moments
from sdfmpneo_bfzi.source import MU0
from sdfmpneo_bfzi.geometry import RigidPose


def _relative(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-40))


def _self_change(a, b):
    return _relative(np.diag(a),np.diag(b))


def _mutual_change(a, b):
    mask = np.triu(np.ones_like(a,dtype=bool),1)
    return float(np.linalg.norm((a-b)[mask])/
                 max(np.linalg.norm(b),1e-40))


def _quadrature(scene, basis, rule, cache):
    if rule not in cache:
        if isinstance(rule[0],str) and rule[0] == "shell":
            # Physical coordinate separation: local box faces and edges
            # are represented directly; no spherical angular wet-mask.
            cache[rule] = scene.box_shell_quadrature(
                *rule[1:4],wall_scale=basis.wall_scale,
                shell_width_factor=12.0,
                shell_normal_order=rule[4],
                shell_tangential_order=4,
                shell_normal_mapping="log",
                outer_taper_width=basis.outer_taper_width,
            )
        else:
            cache[rule] = scene.quadrature(
                *rule,wall_layers=1,wall_scale=basis.wall_scale,
                outer_taper_width=basis.outer_taper_width,
            )
    return cache[rule]


def trial(basis,x_rule,y_rule,cache=None,method="difference",angle_scheme="uniform"):
    # Legacy x/y disjoint requirement remains for Taylor benchmark.
    # New direct radial solver uses y_rule as (radial,polar,azimuth) orders.
    if x_rule == y_rule:
        raise ValueError("P1-C requires disjoint x/y quadrature rules")
    if method not in ("difference","taylor","radial"):
        raise ValueError("unknown original Coulomb assembly method")
    if angle_scheme not in ("uniform","box_corners"):
        raise ValueError("unknown angular integration scheme")
    if cache is None:
        cache={}
    scene=basis.scene
    x,wx=_quadrature(scene,basis,x_rule,cache)
    if not np.all(scene.in_seawater(x)):
        raise ValueError("physical x cubature produced a solid point")
    if method=="radial":
        L,diagnostic=assemble_basis_coulomb_radial(
            basis,np.eye(basis.size),x,wx,
            radial_order=y_rule[0],
            polar_order=y_rule[1],
            azimuthal_order=y_rule[2],
            box_corner_azimuth_panels=(angle_scheme=="box_corners"),
            return_diagnostics=True,
        )
        ny=(
            f"inner_radial_{y_rule[0]}_polar_{y_rule[1]}_"
            f"azimuth_{y_rule[2]}_scheme_{angle_scheme}"
        )
    else:
        y,wy=_quadrature(scene,basis,y_rule,cache)
        if not np.all(scene.in_seawater(y)):
            raise ValueError("physical y cubature produced a solid point")
        assembler=(
            assemble_basis_coulomb_difference
            if method=="difference" else assemble_basis_coulomb_taylor
        )
        L,diagnostic=assembler(
            basis,np.eye(basis.size),x,wx,y,wy,
            return_diagnostics=True,
        )
        ny=str(len(y))
    eig=np.linalg.eigvalsh(L)
    return dict(
        L=L,method=method,diagnostic=diagnostic,nx=len(x),ny=ny,
        norm=float(np.linalg.norm(L)),trace=float(np.trace(L)),
        min_eig=float(np.min(eig)),max_eig=float(np.max(eig)),
    )


def _report_modes(name,L):
    diag=",".join(f"L{i}{i}={L[i,i]:.10g}" for i in range(len(L)))
    mutual=",".join(
        f"L{i}{j}={L[i,j]:.10g}"
        for i in range(len(L)) for j in range(i+1,len(L))
    )
    print(f"p1_full_sea_self: variant={name},{diag}",flush=True)
    print(f"p1_full_sea_mutual: variant={name},{mutual}",flush=True)



def _scalar_geometry_diagnostic(scene, wall_scale, pose_label):
    """Independent analytic S_Omega checks only ray geometry and angles.

    A constant vector density gives U(x) = S_Omega(x)*a. Its inner radial
    integral is exactly linear in r, so Gauss-2 has no radial truncation
    error on any correctly clipped wet interval. Differences from the
    exact ball-minus-box rectangle primitives isolate angular geometry.
    The constant vector is a DIAGNOSTIC density, not a legal wall current
    or a claimed physical BFZI basis mode.
    """
    wet_points = []
    kinds = []
    for index, assembly in enumerate(scene.assemblies):
        half = np.asarray(assembly.enclosure.half_extents)
        sign = 1.0 if index == 0 else -1.0
        gap = 2.0*wall_scale
        local = np.array((
            (0.0, 0.0, sign*(half[2]+gap)),
            (half[0]+gap, 0.0, sign*(half[2]+gap)),
            (half[0]+gap, half[1]+gap, sign*(half[2]+gap)),
        ))
        world = assembly.pose.position(local)
        if not np.all(scene.in_seawater(world)):
            raise ValueError("near-wall scalar geometry probes must be wet")
        wet_points.extend(world)
        kinds.extend(("face","edge","corner"))
    wet_points = np.asarray(wet_points)
    true_scalar = sea_coulomb_moments(scene,wet_points)[0]
    far_reach = 2.0*scene.outer_radius

    def constant_current(x):
        return np.broadcast_to(
            np.array([[[1.0,0.0,0.0]]]),(len(x),1,3)
        )

    def intervals(point,directions):
        return _bfzi_wet_radial_segments(
            scene,point,directions,far_reach,
        )

    def axis(point):
        frame = _bfzi_angular_face_frame(scene,point,far_reach)
        return np.array((0.0,0.0,1.0)) if frame is None else frame[0]

    for scheme in ("uniform","box_corners"):
        if scheme=="box_corners":
            azimuthal=lambda p,e1,e2,n: _bfzi_projected_box_azimuth_rule(
                scene,p,e1,e2,n,
            )
        else:
            azimuthal=None
        for polar,azimuth in ((3,8),(8,32)):
            deviations = []
            for point,truth in zip(wet_points,true_scalar):
                numerical = assemble_coulomb_radial(
                    point[None,:],np.ones(1),
                    constant_current,intervals,radial_order=2,
                    polar_order=polar,azimuthal_order=azimuth,
                    polar_axis=axis,azimuthal_rule=azimuthal,
                )[0,0]*4.0*np.pi/MU0
                deviations.append(abs(numerical-truth)/truth)
            error = dict(zip(
                ("face","edge","corner"),
                (max(e for e,k in zip(deviations,kinds) if k==kind)
                 for kind in ("face","edge","corner")),
            ))
            print(
                f"p1_full_sea_scalar: pose={pose_label},"
                f"angular_scheme={scheme},"
                f"polar_per_hemisphere={polar},azimuth={azimuth},"
                f"face_rel={error['face']:.8g},"
                f"edge_rel={error['edge']:.8g},"
                f"corner_rel={error['corner']:.8g},"
                f"max_rel={max(deviations):.8g},"
                "radial_error=analytically_zero_for_constant_density,"
                "truth=exact_ball_minus_two_box_S,"
                "scope=angular_clipping_geometry_only",
                flush=True,
            )


def main():
    scene=build_demo_scene()
    full=build_sea_basis(scene,modes_per_face=2,face_refinement=1)
    chosen=(0,1,12,13,24)
    basis=SeaCurrentBasis(
        scene,tuple(full.modes[i] for i in chosen),
        full.wall_scale,full.source_lines,
        full.outer_envelope,full.outer_taper_width,
    )
    print(
        "p1_full_sea_exact: domain=outer_ball_minus_two_actual_boxes,"
        "five_mode_poses=both_identity,"
        "separate_scalar_geometry_probes=aligned_and_two_rotated_boxes,"
        "modes=5,mode_indices=0-1-12-13-24,"
        "kernel=original_unsoftened_1/r,"
        "inner=point_centered_exact_wet_ray_intervals,"
        "inner_measure=r_dr_dsolidangle,"
        "outer=geometry_aligned_positive_box_shell_cubature,"
        "wall_normal=log_panels,gate=not_certified",
        flush=True,
    )
    _scalar_geometry_diagnostic(scene,basis.wall_scale,"aligned_demo")

    # The ordinary five-mode demonstration has identity rotations on
    # both boxes; do not claim these modes validate rotated poses.
    # A separate scalar exact-Green probe rotates each *complete* BFZI
    # assembly independently while retaining the same physical boxes.
    ax,ay = 0.24,-0.31
    rot_x = np.array([
        [1,0,0],[0,np.cos(ax),-np.sin(ax)],
        [0,np.sin(ax),np.cos(ax)],
    ])
    rot_y = np.array([
        [np.cos(ay),0,np.sin(ay)],
        [0,1,0],[-np.sin(ay),0,np.cos(ay)],
    ])
    rotated = replace(
        scene,
        transmitter=replace(
            scene.transmitter,
            pose=RigidPose(rot_x,scene.transmitter.pose.center),
        ),
        receiver=replace(
            scene.receiver,
            pose=RigidPose(rot_y,scene.receiver.pose.center),
        ),
    )
    _scalar_geometry_diagnostic(rotated,basis.wall_scale,"two_rotated_boxes")
    # The next comparison independently changes x and inner radial/angle rules:
    # (a) the x external cubature, now using oriented box shells
    # (b) the Green potential's x-centred radial/angular chart.
    # Do not interpret these cases as continuous reference truth.
    x0=("shell",2,3,6,2)
    x1=("shell",3,4,8,3)
    inner0=(3,3,8)
    inner1=(4,4,10)
    cases=(
        ("base",x0,inner0),
        ("x_only",x1,inner0),
        ("inner_only",x0,inner1),
        # At the fixed refined x rule, independently isolate the
        # inner radial change before refining angular quadrature.
        ("inner_radial_only",x1,(inner1[0],inner0[1],inner0[2])),
        ("both",x1,inner1),
    )
    cache={}
    results={}
    for name,xrule,irule in cases:
        z=trial(basis,xrule,irule,cache,method="radial")
        results[name]=z
        print(
            f"p1_full_sea: variant={name},method=radial,"
            f"x_nodes={z['nx']},inner_rule={z['ny']},"
            f"norm_L={z['norm']:.10g},trace_L={z['trace']:.10g},"
            f"raw_skew={z['diagnostic']:.7g},"
            f"min_eig={z['min_eig']:.10g},max_eig={z['max_eig']:.10g}",
            flush=True,
        )
        _report_modes(name,z["L"])
    # Independently isolate x-volume quadrature without any Coulomb kernel:
    # G_ij = integral W_i.W_j dx is the uniform-sigma resistive Galerkin
    # Gram matrix multiplied by sigma. A weak self-mode error here is an
    # OUTER cubature defect, not an inner 1/r Green cubature defect.
    grams={}
    for x_label,xrule in (("x0",x0),("x1",x1)):
        xx,ww=_quadrature(scene,basis,xrule,cache)
        currents=basis.evaluate(xx)
        g=np.einsum(
            "n,nic,njc->ij",ww,currents,currents,optimize=True,
        )
        grams[x_label]=g
        print(
            f"p1_full_sea_outer_gram: variant={x_label},"
            f"x_nodes={len(xx)},"
            +",".join(f"G{i}{i}={g[i,i]:.10g}"
                      for i in range(basis.size))
            +f",norm_G={np.linalg.norm(g):.10g},"
            "scope=original_current_W_squared_no_Green_kernel",
            flush=True,
        )
    gc,gf=grams["x0"],grams["x1"]
    gmode=np.abs(np.diag(gf)-np.diag(gc))/np.maximum(
        np.abs(np.diag(gc)),1e-40,
    )
    print(
        "p1_full_sea_outer_gram_change: "
        f"relative_G={_relative(gf,gc):.8g},"
        +",".join(f"self{i}={v:.8g}"
                  for i,v in enumerate(gmode))
        +",scope=x_cubature_only_not_L",
        flush=True,
    )
    # Hold ALL physical x and radial/angular orders fixed to measure the
    # effect of geometry-defined azimuth panels only. This is a controlled
    # pilot, not an automatic replacement or certification.
    corner=trial(
        basis,x1,inner1,cache,method="radial",
        angle_scheme="box_corners",
    )
    print(
        "p1_full_sea: variant=corner_both,method=radial,"
        f"x_nodes={corner['nx']},inner_rule={corner['ny']},"
        f"norm_L={corner['norm']:.10g},trace_L={corner['trace']:.10g},"
        f"raw_skew={corner['diagnostic']:.7g},"
        f"min_eig={corner['min_eig']:.10g},"
        f"max_eig={corner['max_eig']:.10g}",
        flush=True,
    )
    _report_modes("corner_both",corner["L"])
    b=results["base"]["L"]
    xo=results["x_only"]["L"]
    yo=results["inner_only"]["L"]
    both=results["both"]["L"]
    inner_rad=results["inner_radial_only"]["L"]
    print(
        "p1_full_sea_inner_axes: at_fixed_x=x1,"
        f"radial_step={_relative(inner_rad,xo):.7g},"
        f"angle_step={_relative(both,inner_rad):.7g},"
        f"self_radial_step={_self_change(inner_rad,xo):.7g},"
        f"self_angle_step={_self_change(both,inner_rad):.7g},"
        f"mutual_radial_step={_mutual_change(inner_rad,xo):.7g},"
        f"mutual_angle_step={_mutual_change(both,inner_rad):.7g}",
        flush=True,
    )
    print(
        "p1_full_sea_change: method=radial,"
        f"x_only_vs_base={_relative(xo,b):.7g},"
        f"inner_only_vs_base={_relative(yo,b):.7g},"
        f"both_vs_base={_relative(both,b):.7g},"
        f"both_vs_x_only={_relative(both,xo):.7g},"
        f"both_vs_inner_only={_relative(both,yo):.7g},"
        f"self_x={_self_change(xo,b):.7g},"
        f"self_inner={_self_change(yo,b):.7g},"
        f"self_both={_self_change(both,b):.7g},"
        f"mutual_x={_mutual_change(xo,b):.7g},"
        f"mutual_inner={_mutual_change(yo,b):.7g},"
        f"mutual_both={_mutual_change(both,b):.7g},"
        "pass_status=not_certified_without_independent_reference",
        flush=True,
    )
    def per_mode_changes(label, matrix_a, matrix_b):
        da,db=np.diag(matrix_a),np.diag(matrix_b)
        relative=np.abs(da-db)/np.maximum(np.abs(db),1e-40)
        # Unit-robust pair metric relative to each current mode's own
        # magnetic self-energy (not dominated by global mode 44).
        scale=np.sqrt(np.outer(np.maximum(np.abs(db),1e-40),
                               np.maximum(np.abs(db),1e-40)))
        normalized_pair=np.abs((matrix_a-matrix_b)/scale)
        upper=normalized_pair[np.triu_indices(len(db),1)]
        print(
            f"p1_full_sea_mode_delta: comparison={label},"
            +",".join(f"self{i}={v:.8g}" for i,v in enumerate(relative))
            +f",max_self={max(relative):.8g},"
            +f"max_mutual_self_normalized={max(upper):.8g}",
            flush=True,
        )

    per_mode_changes("x_only_vs_base",xo,b)
    per_mode_changes("radial_only_vs_x_only",inner_rad,xo)
    per_mode_changes("angular_only_vs_inner_radial",both,inner_rad)
    per_mode_changes("corner_both_vs_uniform_both",corner["L"],both)
    print(
        "p1_full_sea_corner_cross: same_x_and_radial_rules,"
        f"relative_L={_relative(corner['L'],both):.8g},"
        f"self_relative={_self_change(corner['L'],both):.8g},"
        f"mutual_scaled={_mutual_change(corner['L'],both):.8g},"
        "gate=diagnostic_only",
        flush=True,
    )
    # Previous 97-pass P1-C run already falsified the globally subtracted
    # difference pilot; it is not silently reused as magnetic teacher.
    print(
        "p1_full_sea_note: previous difference pilot had "
        "23.56018-percent y-axis L change, 80.94096-percent Taylor "
        "cross-formulation gap, and 98-percent subtraction ratio. "
        "New radial Green operator has no large matrix subtraction. "
        "It must still pass independent x/inner angular/radial and "
        "near-wall/corner convergence; a small raw skew or one PSD "
        "matrix is not a true solution certificate. G1 remains FAIL.",
        flush=True,
    )


if __name__=="__main__":
    main()
