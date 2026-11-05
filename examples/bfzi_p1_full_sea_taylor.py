"""P1-C: genuine five-mode original 1/r Green L on the REAL BFZI wet domain.

The 2026-11-05 run found 23.6% y-axis change and an 80.9% difference
versus Taylor: the previous symmetric-difference representation suffered
98% cancellation on a coarse origin-centred ray grid. Rather than
tuning this cancellation away, assemble the Green potential directly
in x-centred spherical coordinates, where (r**2 / r) dr = r dr.
Exact wet ray intervals intersect both independently posed insulating
boxes and the outer ball. The outer x integration uses already available
face/edge-aligned positive box-shell cubature (not a volumetric mesh).

The five-rule x/inner experiment plus independently exact
S/M-moment-guided angular refinement is diagnostic until five-mode
weak self-energy convergence and actual continuous accuracy are established.
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
        if isinstance(rule[0],str) and rule[0] in ("shell","shell_phys"):
            # Physical corner silhouettes of BOTH actual OBBs constrain the
            # spherical part of the positive shell partition of unity.
            # The accepted historical "shell" rules stay unchanged.
            geometry = (
                dict(radial_geometry_cuts=True,
                     polar_geometry_order=2,
                     azimuthal_geometry_order=2,
                     max_radial_panel_width=min(m.width for m in basis.modes),
                     shell_tangential_max_panel_width=
                         0.5*min(m.width for m in basis.modes))
                if rule[0]=="shell_phys" else {}
            )
            cache[rule] = scene.box_shell_quadrature(
                *rule[1:4],wall_scale=basis.wall_scale,
                shell_width_factor=12.0,
                shell_normal_order=rule[4],
                shell_tangential_order=4,
                shell_normal_mapping="log",
                outer_taper_width=basis.outer_taper_width,
                **geometry,
            )
        else:
            cache[rule] = scene.quadrature(
                *rule,wall_layers=1,wall_scale=basis.wall_scale,
                outer_taper_width=basis.outer_taper_width,
            )
    return cache[rule]


def trial(basis,x_rule,y_rule,cache=None,method="difference",angle_scheme="uniform",scalar_control=False,affine_control=False,moment_angle_tol=None,angle_levels=1):
    # Legacy x/y disjoint requirement remains for Taylor benchmark.
    # New direct radial solver uses y_rule as (radial,polar,azimuth) orders.
    if x_rule == y_rule:
        raise ValueError("P1-C requires disjoint x/y quadrature rules")
    if method not in ("difference","taylor","radial"):
        raise ValueError("unknown original Coulomb assembly method")
    if angle_scheme not in ("uniform","box_corners"):
        raise ValueError("unknown angular integration scheme")
    if (scalar_control or affine_control) and method!="radial":
        raise ValueError("exact moment control requires radial Green assembly")
    if affine_control and not scalar_control:
        raise ValueError("affine control requires scalar Green control")
    if moment_angle_tol is not None and method!="radial":
        raise ValueError("angular moment refinement requires radial Green")
    if cache is None:
        cache={}
    scene=basis.scene
    x,wx=_quadrature(scene,basis,x_rule,cache)
    if not np.all(scene.in_seawater(x)):
        raise ValueError("physical x cubature produced a solid point")
    angle_stats={}
    if method=="radial":
        L,diagnostic=assemble_basis_coulomb_radial(
            basis,np.eye(basis.size),x,wx,
            radial_order=y_rule[0],
            polar_order=y_rule[1],
            azimuthal_order=y_rule[2],
            box_corner_azimuth_panels=(angle_scheme=="box_corners"),
            exact_scalar_control=scalar_control,
            exact_affine_control=affine_control,
            angular_moment_tolerance=moment_angle_tol,
            angular_refine_levels=angle_levels,
            angular_moment_stats=angle_stats if moment_angle_tol is not None else None,
            return_diagnostics=True,
        )
        ny=(
            f"inner_radial_{y_rule[0]}_polar_{y_rule[1]}_"
            f"azimuth_{y_rule[2]}_scheme_{angle_scheme}"
            +("_exact_affine_SM" if affine_control else
              "_exact_scalar_S" if scalar_control else "")
            +("_moment_adaptive_angle" if moment_angle_tol is not None else "")
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
        angular_stats=angle_stats,
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



def _outer_gram_diagnostics(scene,basis):
    """Separate box normal, bulk spherical and face tangential x errors.

    The exact wet volume and its second moment are available WITHOUT
    electromagnetic field solutions. Their agreement checks the cubature
    partition before current energy is considered. The physical-current
    Gram G=integral Wi.Wj then isolates x from every Green inner rule.
    """
    radius=scene.outer_radius
    center=np.asarray(scene.outer_center)
    volume_exact=4*np.pi*radius**3/3
    second_exact=4*np.pi*radius**5/5
    for assembly in scene.assemblies:
        half=np.asarray(assembly.enclosure.half_extents)
        vol=8*np.prod(half)
        shift=np.asarray(assembly.pose.center)-center
        volume_exact-=vol
        second_exact-=vol*(
            np.dot(shift,shift)+np.dot(half,half)/3
        )

    # x0 -> spherical_only changes physical bulk angular/radial sampling;
    # x0 -> normal_only changes ONLY the true box-wall normal Gauss degree.
    # The combination exactly reproduces the x1 geometry rule.
    # Last case halves max local tangential panel width relative to the
    # narrowest actual W Gaussian window. No other knob changes.
    half_mode_width=0.5*min(mode.width for mode in basis.modes)
    mode_width=min(mode.width for mode in basis.modes)
    radial_physical=dict(
        radial_geometry_cuts=True,
        max_radial_panel_width=mode_width,
    )
    angular_physical=dict(
        polar_geometry_order=2,
        azimuthal_geometry_order=2,
    )
    physical={**radial_physical,**angular_physical}
    cases=(
        ("x0",(2,3,6,2),None,{}),
        ("sphere_only",(3,4,8,2),None,{}),
        ("normal_only",(2,3,6,3),None,{}),
        ("x1",(3,4,8,3),None,{}),
        ("face_tangent",(3,4,8,3),half_mode_width,{}),
        # Physically specified knot positions keep the same continuous
        # volume/partition. Separate these geometric subspaces before
        # combining them in a testable candidate for the actual L.
        ("sphere_radial",(3,4,8,3),None,radial_physical),
        ("sphere_angles",(3,4,8,3),None,angular_physical),
        ("sphere_phys",(3,4,8,3),None,physical),
        ("phys_tangent",(3,4,8,3),half_mode_width,physical),
        # Two independent physical-coordinate refinements after the
        # 102-pass full-L pilot still varied weak self-energies by 12-23%.
        # Only the named axis is changed at each step, and these tests
        # evaluate G (not a second 20k-point Green double integral).
        ("phys_tangent_half",(3,4,8,3),0.5*half_mode_width,physical),
        ("phys_angle3",(3,4,8,3),half_mode_width,{
            **radial_physical,
            "polar_geometry_order":3,
            "azimuthal_geometry_order":3,
        }),
        # The 103-pass log showed that physical angle order 2->3
        # changes weak Gram self terms by >2% while the tangent axis
        # is already stable at 6e-5: refine ANGLES ONLY once more.
        ("phys_angle4",(3,4,8,3),half_mode_width,{
            **radial_physical,
            "polar_geometry_order":4,
            "azimuthal_geometry_order":4,
        }),
    )
    gram={}
    gram_parts={}
    for label,(nr,npole,naz,nwall),tangent_cap,extra in cases:
        xx,ww,nsphere=scene.box_shell_quadrature(
            nr,npole,naz,wall_scale=basis.wall_scale,
            shell_width_factor=12.0,shell_normal_order=nwall,
            shell_tangential_order=4,
            shell_tangential_max_panel_width=tangent_cap,
            shell_normal_mapping="log",
            outer_taper_width=basis.outer_taper_width,
            return_sphere_count=True,
            **extra,
        )
        currents=basis.evaluate(xx)
        sphere=np.einsum(
            "n,nic,njc->ij",ww[:nsphere],
            currents[:nsphere],currents[:nsphere],optimize=True,
        )
        shell=np.einsum(
            "n,nic,njc->ij",ww[nsphere:],
            currents[nsphere:],currents[nsphere:],optimize=True,
        )
        gram[label]=sphere+shell
        gram_parts[label]=(sphere,shell)
        wet=np.einsum("ni,ni->n",xx-center,xx-center)
        volume=float(np.sum(ww))
        moment=float(np.dot(ww,wet))
        diag=np.diag(gram[label])
        contribution=np.diag(shell)/np.maximum(diag,1e-40)
        print(
            f"p1_full_sea_outer_gram: variant={label},"
            f"x_nodes={len(xx)},sphere_nodes={nsphere},"
            f"tangent_cap={tangent_cap},"
            +",".join(f"G{i}{i}={v:.10g}" for i,v in enumerate(diag))
            +f",norm_G={np.linalg.norm(gram[label]):.10g},"
            +f"volume_relative_error={abs(volume-volume_exact)/volume_exact:.8g},"
            +f"second_moment_relative_error={abs(moment-second_exact)/second_exact:.8g},"
            +",".join(f"shell_fraction{i}={v:.8g}"
                      for i,v in enumerate(contribution))
            +",scope=x_only_original_W_no_Green_kernel",
            flush=True,
        )

    def compare(label,new,base):
        a,b=gram[new],gram[base]
        d=np.abs(np.diag(a)-np.diag(b))/np.maximum(np.abs(np.diag(b)),1e-40)
        asp,ash=gram_parts[new]
        bsp,bsh=gram_parts[base]
        print(
            f"p1_full_sea_outer_gram_axis: comparison={label},"
            f"relative_G={_relative(a,b):.8g},"
            f"relative_sphere={_relative(asp,bsp):.8g},"
            f"relative_shell={_relative(ash,bsh):.8g},"
            +",".join(f"self{i}={v:.8g}" for i,v in enumerate(d))
            +",scope=independent_x_axis_no_Green",
            flush=True,
        )
    compare("sphere_only_vs_x0","sphere_only","x0")
    compare("normal_only_vs_x0","normal_only","x0")
    compare("x1_vs_sphere_only","x1","sphere_only")
    compare("x1_vs_normal_only","x1","normal_only")
    compare("face_tangent_vs_x1","face_tangent","x1")
    compare("sphere_radial_vs_x1","sphere_radial","x1")
    compare("sphere_angles_vs_x1","sphere_angles","x1")
    compare("sphere_phys_vs_sphere_radial","sphere_phys","sphere_radial")
    compare("sphere_phys_vs_sphere_angles","sphere_phys","sphere_angles")
    compare("phys_tangent_vs_sphere_phys","phys_tangent","sphere_phys")
    compare("phys_tangent_vs_face_tangent","phys_tangent","face_tangent")
    compare("phys_tangent_half_vs_phys_tangent","phys_tangent_half","phys_tangent")
    compare("phys_angle3_vs_phys_tangent","phys_angle3","phys_tangent")
    compare("phys_angle4_vs_phys_angle3","phys_angle4","phys_angle3")

    # Independent sampling algorithm on the same continuous box-shell
    # partition. Two independent Sobol scrambles are an uncertainty
    # diagnostic, NOT an error bound or a pass criterion. Never use this
    # QMC experiment as the full Coulomb Green reference.
    qmc={}
    for seed in (13,37):
        xq,wq=scene.box_shell_qmc(
            m=9,seed=seed,wall_scale=basis.wall_scale,
            shell_width_factor=12.0,sphere_extra_bits=3,
        )
        wval=basis.evaluate(xq)
        q=np.einsum("n,nic,njc->ij",wq,wval,wval,optimize=True)
        qmc[seed]=q
        qself=np.diag(q)
        print(
            f"p1_full_sea_outer_gram_qmc: seed={seed},"
            f"x_nodes={len(xq)},"
            +",".join(f"G{i}{i}={v:.10g}" for i,v in enumerate(qself))
            +f",relative_vs_gauss_face_tangent={_relative(q,gram['face_tangent']):.8g},"
            +f"relative_vs_phys_tangent={_relative(q,gram['phys_tangent']):.8g},"
            "scope=independent_scrambled_positive_volume_not_Green",
            flush=True,
        )
    q0,q1=qmc[13],qmc[37]
    d=np.abs(np.diag(q1)-np.diag(q0))/np.maximum(
        np.abs(np.diag(q0)),1e-40,
    )
    print(
        "p1_full_sea_outer_gram_qmc_spread: "
        f"relative_G={_relative(q1,q0):.8g},"
        +",".join(f"self{i}={v:.8g}" for i,v in enumerate(d))
        +",scope=two_seeds_not_error_bound",
        flush=True,
    )

    # Backward-compatible key to compare the previous completed run.
    a,b=gram["x1"],gram["x0"]
    d=np.abs(np.diag(a)-np.diag(b))/np.maximum(np.diag(b),1e-40)
    print(
        f"p1_full_sea_outer_gram_change: relative_G={_relative(a,b):.8g},"
        +",".join(f"self{i}={v:.8g}" for i,v in enumerate(d))
        +",scope=x_cubature_only_not_L",
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
    # A deliberately separate no-Coulomb operator, isolating the x
    # cubature axes and the physical face tangential Gaussian footprint.
    _outer_gram_diagnostics(scene,basis)
    # The previous full five-mode box-corner pilot was falsified by
    # independent scalar S_Omega benchmarks and worsening raw skew.
    # Retain its inexpensive analytic scalar geometry checks above,
    # but replace its expensive Green run by the exact-domain-moment
    # control variate at the SAME x, angular and radial nodes as "both".
    controlled=trial(
        basis,x1,inner1,cache,method="radial",
        angle_scheme="uniform",scalar_control=True,
    )
    print(
        "p1_full_sea: variant=scalar_control_both,method=radial,"
        f"x_nodes={controlled['nx']},inner_rule={controlled['ny']},"
        f"norm_L={controlled['norm']:.10g},"
        f"trace_L={controlled['trace']:.10g},"
        f"raw_skew={controlled['diagnostic']:.7g},"
        f"min_eig={controlled['min_eig']:.10g},"
        f"max_eig={controlled['max_eig']:.10g}",
        flush=True,
    )
    _report_modes("scalar_control_both",controlled["L"])
    print(
        "p1_full_sea_scalar_control_cross: "
        "same_x_inner_radial_polar_azimuth_and_modes,"
        f"relative_L={_relative(controlled['L'],results['both']['L']):.8g},"
        f"self_relative={_self_change(controlled['L'],results['both']['L']):.8g},"
        f"raw_skew_before={results['both']['diagnostic']:.8g},"
        f"raw_skew_after={controlled['diagnostic']:.8g},"
        "gate=diagnostic_not_truth_for_varying_W",
        flush=True,
    )
    # The global affine correction in the 105-pass run inflated actual
    # weak-mode self energies 9-36x and global L by 3.5x. Its exact
    # affine test is valid, but its full-domain Taylor residual is NOT
    # numerically useful for sharp actual wall currents. Replace that
    # expensive pilot with actual source evaluation on angular rays
    # selected by independently known exact S and M geometry moments.
    # For the matched x1 + inner1 baseline, only angle nodes may change;
    # the true wet intervals and all radial rules remain untouched.
    adaptive=trial(
        basis,x1,inner1,cache,method="radial",
        angle_scheme="uniform",scalar_control=True,
        moment_angle_tol=0.0025,angle_levels=1,
    )
    print(
        "p1_full_sea: variant=moment_angle_both,method=radial,"
        f"x_nodes={adaptive['nx']},inner_rule={adaptive['ny']},"
        f"norm_L={adaptive['norm']:.10g},"
        f"trace_L={adaptive['trace']:.10g},"
        f"raw_skew={adaptive['diagnostic']:.7g},"
        f"min_eig={adaptive['min_eig']:.10g},"
        f"max_eig={adaptive['max_eig']:.10g}",
        flush=True,
    )
    _report_modes("moment_angle_both",adaptive["L"])
    st=adaptive["angular_stats"]
    print(
        "p1_full_sea_moment_angle_quality: "
        f"target=0.0025,levels=1,"
        f"points={st['total']},refined={st['refined']},"
        f"unresolved={st['unresolved']},"
        f"max_scalar_rel={st['maximum_scalar_rel']:.8g},"
        f"max_first_scaled={st['maximum_first_scaled']:.8g},"
        "first_moment_scale=S_exact_times_max_ray,"
        "selection=geometry_only_before_curl_evaluation,"
        "gate=diagnostic_unresolved_not_certified",
        flush=True,
    )
    print(
        "p1_full_sea_moment_angular_cross: "
        "same_x_true_wet_domain_radial_rules_and_modes,"
        f"relative_vs_scalar={_relative(adaptive['L'],controlled['L']):.8g},"
        f"relative_vs_original={_relative(adaptive['L'],results['both']['L']):.8g},"
        f"raw_skew_scalar={controlled['diagnostic']:.8g},"
        f"raw_skew_adaptive={adaptive['diagnostic']:.8g},"
        "gate=diagnostic_not_continuum_certification",
        flush=True,
    )
    # The affine experiment stays callable as an explicit research
    # reference, but no longer runs on every five-mode G1 attempt.
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
    per_mode_changes("scalar_control_vs_uniform_both",controlled["L"],both)
    per_mode_changes("moment_angle_vs_scalar_both",adaptive["L"],controlled["L"])
    per_mode_changes("moment_angle_vs_uniform_both",adaptive["L"],both)
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
