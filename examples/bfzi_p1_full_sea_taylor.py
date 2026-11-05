"""P1-C: genuine five-mode original 1/r Green L on the REAL BFZI wet domain.

The 2026-11-05 run found 23.6% y-axis change and an 80.9% difference
versus Taylor: the previous symmetric-difference representation suffered
98% cancellation on a coarse origin-centred ray grid. Rather than
tuning this cancellation away, assemble the Green potential directly
in x-centred spherical coordinates, where (r**2 / r) dr = r dr.
Exact wet ray intervals intersect both independently posed insulating
boxes and the outer ball. The outer x integration uses already available
face/edge-aligned positive box-shell cubature (not a volumetric mesh).

The present four-rule experiment is diagnostic until actual x/angular/
radial, near-wall and near-edge convergence has independent evidence.
No neural teacher or G1 certificate is produced.
"""
from __future__ import annotations

import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.sea_basis import SeaCurrentBasis, build_sea_basis
from sdfmpneo_bfzi.green_coulomb import (
    assemble_basis_coulomb_taylor,
    assemble_basis_coulomb_difference,
    assemble_basis_coulomb_radial,
)


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


def trial(basis,x_rule,y_rule,cache=None,method="difference"):
    # Legacy x/y disjoint requirement remains for Taylor benchmark.
    # New direct radial solver uses y_rule as (radial,polar,azimuth) orders.
    if x_rule == y_rule:
        raise ValueError("P1-C requires disjoint x/y quadrature rules")
    if method not in ("difference","taylor","radial"):
        raise ValueError("unknown original Coulomb assembly method")
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
            return_diagnostics=True,
        )
        ny=f"inner_radial_{y_rule[0]}_polar_{y_rule[1]}_azimuth_{y_rule[2]}"
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
        "p1_full_sea_exact: domain=outer_ball_minus_two_actual_rotated_boxes,"
        "modes=5,mode_indices=0-1-12-13-24,"
        "kernel=original_unsoftened_1/r,"
        "inner=point_centered_exact_wet_ray_intervals,"
        "inner_measure=r_dr_dsolidangle,"
        "outer=geometry_aligned_positive_box_shell_cubature,"
        "wall_normal=log_panels,gate=not_certified",
        flush=True,
    )
    # Only two independent physical axes are changed in this experiment:
    # (a) the x external cubature, now using oriented box shells
    # (b) the Green potential's x-centred radial/angular chart.
    # Do not interpret the four outcomes as continuous reference truth.
    x0=("shell",2,3,6,2)
    x1=("shell",3,4,8,3)
    inner0=(3,3,8)
    inner1=(4,4,10)
    cases=(
        ("base",x0,inner0),
        ("x_only",x1,inner0),
        ("inner_only",x0,inner1),
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
    b=results["base"]["L"]
    xo=results["x_only"]["L"]
    yo=results["inner_only"]["L"]
    both=results["both"]["L"]
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
