"""P1-C: original Coulomb 1/r for ACTUAL wall-following BFZI seawater curl modes.

Unlike P1-B's two isolated compact support balls, this pilot integrates
global physical seawater volume: the outer sphere MINUS two independently
posed insulating boxes. The current modes are drawn byte-for-byte from
the existing BFZI SeaCurrentBasis (no artificial surrogate vortices).

Its Taylor subtraction uses exact full-domain S_Omega/M_Omega and the
analytic Hessian of the actual curl-potential envelope. It cannot be
declared converged from just two numerical x/y rules: report the
independent changes and raw skew, then advance on actual evidence.
"""
from __future__ import annotations

import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.sea_basis import SeaCurrentBasis, build_sea_basis
from sdfmpneo_bfzi.green_coulomb import assemble_basis_coulomb_taylor


def trial(basis,x_rule,y_rule,cache=None):
    scene=basis.scene
    if cache is None:
        cache={}
    opts=dict(
        wall_layers=1,wall_scale=basis.wall_scale,
        outer_taper_width=basis.outer_taper_width,
    )
    for rule in (x_rule,y_rule):
        if rule not in cache:
            cache[rule]=scene.quadrature(*rule,**opts)
    x,wx=cache[x_rule]
    y,wy=cache[y_rule]
    if not np.all(scene.in_seawater(x)) or not np.all(scene.in_seawater(y)):
        raise ValueError("full sea Coulomb cubature produced a solid point")
    l,skew=assemble_basis_coulomb_taylor(
        basis,np.eye(basis.size),x,wx,y,wy,
        return_diagnostics=True,
    )
    eig=np.linalg.eigvalsh(l)
    return {
        "L":l,"skew":skew,"nx":len(x),"ny":len(y),
        "trace":float(np.trace(l)),
        "norm":float(np.linalg.norm(l)),
        "min_eig":float(np.min(eig)),
        "max_eig":float(np.max(eig)),
    }


def _relative(a,b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-40))


def main():
    scene=build_demo_scene()
    full=build_sea_basis(
        scene,modes_per_face=2,face_refinement=1,
    )
    # Two explicit face curls on each independently mounted housing,
    # plus one actual global curl mode. All have their genuine box
    # distance factors and outer taper. No source-guided mode is faked.
    chosen=(0,1,12,13,24)
    basis=SeaCurrentBasis(
        scene,tuple(full.modes[i] for i in chosen),
        full.wall_scale,full.source_lines,
        full.outer_envelope,full.outer_taper_width,
    )
    if any(m.source_port>=0 for m in basis.modes):
        raise ValueError("source-guided C curls await analytic Jacobians")
    print(
        "p1_full_sea_exact: physical_domain=outer_ball_minus_two_actual_boxes,"
        "modes=5,mode_indices=0-1-12-13-24,"
        "current=existing_box_following_curl_basis,"
        "domain_S_M=exact_true_rotated_geometry,"
        "current_grad=analytic_mode_envelope_hessian,"
        "original_kernel=1/r,no_softening=True,"
        "gate=convergence_study_not_certified",
        flush=True,
    )
    x0=(3,5,8)
    y0=(4,6,10)
    x1=(4,6,10)
    y1=(5,7,12)
    trials={}
    rules_cache={}
    for name,xr,yr in (
        ("base",x0,y0),
        ("x_only",x1,y0),
        ("y_only",x0,y1),
        ("both",x1,y1),
    ):
        z=trial(basis,xr,yr,rules_cache)
        trials[name]=z
        print(
            f"p1_full_sea: variant={name},"
            f"x_nodes={z['nx']},y_nodes={z['ny']},"
            f"norm_L={z['norm']:.10g},"
            f"trace_L={z['trace']:.10g},"
            f"raw_skew={z['skew']:.7g},"
            f"min_eig={z['min_eig']:.10g},"
            f"max_eig={z['max_eig']:.10g}",
            flush=True,
        )
    base=trials["base"]["L"]
    fine=trials["both"]["L"]
    print(
        f"p1_full_sea_change: "
        f"x_only_vs_base={_relative(trials['x_only']['L'],base):.7g},"
        f"y_only_vs_base={_relative(trials['y_only']['L'],base):.7g},"
        f"both_vs_base={_relative(fine,base):.7g},"
        f"both_vs_x_only={_relative(fine,trials['x_only']['L']):.7g},"
        f"both_vs_y_only={_relative(fine,trials['y_only']['L']):.7g},"
        "pass_status=not_determined_without_independent_truth,"
        flush=True,
    )
    print(
        "p1_full_sea_note: this is the first actual wall-following BFZI "
        "curl space on the REAL two-box finite seawater domain using "
        "analytic physical S/M and DW. Same five physical modes for every "
        "x/y comparison. No known exact L for this space, so these are "
        "uncertified numerical changes, not a 0.1-percent accuracy gate. "
        "Source-guided modes, wall-edge regularity, full basis density and "
        "infinite ocean remain separate unresolved gates.",
        flush=True,
    )


if __name__=="__main__":
    main()
