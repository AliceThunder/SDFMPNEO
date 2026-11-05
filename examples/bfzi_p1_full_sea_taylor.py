"""P1-C: two independent unsoftened 1/r representations on REAL BFZI seawater.

Five genuine wall-following curl modes; outer ball minus two independently
posed insulating boxes. The new symmetric-difference Green energy removes
the diagonal singularity without the global Taylor Hessian subtraction.
The original Taylor result is retained as a cross-formulation comparison,
not as known truth. All P1-C physical convergence gates remain open.
"""
from __future__ import annotations

import numpy as np

from examples.bfzi_mqs_demo import build_demo_scene
from sdfmpneo_bfzi.sea_basis import SeaCurrentBasis, build_sea_basis
from sdfmpneo_bfzi.green_coulomb import (
    assemble_basis_coulomb_taylor,
    assemble_basis_coulomb_difference,
)


def trial(basis, x_rule, y_rule, cache=None, method="difference"):
    # Keep independent axes for an interpretable P1-C convergence matrix.
    # The symmetric difference kernel itself ALSO permits x=y nodes.
    if x_rule == y_rule:
        raise ValueError("P1-C requires disjoint x/y quadrature rules")
    if method not in ("difference", "taylor"):
        raise ValueError("unknown original Coulomb assembly method")
    scene = basis.scene
    if cache is None:
        cache = {}
    opts = dict(
        wall_layers=1,
        wall_scale=basis.wall_scale,
        outer_taper_width=basis.outer_taper_width,
    )
    for rule in (x_rule, y_rule):
        if rule not in cache:
            cache[rule] = scene.quadrature(*rule, **opts)
    x, wx = cache[x_rule]
    y, wy = cache[y_rule]
    if not np.all(scene.in_seawater(x)) or not np.all(scene.in_seawater(y)):
        raise ValueError("full sea Coulomb cubature produced a solid point")
    assembler = (
        assemble_basis_coulomb_difference
        if method == "difference" else assemble_basis_coulomb_taylor
    )
    L, diagnostic = assembler(
        basis, np.eye(basis.size), x, wx, y, wy,
        return_diagnostics=True,
    )
    eig = np.linalg.eigvalsh(L)
    return dict(
        L=L, method=method, diagnostic=diagnostic,
        nx=len(x), ny=len(y), trace=float(np.trace(L)),
        norm=float(np.linalg.norm(L)),
        min_eig=float(np.min(eig)),
        max_eig=float(np.max(eig)),
    )


def _relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-40))


def _mode_report(variant, L):
    # These numbers distinguish genuine self terms from all cross terms,
    # including face-to-face and face-to-global coupling.
    self_terms = ",".join(f"L{i}{i}={L[i,i]:.10g}" for i in range(len(L)))
    mutual_terms = ",".join(
        f"L{i}{j}={L[i,j]:.10g}"
        for i in range(len(L)) for j in range(i+1, len(L))
    )
    print(f"p1_full_sea_self: variant={variant},{self_terms}", flush=True)
    print(f"p1_full_sea_mutual: variant={variant},{mutual_terms}", flush=True)


def _diagonal_relative(a, b):
    return _relative(np.diag(a), np.diag(b))


def _mutual_relative(a, b):
    # Scale by the WHOLE reference matrix: small cross couplings are not
    # divided by an almost zero off-diagonal norm.
    mask = np.triu(np.ones_like(a, dtype=bool), 1)
    return float(
        np.linalg.norm((a-b)[mask])
        / max(np.linalg.norm(b), 1e-40)
    )


def main():
    scene = build_demo_scene()
    full = build_sea_basis(scene, modes_per_face=2, face_refinement=1)
    # Two genuine face curls on each housing plus one genuine global curl.
    chosen = (0, 1, 12, 13, 24)
    basis = SeaCurrentBasis(
        scene, tuple(full.modes[i] for i in chosen),
        full.wall_scale, full.source_lines,
        full.outer_envelope, full.outer_taper_width,
    )
    if any(m.source_port >= 0 for m in basis.modes):
        raise ValueError("the Taylor comparison requires non-guided modes")
    print(
        "p1_full_sea_exact: physical_domain=outer_ball_minus_two_actual_boxes,"
        "modes=5,mode_indices=0-1-12-13-24,"
        "current=existing_box_following_curl_basis,"
        "domain_S=exact_true_rotated_geometry,"
        "method=symmetric_difference_unsoftened_1/r,"
        "comparison=global_affine_Taylor,"
        "gate=convergence_study_not_certified",
        flush=True,
    )
    # All x/y angle pairs have disjoint midpoint azimuth sets.
    x0, y0 = (3, 5, 8), (4, 6, 10)
    x1, y1 = (5, 7, 12), (6, 8, 14)
    trials = {}
    rules_cache = {}
    for name, xr, yr in (
        ("base", x0, y0),
        ("x_only", x1, y0),
        ("y_only", x0, y1),
        ("both", x1, y1),
    ):
        z = trial(basis, xr, yr, rules_cache, method="difference")
        trials[name] = z
        print(
            f"p1_full_sea: variant={name},method=difference,"
            f"x_nodes={z['nx']},y_nodes={z['ny']},"
            f"norm_L={z['norm']:.10g},trace_L={z['trace']:.10g},"
            f"subtraction_ratio={z['diagnostic']:.7g},"
            f"min_eig={z['min_eig']:.10g},"
            f"max_eig={z['max_eig']:.10g}",
            flush=True,
        )
        _mode_report(name, z["L"])
    base, fine = trials["base"]["L"], trials["both"]["L"]
    xo, yo = trials["x_only"]["L"], trials["y_only"]["L"]
    print(
        "p1_full_sea_change: method=difference,"
        f"x_only_vs_base={_relative(xo,base):.7g},"
        f"y_only_vs_base={_relative(yo,base):.7g},"
        f"both_vs_base={_relative(fine,base):.7g},"
        f"both_vs_x_only={_relative(fine,xo):.7g},"
        f"both_vs_y_only={_relative(fine,yo):.7g},"
        f"self_x={_diagonal_relative(xo,base):.7g},"
        f"self_y={_diagonal_relative(yo,base):.7g},"
        f"self_both={_diagonal_relative(fine,base):.7g},"
        f"mutual_x={_mutual_relative(xo,base):.7g},"
        f"mutual_y={_mutual_relative(yo,base):.7g},"
        f"mutual_both={_mutual_relative(fine,base):.7g},"
        "pass_status=not_determined_without_independent_truth",
        flush=True,
    )
    # ONE independent formulation at fixed highest tested x/y rules.
    # Agreement is supporting evidence, not continuous reference truth.
    old = trial(basis, x1, y1, rules_cache, method="taylor")
    print(
        "p1_full_sea_cross: variant=both,"
        "difference_vs_taylor_unsoftened=1/r,"
        f"relative_L={_relative(fine,old['L']):.7g},"
        f"self_relative={_diagonal_relative(fine,old['L']):.7g},"
        f"mutual_scaled={_mutual_relative(fine,old['L']):.7g},"
        f"taylor_raw_skew={old['diagnostic']:.7g},"
        f"taylor_min_eig={old['min_eig']:.10g}",
        flush=True,
    )
    print(
        "p1_full_sea_note: all five currents and both rotated insulating "
        "solids are unchanged across rules and formulations. The difference "
        "integral is symmetric by identity (raw_skew is not a gate); "
        "near-wall/edge angular x and y cubature, true reference L and "
        "infinite ocean remain unverified. Ball analytic tests alone "
        "cannot certify BFZI physical accuracy or teacher training.",
        flush=True,
    )


if __name__ == "__main__":
    main()
