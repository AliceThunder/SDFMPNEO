"""Accurate, equilibrated residual replacement for the finest Maxwell solve.

The 254k-edge two-level solve can reduce the residual to O(1e-7) and then
stagnate because the raw curl-curl equation has a very large numerical dynamic
range.  This patch performs mixed-precision iterative refinement without
changing the physical equation:

* form the fine residual accurately from the original complex128 A, x and b;
* Ruiz-equilibrate only the correction equation ``A delta = r``;
* map the existing two-level preconditioner into the scaled coordinates;
* map the correction back to the original field coordinates;
* certify both the correction equation and every updated field against the
  original unscaled physical matrix with compensated residual accumulation.

No physical operator, source, or tolerance is modified.
"""
from __future__ import annotations

import time

import numpy as np

from .unified_accurate_residual import accurate_residual_vector
from .unified_equilibrated_defect import build_equilibrated_defect_system


def _relative_norm(vector, rhs):
    return float(
        np.linalg.norm(np.asarray(vector, complex).reshape(-1))
        / max(float(np.linalg.norm(np.asarray(rhs, complex).reshape(-1))), np.finfo(float).tiny)
    )


def _accurate_relative_residual(A, field, rhs, target):
    residual, diagnostics = accurate_residual_vector(
        A,
        field,
        rhs,
        target_relative=float(target),
    )
    return residual, _relative_norm(residual, rhs), diagnostics


def install(two_level_module, local_solver_module):
    if bool(getattr(two_level_module, "_fine_residual_replacement_installed", False)):
        return two_level_module

    original = two_level_module._galerkin_defect_polish

    def residual_replacement_then_galerkin(
        local_solver_module_arg,
        two_level,
        A,
        rhs,
        field,
        field_residual,
        fine_M,
        residual_tolerance,
        cfg,
    ):
        x = np.asarray(field, complex).reshape(-1).copy()
        rhs_value = np.asarray(rhs, complex).reshape(-1)
        reported = float(field_residual)
        standard = float(local_solver_module._relative_residual(A, x, rhs_value))
        defect, current, diagnostics = _accurate_relative_residual(
            A, x, rhs_value, residual_tolerance
        )
        history = []

        discrepancy = abs(current - reported) / max(
            current,
            reported,
            np.finfo(float).tiny,
        )
        print(
            "local Maxwell accurate residual entry: "
            f"standard={standard:.3e}, accurate={current:.3e}, reported={reported:.3e}, "
            f"roundoff_bound={diagnostics['roundoff_bound_relative']:.3e}, "
            f"dynamic={diagnostics['action_rhs_dynamic_range']:.3e}, "
            f"mode={diagnostics['accumulation_mode']}, discrepancy={discrepancy:.3e}",
            flush=True,
        )
        history.append({
            "solver": "two-level-accurate-residual-entry",
            "reported_relative_residual": reported,
            "standard_relative_residual": standard,
            "relative_residual": current,
            "roundoff_bound_relative": float(diagnostics["roundoff_bound_relative"]),
            "action_rhs_dynamic_range": float(diagnostics["action_rhs_dynamic_range"]),
            "maximum_row_cancellation_ratio": float(
                diagnostics["maximum_row_cancellation_ratio"]
            ),
            "accumulation_mode": str(diagnostics["accumulation_mode"]),
            "reported_accurate_discrepancy": float(discrepancy),
        })

        if current <= residual_tolerance:
            print(
                f"local Maxwell accurate residual certified: residual={current:.3e}",
                flush=True,
            )
            return x, current, history

        if fine_M is not None:
            steps = int(cfg.get("linear_two_level_residual_replacement_steps", 3))
            maxiter = int(cfg.get("linear_two_level_residual_replacement_maxiter", 24))
            inner_m = int(cfg.get("linear_two_level_residual_replacement_inner_m", 36))
            backtracks = int(cfg.get("linear_two_level_residual_replacement_backtracks", 4))
            equilibration_iterations = int(cfg.get("linear_two_level_equilibration_iterations", 4))
            scaled_attempts = int(cfg.get("linear_two_level_equilibrated_attempts", 2))
            if (
                steps < 1
                or maxiter < 1
                or inner_m < 2
                or backtracks < 0
                or equilibration_iterations < 1
                or scaled_attempts < 1
            ):
                raise ValueError("two-level accurate residual replacement settings are invalid")

            equilibrated = build_equilibrated_defect_system(
                A,
                defect,
                fine_M,
                iterations=equilibration_iterations,
                scale_limit=float(cfg.get("linear_two_level_equilibration_scale_limit", 1e12)),
            )
            print(
                "local Maxwell defect equilibration: "
                f"row_span={equilibrated.row_span_before:.3e}->{equilibrated.row_span_after:.3e}, "
                f"col_span={equilibrated.column_span_before:.3e}->{equilibrated.column_span_after:.3e}, "
                f"R_span={equilibrated.row_scale_span:.3e}, C_span={equilibrated.column_scale_span:.3e}",
                flush=True,
            )
            history.append({
                "solver": "two-level-defect-equilibration",
                "row_span_before": equilibrated.row_span_before,
                "row_span_after": equilibrated.row_span_after,
                "column_span_before": equilibrated.column_span_before,
                "column_span_after": equilibrated.column_span_after,
                "row_scale_span": equilibrated.row_scale_span,
                "column_scale_span": equilibrated.column_scale_span,
            })

            for k in range(1, steps + 1):
                if current <= residual_tolerance:
                    return x, current, history

                eta = min(
                    2e-2,
                    max(
                        5e-7,
                        0.20
                        * float(residual_tolerance)
                        / max(current, np.finfo(float).tiny),
                    ),
                )
                scaled_rhs = equilibrated.row_scale * defect
                delta = None
                info = -1
                correction_relative = float("inf")
                correction_diag = None
                scaled_rtol_used = float("nan")
                solve_started = time.perf_counter()

                # A scaled Krylov stopping rule is only a candidate certificate.
                # Verify A*delta=r in the original coordinates, and retry with a
                # tighter scaled target if necessary.
                for attempt in range(scaled_attempts):
                    scaled_rtol = max(
                        1e-10,
                        min(5e-3, 0.25 * eta) * (0.1**attempt),
                    )
                    y, info = local_solver_module._lgmres(
                        equilibrated.operator,
                        scaled_rhs,
                        x0=None,
                        M=equilibrated.preconditioner,
                        rtol=float(scaled_rtol),
                        maxiter=maxiter,
                        inner_m=inner_m,
                    )
                    candidate_delta = equilibrated.physical_correction(y)
                    correction_defect, correction_relative, correction_diag = (
                        _accurate_relative_residual(
                            A,
                            candidate_delta,
                            defect,
                            eta,
                        )
                    )
                    delta = np.asarray(candidate_delta, complex).reshape(-1)
                    scaled_rtol_used = float(scaled_rtol)
                    if np.isfinite(correction_relative) and correction_relative <= eta:
                        break

                elapsed_solve = float(time.perf_counter() - solve_started)
                if delta is None or np.any(~np.isfinite(delta)):
                    break

                update_relative = float(
                    np.linalg.norm(delta)
                    / max(float(np.linalg.norm(x)), np.finfo(float).tiny)
                )

                best_candidate = None
                best_candidate_residual = float("inf")
                best_candidate_defect = None
                best_scale = 0.0
                candidate_mode = "unknown"
                for j in range(backtracks + 1):
                    scale = float(0.5**j)
                    candidate = x + scale * delta
                    candidate_defect, candidate_residual, candidate_diag = (
                        _accurate_relative_residual(
                            A,
                            candidate,
                            rhs_value,
                            residual_tolerance,
                        )
                    )
                    if candidate_residual < best_candidate_residual:
                        best_candidate = candidate
                        best_candidate_residual = candidate_residual
                        best_candidate_defect = candidate_defect
                        best_scale = scale
                        candidate_mode = str(candidate_diag["accumulation_mode"])
                    if candidate_residual < current:
                        break

                elapsed = float(time.perf_counter() - solve_started)
                accepted = bool(
                    best_candidate is not None
                    and np.isfinite(best_candidate_residual)
                    and best_candidate_residual < current
                )
                correction_certified = bool(
                    np.isfinite(correction_relative) and correction_relative <= eta
                )
                history.append({
                    "solver": f"two-level-equilibrated-residual-replacement-{k}",
                    "krylov_info": int(info),
                    "scaled_krylov_rtol": scaled_rtol_used,
                    "correction_relative_target": float(eta),
                    "correction_true_relative_residual": correction_relative,
                    "correction_certified": correction_certified,
                    "update_relative_norm": update_relative,
                    "accepted_scale": float(best_scale),
                    "seconds": elapsed,
                    "starting_relative_residual": current,
                    "relative_residual": float(best_candidate_residual),
                    "accepted": accepted,
                    "accumulation_mode": candidate_mode,
                    "correction_roundoff_bound_relative": (
                        float(correction_diag["roundoff_bound_relative"])
                        if correction_diag is not None else float("inf")
                    ),
                })
                print(
                    f"local Maxwell equilibrated residual replacement-{k}: "
                    f"residual={best_candidate_residual:.3e}, start={current:.3e}, "
                    f"correction_residual={correction_relative:.3e}, target={eta:.3e}, "
                    f"scaled_rtol={scaled_rtol_used:.1e}, update={update_relative:.3e}, "
                    f"scale={best_scale:.3g}, info={int(info)}, "
                    f"correction_certified={'yes' if correction_certified else 'no'}, "
                    f"accepted={'yes' if accepted else 'no'}, time={elapsed:.1f}s",
                    flush=True,
                )
                if not accepted:
                    break
                x = np.asarray(best_candidate, complex).reshape(-1)
                current = float(best_candidate_residual)
                defect = np.asarray(best_candidate_defect, complex).reshape(-1)

            if current <= residual_tolerance:
                print(
                    f"local Maxwell accurate residual certified after equilibration: residual={current:.3e}",
                    flush=True,
                )
                return x, current, history

        # Retain the matrix-free Galerkin correction as a fallback.  It cannot
        # overwrite the best accurately certified fine-grid field unless a fresh
        # compensated residual check confirms an actual improvement.
        coarse_field, coarse_residual, coarse_history = original(
            local_solver_module_arg,
            two_level,
            A,
            rhs_value,
            x,
            current,
            fine_M,
            residual_tolerance,
            cfg,
        )
        history.extend(coarse_history)
        if np.isfinite(coarse_residual) and coarse_residual < current:
            candidate = np.asarray(coarse_field, complex).reshape(-1)
            candidate_defect, accurate_candidate, _candidate_diag = (
                _accurate_relative_residual(
                    A,
                    candidate,
                    rhs_value,
                    residual_tolerance,
                )
            )
            if accurate_candidate < current:
                x = candidate
                current = accurate_candidate
        return x, current, history

    two_level_module._galerkin_defect_polish = residual_replacement_then_galerkin
    two_level_module._fine_residual_replacement_installed = True
    return two_level_module


__all__ = ["install"]
