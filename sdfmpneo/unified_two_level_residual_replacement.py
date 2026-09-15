"""Accurate residual-replacement cleanup for the finest local Maxwell solve.

The 254k-edge two-level solve can reduce the ordinary binary64 residual to the
O(1e-7) range and then stagnate.  At that scale a curl-curl matrix may suffer
severe cancellation in a conventional SciPy CSR ``b - A @ x`` evaluation.
This patch therefore performs mixed-precision iterative refinement:

* the main Maxwell/Krylov arithmetic and stored physical matrix remain complex128;
* the final defect is independently accumulated from the original A, x and b;
* on standard Windows builds this uses compensated double-double CSR arithmetic;
* the accurately accumulated defect is solved by the existing two-level map;
* every candidate is accepted only after another accurate residual evaluation.

No physical operator, source, or tolerance is modified.
"""
from __future__ import annotations

import time

import numpy as np

from .unified_accurate_residual import accurate_residual_vector


def _relative_norm(vector, rhs):
    return float(
        np.linalg.norm(np.asarray(vector, complex).reshape(-1))
        / max(float(np.linalg.norm(np.asarray(rhs, complex).reshape(-1))), np.finfo(float).tiny)
    )


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
        defect, diagnostics = accurate_residual_vector(
            A,
            x,
            rhs_value,
            target_relative=float(residual_tolerance),
        )
        current = _relative_norm(defect, rhs_value)
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
            maxiter = int(cfg.get("linear_two_level_residual_replacement_maxiter", 20))
            inner_m = int(cfg.get("linear_two_level_residual_replacement_inner_m", 30))
            backtracks = int(cfg.get("linear_two_level_residual_replacement_backtracks", 3))
            if steps < 1 or maxiter < 1 or inner_m < 2 or backtracks < 0:
                raise ValueError("two-level accurate residual replacement settings are invalid")

            for k in range(1, steps + 1):
                if current <= residual_tolerance:
                    return x, current, history
                eta = min(
                    2e-2,
                    max(
                        1e-6,
                        0.25
                        * float(residual_tolerance)
                        / max(current, np.finfo(float).tiny),
                    ),
                )
                started = time.perf_counter()
                delta, info = local_solver_module._lgmres(
                    A,
                    defect,
                    x0=None,
                    M=fine_M,
                    rtol=float(eta),
                    maxiter=maxiter,
                    inner_m=inner_m,
                )
                delta = np.asarray(delta, complex).reshape(-1)
                defect_norm = max(float(np.linalg.norm(defect)), np.finfo(float).tiny)
                correction_defect = np.asarray(defect - A @ delta, complex).reshape(-1)
                correction_relative = float(np.linalg.norm(correction_defect) / defect_norm)
                update_relative = float(
                    np.linalg.norm(delta)
                    / max(float(np.linalg.norm(x)), np.finfo(float).tiny)
                )

                best_candidate = None
                best_candidate_residual = float("inf")
                best_candidate_defect = None
                best_scale = 0.0
                for j in range(backtracks + 1):
                    scale = float(0.5**j)
                    candidate = x + scale * delta
                    candidate_defect, candidate_diag = accurate_residual_vector(
                        A,
                        candidate,
                        rhs_value,
                        target_relative=float(residual_tolerance),
                    )
                    candidate_residual = _relative_norm(candidate_defect, rhs_value)
                    if candidate_residual < best_candidate_residual:
                        best_candidate = candidate
                        best_candidate_residual = candidate_residual
                        best_candidate_defect = candidate_defect
                        best_scale = scale
                    if candidate_residual < current:
                        break

                elapsed = float(time.perf_counter() - started)
                accepted = bool(
                    best_candidate is not None
                    and np.isfinite(best_candidate_residual)
                    and best_candidate_residual < current
                )
                history.append({
                    "solver": f"two-level-accurate-residual-replacement-{k}",
                    "krylov_info": int(info),
                    "correction_relative_target": float(eta),
                    "correction_true_relative_residual": correction_relative,
                    "update_relative_norm": update_relative,
                    "accepted_scale": float(best_scale),
                    "seconds": elapsed,
                    "starting_relative_residual": current,
                    "relative_residual": float(best_candidate_residual),
                    "accepted": accepted,
                })
                print(
                    f"local Maxwell accurate residual replacement-{k}: "
                    f"residual={best_candidate_residual:.3e}, start={current:.3e}, "
                    f"correction_residual={correction_relative:.3e}, target={eta:.3e}, "
                    f"update={update_relative:.3e}, scale={best_scale:.3g}, info={int(info)}, "
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
                    f"local Maxwell accurate residual certified after refinement: residual={current:.3e}",
                    flush=True,
                )
                return x, current, history

        # Retain the matrix-free Galerkin correction as a fallback.  It still
        # cannot overwrite the best accurately certified fine-grid field unless
        # its returned residual is genuinely smaller.
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
            candidate_defect, _candidate_diag = accurate_residual_vector(
                A,
                candidate,
                rhs_value,
                target_relative=float(residual_tolerance),
            )
            accurate_candidate = _relative_norm(candidate_defect, rhs_value)
            if accurate_candidate < current:
                x = candidate
                current = accurate_candidate
        return x, current, history

    two_level_module._galerkin_defect_polish = residual_replacement_then_galerkin
    two_level_module._fine_residual_replacement_installed = True
    return two_level_module


__all__ = ["install"]
