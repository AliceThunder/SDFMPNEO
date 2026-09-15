"""Residual-replacement cleanup for the finest two-level local Maxwell solve.

The two-level H(curl) V-cycle can drive the 254k-edge validation problem from an
O(1) residual to O(1e-7), but a long restarted Krylov solve may then stagnate.
At that point the correct next operation is iterative refinement on the *fine*
physical equation, not another approximate coarse PDE solve:

    r = b - A_f x,
    A_f delta = r,
    x <- x + delta.

The correction solve uses the already successful two-level preconditioner and
is certified only by a freshly recomputed residual of the original fine
Maxwell matrix.  This module patches the Galerkin-polish entry point so residual
replacement is attempted first; the existing matrix-free Galerkin correction
remains a fallback when refinement does not finish the certificate.
"""
from __future__ import annotations

import numpy as np


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
        current = float(local_solver_module._relative_residual(A, x, rhs_value))
        history = []

        discrepancy = abs(current - reported) / max(
            current,
            reported,
            np.finfo(float).tiny,
        )
        print(
            "local Maxwell fine residual replacement entry: "
            f"recomputed={current:.3e}, reported={reported:.3e}, "
            f"discrepancy={discrepancy:.3e}",
            flush=True,
        )
        history.append({
            "solver": "two-level-fine-residual-replacement-entry",
            "reported_relative_residual": reported,
            "relative_residual": current,
            "reported_recomputed_discrepancy": float(discrepancy),
        })

        if current <= residual_tolerance:
            return x, current, history

        if fine_M is not None:
            refined, refined_residual, refinement_history = local_solver_module._defect_refine(
                A,
                rhs_value,
                x,
                fine_M,
                local_solver_module._lgmres,
                local_solver_module._relative_residual,
                residual_tolerance,
                steps=int(cfg.get("linear_two_level_residual_replacement_steps", 3)),
                maxiter=int(cfg.get("linear_two_level_residual_replacement_maxiter", 20)),
                inner_m=int(cfg.get("linear_two_level_residual_replacement_inner_m", 30)),
                label="two-level-hcurl-residual-replacement",
            )
            history.extend(refinement_history)
            if np.isfinite(refined_residual) and refined_residual < current:
                x = np.asarray(refined, complex).reshape(-1)
                current = float(refined_residual)
            if current <= residual_tolerance:
                return x, current, history

        # Retain the true Galerkin coarse correction as a fallback, but never
        # let a non-improving fallback overwrite the best fine-grid field.
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
            x = np.asarray(coarse_field, complex).reshape(-1)
            current = float(coarse_residual)
        return x, current, history

    two_level_module._galerkin_defect_polish = residual_replacement_then_galerkin
    two_level_module._fine_residual_replacement_installed = True
    return two_level_module


__all__ = ["install"]
