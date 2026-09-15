"""Compatible Krylov policy for large Cartesian Maxwell solves.

The production source may be an open two-terminal current and therefore may
have a genuine longitudinal component.  We solve that component with the exact
scalar-gradient block.  The remaining transverse correction is preconditioned
with a compatible grad-div stabilized ILU whose augmentation vanishes on the
exact transverse solution.  The original Maxwell matrix/RHS are unchanged and
every accepted field is certified with their true residual.
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import scipy.sparse.linalg as spla

from .unified_gradient_block_maxwell import (
    build_gradient_block,
    compose_block_preconditioner,
)
from .unified_transverse_ilu import build_transverse_ilu


def _defect_refine(
    A,
    rhs,
    field,
    M,
    solve,
    residual_fn,
    tolerance,
    *,
    steps=2,
    maxiter=12,
    inner_m=20,
    label="defect",
):
    """True-residual iterative refinement retained only for near-certified cleanup."""
    x = np.asarray(field, complex).reshape(-1).copy()
    rhs = np.asarray(rhs, complex).reshape(-1)
    current = float(residual_fn(A, x, rhs))
    history = []
    for k in range(int(steps)):
        if current <= tolerance:
            break
        defect = np.asarray(rhs - A @ x, complex).reshape(-1)
        if np.any(~np.isfinite(defect)):
            break
        eta = min(
            2e-2,
            max(1e-5, 0.25 * float(tolerance) / max(current, np.finfo(float).tiny)),
        )
        started = time.perf_counter()
        delta, info = solve(
            A,
            defect,
            x0=None,
            M=M,
            rtol=eta,
            maxiter=int(maxiter),
            inner_m=int(inner_m),
        )
        delta = np.asarray(delta, complex).reshape(-1)
        if np.any(~np.isfinite(delta)):
            break
        candidate = x + delta
        candidate_residual = float(residual_fn(A, candidate, rhs))
        elapsed = time.perf_counter() - started
        history.append(
            {
                "solver": f"{label}-{k+1}",
                "krylov_info": int(info),
                "correction_relative_target": float(eta),
                "seconds": float(elapsed),
                "relative_residual": candidate_residual,
            }
        )
        print(
            f"local Maxwell {label}-{k+1}: residual={candidate_residual:.3e}, "
            f"info={int(info)}, target={eta:.3e}, time={elapsed:.1f}s",
            flush=True,
        )
        if not np.isfinite(candidate_residual) or candidate_residual >= current:
            break
        x = candidate
        current = candidate_residual
    return x, current, history


def install(local_solver_module):
    """Patch the certified local solver with compatible block preconditioning."""

    def compatible_lgmres(A, rhs, *, x0, M, rtol, maxiter, inner_m):
        params = inspect.signature(spla.lgmres).parameters
        kwargs = {
            "x0": x0,
            "M": M,
            "maxiter": int(maxiter),
            "inner_m": int(inner_m),
            "outer_k": 3,
        }
        if "atol" in params:
            kwargs["atol"] = 0.0
        if "rtol" in params:
            kwargs["rtol"] = float(rtol)
        else:
            kwargs["tol"] = float(rtol)
        return spla.lgmres(A, rhs, **kwargs)

    def iterative_solve(
        A,
        rhs,
        x0,
        cfg,
        residual_tolerance,
        *,
        background=None,
        context=None,
        mqs=False,
        mqs_admittance=None,
    ):
        maxiter = int(cfg.get("linear_iterative_maxiter", 24))
        inner_m = int(cfg.get("linear_iterative_inner_m", 24))
        defect_steps = int(cfg.get("linear_iterative_defect_steps", 2))
        defect_maxiter = int(cfg.get("linear_iterative_defect_maxiter", 12))
        defect_inner_m = int(cfg.get("linear_iterative_defect_inner_m", 20))
        # Defect solves are useful only after the main compatible solve is
        # genuinely near certification.  Old configurations used 5e-6, which
        # causes expensive no-op cleanup on the 254k validation grid.
        defect_start = min(
            float(cfg.get("linear_iterative_defect_start_residual", 1e-8)),
            1e-8,
        )
        if maxiter < 1 or inner_m < 2 or defect_steps < 0:
            raise ValueError("local iterative Maxwell solver iteration limits are invalid")

        if background is None:
            background = getattr(A, "_sdfmpneo_background", None)
        if context is None:
            context = getattr(A, "_sdfmpneo_context", None)
        if bool(getattr(A, "_sdfmpneo_mqs", False)):
            mqs = True
            if mqs_admittance is None:
                mqs_admittance = getattr(A, "_sdfmpneo_mqs_admittance", None)

        gradient_block = None
        if background is not None and context is not None:
            gradient_block = build_gradient_block(
                background,
                context,
                mqs=bool(mqs),
                mqs_admittance=mqs_admittance,
                check_topology=True,
            )

        fast_drop = float(cfg.get("linear_ilu_drop_tolerance", 5e-3))
        fast_fill = float(cfg.get("linear_ilu_fill_factor", 4.0))
        strong_drop = float(cfg.get("linear_ilu_strong_drop_tolerance", 1e-3))
        strong_fill = float(cfg.get("linear_ilu_strong_fill_factor", 8.0))
        fast_stabilization = float(cfg.get("linear_transverse_stabilization_factor", 3e-2))
        strong_stabilization = float(
            cfg.get("linear_transverse_strong_stabilization_factor", 1e-1)
        )
        fast_shift = float(cfg.get("linear_ilu_shift_factor", 3e-2))
        strong_shift = float(cfg.get("linear_ilu_strong_shift_factor", 1e-1))

        if gradient_block is not None:
            attempts = (
                (
                    fast_drop,
                    fast_fill,
                    fast_stabilization,
                    "compatible-transverse-ilu-fast",
                    "transverse",
                ),
                (
                    strong_drop,
                    strong_fill,
                    strong_stabilization,
                    "compatible-transverse-ilu-strong",
                    "transverse",
                ),
            )
        else:
            attempts = (
                (fast_drop, fast_fill, fast_shift, "shifted-ilu-fast", "shifted"),
                (strong_drop, strong_fill, strong_shift, "shifted-ilu-strong", "shifted"),
            )

        target = max(float(residual_tolerance) * 0.2, 1e-12)
        best = None if x0 is None else np.asarray(x0, complex).reshape(-1).copy()
        best_residual = (
            float("inf")
            if best is None
            else local_solver_module._relative_residual(A, best, rhs)
        )
        history = []

        for drop_tol, fill_factor, stabilization, label, mode in attempts:
            t0 = time.perf_counter()
            preconditioner_stats = {}
            try:
                if mode == "transverse":
                    edge_M, preconditioner_stats = build_transverse_ilu(
                        A,
                        background,
                        context,
                        gradient_block,
                        drop_tol=drop_tol,
                        fill_factor=fill_factor,
                        stabilization_factor=stabilization,
                        mqs=bool(mqs),
                        mqs_admittance=mqs_admittance,
                    )
                else:
                    edge_M = local_solver_module._ilu_preconditioner(
                        A,
                        drop_tol=drop_tol,
                        fill_factor=fill_factor,
                        shift_factor=stabilization,
                    )
                M = (
                    compose_block_preconditioner(
                        A,
                        edge_M,
                        gradient_block,
                        post_correct=True,
                    )
                    if gradient_block is not None
                    else edge_M
                )
            except (RuntimeError, ValueError, MemoryError) as exc:
                history.append(
                    {
                        "solver": label,
                        "preconditioner_failed": True,
                        "error": type(exc).__name__,
                        "seconds": float(time.perf_counter() - t0),
                    }
                )
                continue

            t1 = time.perf_counter()
            candidate, info = compatible_lgmres(
                A,
                rhs,
                x0=best,
                M=M,
                rtol=target,
                maxiter=maxiter,
                inner_m=inner_m,
            )
            candidate = np.asarray(candidate, complex).reshape(-1)
            residual = local_solver_module._relative_residual(A, candidate, rhs)
            elapsed = time.perf_counter() - t0
            row = {
                "solver": label,
                "drop_tolerance": float(drop_tol),
                "fill_factor": float(fill_factor),
                "krylov_info": int(info),
                "preconditioner_seconds": float(t1 - t0),
                "gradient_scalar_dofs": (
                    0 if gradient_block is None else gradient_block.scalar_dofs
                ),
                "gradient_factor_seconds": (
                    0.0 if gradient_block is None else gradient_block.build_seconds
                ),
                "seconds": float(elapsed),
                "relative_residual": float(residual),
            }
            if mode == "transverse":
                row.update({f"transverse_{k}": v for k, v in preconditioner_stats.items()})
            else:
                row["shift_factor"] = float(stabilization)
            history.append(row)
            print(
                f"local Maxwell {label}: residual={residual:.3e}, "
                f"info={int(info)}, time={elapsed:.1f}s",
                flush=True,
            )
            if np.isfinite(residual) and residual < best_residual:
                best = candidate
                best_residual = residual
            if best is not None and best_residual <= residual_tolerance:
                return best, best_residual, history

            if (
                defect_steps > 0
                and best is not None
                and np.isfinite(best_residual)
                and best_residual <= defect_start
            ):
                refined, refined_residual, refinement_history = _defect_refine(
                    A,
                    rhs,
                    best,
                    M,
                    compatible_lgmres,
                    local_solver_module._relative_residual,
                    residual_tolerance,
                    steps=defect_steps,
                    maxiter=defect_maxiter,
                    inner_m=defect_inner_m,
                    label=f"{label}-defect",
                )
                history.extend(refinement_history)
                if np.isfinite(refined_residual) and refined_residual < best_residual:
                    best = refined
                    best_residual = refined_residual
                if best_residual <= residual_tolerance:
                    return best, best_residual, history

        return best, best_residual, history

    local_solver_module._lgmres = compatible_lgmres
    local_solver_module._iterative_solve = iterative_solve
    local_solver_module._defect_refine = _defect_refine
    local_solver_module._compatible_gradient_block_installed = True
    local_solver_module._compatible_transverse_ilu_installed = True
    return local_solver_module


__all__ = ["_defect_refine", "install"]
