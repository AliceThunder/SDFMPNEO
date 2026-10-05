"""Compatible Krylov policy for large Cartesian Maxwell solves.

The production source may be an open two-terminal current and therefore may
have a genuine longitudinal component.  We solve that component with the exact
scalar-gradient block.  The remaining transverse correction is preconditioned
with a compatible grad-div stabilized ILU whose augmentation vanishes on the
exact transverse solution.  The original Maxwell matrix/RHS are unchanged and
every accepted field is certified with their true residual.

On the 2.25-mm validation grid two additional safeguards are essential:
coarse-to-fine interpolation is rejected when it is algebraically much worse
than the zero field, and an expensive full Krylov run is started only after a
small pilot Krylov cycle demonstrates that the chosen ILU actually reduces the
true residual.  A SuperLU factorization that merely exists is not automatically
accepted as a useful Maxwell preconditioner.
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


def _pilot_krylov(
    A,
    rhs,
    x0,
    M,
    solve,
    residual_fn,
    *,
    maxiter=2,
    inner_m=8,
    accept_ratio=0.95,
):
    """Cheaply reject a numerically destructive large-grid preconditioner."""
    start = np.asarray(x0, complex).reshape(-1)
    before = float(residual_fn(A, start, rhs))
    started = time.perf_counter()
    candidate, info = solve(
        A,
        rhs,
        x0=start,
        M=M,
        rtol=0.5,
        maxiter=int(maxiter),
        inner_m=int(inner_m),
    )
    candidate = np.asarray(candidate, complex).reshape(-1)
    after = float(residual_fn(A, candidate, rhs))
    elapsed = float(time.perf_counter() - started)
    accepted = bool(
        np.isfinite(after)
        and np.all(np.isfinite(candidate))
        and after < before * float(accept_ratio)
    )
    return candidate, after, int(info), elapsed, before, accepted


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
        defect_start = min(
            float(cfg.get("linear_iterative_defect_start_residual", 1e-8)),
            1e-8,
        )
        pilot_min_dofs = int(cfg.get("linear_preconditioner_pilot_min_dofs", 200000))
        pilot_maxiter = int(cfg.get("linear_preconditioner_pilot_maxiter", 2))
        pilot_inner_m = int(cfg.get("linear_preconditioner_pilot_inner_m", 8))
        pilot_accept_ratio = float(cfg.get("linear_preconditioner_pilot_accept_ratio", 0.95))
        warm_limit = float(cfg.get("linear_warm_start_max_relative_residual", 10.0))
        if (
            maxiter < 1
            or inner_m < 2
            or defect_steps < 0
            or pilot_maxiter < 1
            or pilot_inner_m < 2
            or not 0.0 < pilot_accept_ratio < 1.0
            or warm_limit <= 0.0
        ):
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
        zero = np.zeros_like(np.asarray(rhs, complex).reshape(-1))
        best = zero.copy()
        best_residual = 1.0
        history = []
        if x0 is not None:
            warm = np.asarray(x0, complex).reshape(-1).copy()
            warm_residual = float(local_solver_module._relative_residual(A, warm, rhs))
            if np.isfinite(warm_residual) and warm_residual <= warm_limit:
                best = warm
                best_residual = warm_residual
            else:
                print(
                    "local Maxwell warm start discarded: "
                    f"residual={warm_residual:.3e}, zero_baseline=1.000e+00, "
                    f"limit={warm_limit:.3e}",
                    flush=True,
                )
                history.append(
                    {
                        "solver": "warm-start-screen",
                        "discarded": True,
                        "relative_residual": warm_residual,
                        "zero_baseline": 1.0,
                    }
                )

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

            run_start = best
            if A.shape[0] >= pilot_min_dofs:
                pilot, pilot_residual, pilot_info, pilot_seconds, before, accepted = _pilot_krylov(
                    A,
                    rhs,
                    best,
                    M,
                    compatible_lgmres,
                    local_solver_module._relative_residual,
                    maxiter=pilot_maxiter,
                    inner_m=pilot_inner_m,
                    accept_ratio=pilot_accept_ratio,
                )
                history.append(
                    {
                        "solver": f"{label}-pilot",
                        "krylov_info": int(pilot_info),
                        "seconds": float(pilot_seconds),
                        "starting_relative_residual": float(before),
                        "relative_residual": float(pilot_residual),
                        "accepted": bool(accepted),
                    }
                )
                print(
                    f"local Maxwell {label}-pilot: residual={pilot_residual:.3e}, "
                    f"start={before:.3e}, info={pilot_info}, "
                    f"accepted={'yes' if accepted else 'no'}, time={pilot_seconds:.1f}s",
                    flush=True,
                )
                if not accepted:
                    continue
                run_start = pilot
                if pilot_residual < best_residual:
                    best = pilot
                    best_residual = pilot_residual
                if best_residual <= residual_tolerance:
                    return best, best_residual, history

            t1 = time.perf_counter()
            candidate, info = compatible_lgmres(
                A,
                rhs,
                x0=run_start,
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
            if best_residual <= residual_tolerance:
                return best, best_residual, history

            if (
                defect_steps > 0
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
    local_solver_module._pilot_krylov = _pilot_krylov
    local_solver_module._compatible_gradient_block_installed = True
    local_solver_module._compatible_transverse_ilu_installed = True
    return local_solver_module


__all__ = ["_defect_refine", "_pilot_krylov", "install"]
