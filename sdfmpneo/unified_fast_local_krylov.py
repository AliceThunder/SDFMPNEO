"""Maxwell-aware Krylov policy for the accelerated local-self solver.

The physical local operator is *not* shifted.  The added diagonal appears only
inside the ILU preconditioner.  It regularizes the curl-curl gradient near-null
space enough for bounded-fill ILU to be useful, while acceptance is still based
on the true residual of the original Maxwell matrix.

The main Krylov iteration can stagnate around 1e-6--1e-7 on the finest local
problems even though it has already produced an accurate field.  Rather than
falling back to a huge sparse LU, we perform true-residual defect correction:
for r=b-Ax, solve A*delta=r with the same preconditioner and update x+=delta.
Only the original unmodified Maxwell residual decides acceptance.
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import scipy.sparse.linalg as spla


def _defect_refine(
    A,
    rhs,
    field,
    M,
    solve,
    residual_fn,
    tolerance,
    *,
    steps=3,
    maxiter=16,
    inner_m=20,
    label="defect",
):
    """Iterative refinement using true residuals and the existing ILU.

    Each correction solve only needs enough *relative* accuracy to reduce the
    current true residual below the requested absolute relative certificate.
    This is much cheaper than asking one Krylov solve to span the whole dynamic
    range in one run on a nearly singular curl-curl operator.
    """
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

        # If the current global residual is rho, a correction solve with
        # relative residual eta leaves roughly rho*eta.  Aim for half the
        # certificate, but avoid oversolving the correction equation.
        eta = min(5e-2, max(1e-4, 0.5 * float(tolerance) / max(current, np.finfo(float).tiny)))
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
    """Patch solver policy without changing the physical operator/certificate."""

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

    def iterative_solve(A, rhs, x0, cfg, residual_tolerance):
        maxiter = int(cfg.get("linear_iterative_maxiter", 40))
        inner_m = int(cfg.get("linear_iterative_inner_m", 30))
        defect_steps = int(cfg.get("linear_iterative_defect_steps", 3))
        defect_maxiter = int(cfg.get("linear_iterative_defect_maxiter", 16))
        defect_inner_m = int(cfg.get("linear_iterative_defect_inner_m", 20))
        defect_start = float(cfg.get("linear_iterative_defect_start_residual", 5e-6))
        if maxiter < 1 or inner_m < 2 or defect_steps < 0 or defect_maxiter < 1 or defect_inner_m < 2:
            raise ValueError("local iterative Maxwell solver iteration limits are invalid")

        fast_shift = float(cfg.get("linear_ilu_shift_factor", 3e-2))
        strong_shift = float(cfg.get("linear_ilu_strong_shift_factor", 1e-1))
        attempts = (
            (
                float(cfg.get("linear_ilu_drop_tolerance", 5e-3)),
                float(cfg.get("linear_ilu_fill_factor", 4.0)),
                fast_shift,
                "shifted-ilu-fast",
            ),
            (
                float(cfg.get("linear_ilu_strong_drop_tolerance", 1e-3)),
                float(cfg.get("linear_ilu_strong_fill_factor", 8.0)),
                strong_shift,
                "shifted-ilu-strong",
            ),
            (
                float(cfg.get("linear_ilu_strong_drop_tolerance", 1e-3)),
                float(cfg.get("linear_ilu_strong_fill_factor", 8.0)),
                max(5e-3, 0.5 * fast_shift),
                "shifted-ilu-tight",
            ),
        )
        target = max(float(residual_tolerance) * 0.2, 1e-12)
        best = None if x0 is None else np.asarray(x0, complex).reshape(-1).copy()
        best_residual = (
            float("inf")
            if best is None
            else local_solver_module._relative_residual(A, best, rhs)
        )
        history = []

        for drop_tol, fill_factor, shift_factor, label in attempts:
            t0 = time.perf_counter()
            try:
                M = local_solver_module._ilu_preconditioner(
                    A,
                    drop_tol=drop_tol,
                    fill_factor=fill_factor,
                    shift_factor=shift_factor,
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
            history.append(
                {
                    "solver": label,
                    "drop_tolerance": float(drop_tol),
                    "fill_factor": float(fill_factor),
                    "shift_factor": float(shift_factor),
                    "krylov_info": int(info),
                    "preconditioner_seconds": float(t1 - t0),
                    "seconds": float(elapsed),
                    "relative_residual": float(residual),
                }
            )
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

            # Once the main solve is already in the few-ppm range, residual
            # correction is substantially cheaper than building another large
            # factorization or running a full extra Krylov solve from scratch.
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
    local_solver_module._maxwell_shifted_ilu_installed = True
    return local_solver_module


__all__ = ["_defect_refine", "install"]
