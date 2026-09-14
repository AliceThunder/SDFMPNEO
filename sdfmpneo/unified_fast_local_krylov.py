"""Maxwell-aware Krylov policy for the accelerated local-self solver.

The physical local operator is *not* shifted.  The added diagonal appears only
inside the ILU preconditioner.  It regularizes the curl-curl gradient near-null
space enough for bounded-fill ILU to be useful, while acceptance is still based
on the true residual of the original Maxwell matrix.
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import scipy.sparse.linalg as spla


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
        if maxiter < 1 or inner_m < 2:
            raise ValueError("local iterative Maxwell solver iteration limits are invalid")

        # The shift is deliberately O(1e-2) of each row scale.  Tiny shifts do
        # not regularize the discrete gradient near-null space enough for ILU;
        # this value changes only the preconditioner, never A or the reported
        # physical solution.  Later attempts spend more fill only if necessary.
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
                max(1e-2, 0.5 * fast_shift),
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

        return best, best_residual, history

    local_solver_module._lgmres = compatible_lgmres
    local_solver_module._iterative_solve = iterative_solve
    local_solver_module._maxwell_shifted_ilu_installed = True
    return local_solver_module


__all__ = ["install"]
