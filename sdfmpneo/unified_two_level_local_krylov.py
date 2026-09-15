"""Route the finest local Maxwell validation through a two-level H(curl) solve."""
from __future__ import annotations

import time

import numpy as np

from .unified_gradient_block_maxwell import build_gradient_block
from .unified_two_level_maxwell import build_two_level_maxwell


def _relative_pilot(A, rhs, x0, M, solve, residual_fn, *, maxiter, inner_m, accept_ratio):
    """Run a cheap pilot whose target is relative to the *current* residual."""
    start = np.asarray(x0, complex).reshape(-1)
    before = float(residual_fn(A, start, rhs))
    # SciPy measures rtol against ||rhs||, not against the residual at x0.
    # Therefore a fixed rtol=0.5 would immediately accept any warm start whose
    # true relative residual is already <0.5.  Ask instead for roughly a factor
    # two reduction from the actual starting point.
    pilot_rtol = max(min(0.5, 0.5 * before), 1e-12)
    started = time.perf_counter()
    candidate, info = solve(
        A,
        rhs,
        x0=start,
        M=M,
        rtol=pilot_rtol,
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
    return candidate, after, int(info), elapsed, before, accepted, pilot_rtol


def install(local_solver_module):
    original_iterative = local_solver_module._iterative_solve

    def iterative_solve(A, rhs, x0, cfg, residual_tolerance, **kwargs):
        background = kwargs.get("background") or getattr(A, "_sdfmpneo_background", None)
        context = kwargs.get("context") or getattr(A, "_sdfmpneo_context", None)
        minimum_dofs = int(cfg.get("linear_two_level_min_dofs", 200000))
        coarse_state = None if background is None else getattr(background, "_sdfmpneo_coarse_state", None)
        if (
            A.shape[0] < minimum_dofs
            or background is None
            or context is None
            or not isinstance(coarse_state, dict)
            or coarse_state.get("prolongation") is None
        ):
            return original_iterative(A, rhs, x0, cfg, residual_tolerance, **kwargs)

        rhs = np.asarray(rhs, complex).reshape(-1)
        target = max(float(residual_tolerance) * 0.2, 1e-12)
        maxiter = int(cfg.get("linear_iterative_maxiter", 40))
        inner_m = int(cfg.get("linear_iterative_inner_m", 30))
        pilot_maxiter = int(cfg.get("linear_two_level_pilot_maxiter", 2))
        pilot_inner_m = int(cfg.get("linear_two_level_pilot_inner_m", 8))
        pilot_accept_ratio = float(cfg.get("linear_two_level_pilot_accept_ratio", 0.8))
        if maxiter < 1 or inner_m < 2 or pilot_maxiter < 1 or pilot_inner_m < 2:
            raise ValueError("two-level Maxwell iteration limits are invalid")
        if not 0.0 < pilot_accept_ratio < 1.0:
            raise ValueError("linear_two_level_pilot_accept_ratio must lie in (0,1)")

        zero = np.zeros_like(rhs)
        best = zero.copy()
        best_residual = 1.0
        history = []
        if x0 is not None:
            warm = np.asarray(x0, complex).reshape(-1)
            warm_residual = float(local_solver_module._relative_residual(A, warm, rhs))
            if np.isfinite(warm_residual) and warm_residual < 1.0:
                best = warm.copy()
                best_residual = warm_residual
                print(
                    f"local Maxwell compatible H(curl) warm start accepted: residual={warm_residual:.3e}",
                    flush=True,
                )
            else:
                print(
                    "local Maxwell compatible H(curl) warm start rejected: "
                    f"residual={warm_residual:.3e}, zero_baseline=1.000e+00",
                    flush=True,
                )
                history.append({
                    "solver": "hcurl-warm-start-screen",
                    "relative_residual": warm_residual,
                    "accepted": False,
                })

        build_started = time.perf_counter()
        fine_gradient = build_gradient_block(background, context, check_topology=True)
        two_level = build_two_level_maxwell(
            A,
            background,
            fine_gradient,
            coarse_state,
            cfg,
        )
        history.append({
            "solver": "two-level-hcurl-build",
            "seconds": float(time.perf_counter() - build_started),
            "coarse_edges": int(two_level.coarse_edges),
            "fine_edges": int(A.shape[0]),
            "prolongation_nnz": int(two_level.prolongation.nnz),
            "coarse_consistency_error": float(two_level.consistency_error),
        })

        attempts = (
            (1, 0, "two-level-hcurl-coarse-only"),
            (1, 1, "two-level-hcurl-fast"),
            (2, 1, "two-level-hcurl-coarse2"),
            (2, 2, "two-level-hcurl-strong"),
        )
        last_M = None
        for coarse_corrections, smoother_sweeps, label in attempts:
            M = two_level.operator(
                coarse_corrections=coarse_corrections,
                smoother_sweeps=smoother_sweeps,
            )
            last_M = M
            pilot, pilot_residual, pilot_info, pilot_seconds, before, accepted, pilot_rtol = (
                _relative_pilot(
                    A,
                    rhs,
                    best,
                    M,
                    local_solver_module._lgmres,
                    local_solver_module._relative_residual,
                    maxiter=pilot_maxiter,
                    inner_m=pilot_inner_m,
                    accept_ratio=pilot_accept_ratio,
                )
            )
            history.append({
                "solver": f"{label}-pilot",
                "coarse_corrections": int(coarse_corrections),
                "smoother_sweeps": int(smoother_sweeps),
                "krylov_info": int(pilot_info),
                "pilot_rtol": float(pilot_rtol),
                "seconds": float(pilot_seconds),
                "starting_relative_residual": float(before),
                "relative_residual": float(pilot_residual),
                "accepted": bool(accepted),
            })
            print(
                f"local Maxwell {label}-pilot: residual={pilot_residual:.3e}, "
                f"start={before:.3e}, target={pilot_rtol:.3e}, info={pilot_info}, "
                f"accepted={'yes' if accepted else 'no'}, time={pilot_seconds:.1f}s",
                flush=True,
            )
            if not accepted:
                continue
            if pilot_residual < best_residual:
                best = np.asarray(pilot, complex).reshape(-1)
                best_residual = float(pilot_residual)
            if best_residual <= residual_tolerance:
                return best, best_residual, history

            started = time.perf_counter()
            candidate, info = local_solver_module._lgmres(
                A,
                rhs,
                x0=best,
                M=M,
                rtol=target,
                maxiter=maxiter,
                inner_m=inner_m,
            )
            candidate = np.asarray(candidate, complex).reshape(-1)
            residual = float(local_solver_module._relative_residual(A, candidate, rhs))
            elapsed = float(time.perf_counter() - started)
            history.append({
                "solver": label,
                "coarse_corrections": int(coarse_corrections),
                "smoother_sweeps": int(smoother_sweeps),
                "krylov_info": int(info),
                "seconds": elapsed,
                "relative_residual": residual,
            })
            print(
                f"local Maxwell {label}: residual={residual:.3e}, info={int(info)}, "
                f"time={elapsed:.1f}s",
                flush=True,
            )
            if np.isfinite(residual) and residual < best_residual:
                best = candidate
                best_residual = residual
            if best_residual <= residual_tolerance:
                return best, best_residual, history

        if last_M is not None and best_residual <= 1e-8:
            refined, refined_residual, extra = local_solver_module._defect_refine(
                A,
                rhs,
                best,
                last_M,
                local_solver_module._lgmres,
                local_solver_module._relative_residual,
                residual_tolerance,
                steps=int(cfg.get("linear_iterative_defect_steps", 2)),
                maxiter=int(cfg.get("linear_iterative_defect_maxiter", 12)),
                inner_m=int(cfg.get("linear_iterative_defect_inner_m", 20)),
                label="two-level-hcurl-defect",
            )
            history.extend(extra)
            if refined_residual < best_residual:
                best = refined
                best_residual = refined_residual
            if best_residual <= residual_tolerance:
                return best, best_residual, history

        raise RuntimeError(
            "two-level H(curl) local Maxwell solve did not reach the certified residual; "
            f"fine_edges={A.shape[0]}, coarse_edges={two_level.coarse_edges}, "
            f"best_residual={best_residual:.3e}, tolerance={float(residual_tolerance):.3e}, "
            f"coarse_consistency={two_level.consistency_error:.3e}. "
            "The 254k one-level ILU fallback is intentionally disabled."
        )

    local_solver_module._iterative_solve = iterative_solve
    local_solver_module._two_level_hcurl_local_installed = True
    return local_solver_module


__all__ = ["_relative_pilot", "install"]
