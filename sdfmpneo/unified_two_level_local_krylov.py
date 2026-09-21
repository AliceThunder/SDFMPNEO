"""Route the finest local Maxwell validation through a two-level H(curl) solve."""
from __future__ import annotations

import time

import numpy as np

from .unified_gradient_block_maxwell import build_gradient_block
from .unified_two_level_maxwell import build_two_level_maxwell


def _two_level_minimum_dofs(cfg):
    """First local size that must use the coarse H(curl) hierarchy by default.

    Keep this boundary adjacent to the localized transverse direct band.  The
    previous hard-coded 200k threshold left a 100k--200k one-level ILU/LGMRES
    gap; production geometry sampling can land in that gap (for example 108327
    edges) and then make essentially no residual progress.
    """
    if "linear_two_level_min_dofs" in cfg:
        minimum = int(cfg["linear_two_level_min_dofs"])
    else:
        direct = int(
            cfg.get(
                "linear_transverse_direct_max_dofs",
                cfg.get("linear_direct_max_dofs", 60000),
            )
        )
        minimum = direct + 1
    if minimum < 1:
        raise ValueError("linear_two_level_min_dofs must be positive")
    return minimum


def _relative_pilot(A, rhs, x0, M, solve, residual_fn, *, maxiter, inner_m, accept_ratio):
    """Run a cheap pilot whose target is relative to the *current* residual."""
    start = np.asarray(x0, complex).reshape(-1)
    before = float(residual_fn(A, start, rhs))
    # SciPy measures rtol against ||rhs||, not against the residual at x0.
    # Therefore a fixed rtol=0.5 would immediately accept any warm start whose
    # true relative residual is already <0.5. Ask instead for roughly a factor
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


def _galerkin_defect_polish(
    local_solver_module,
    two_level,
    A,
    rhs,
    field,
    field_residual,
    fine_M,
    residual_tolerance,
    cfg,
):
    """Remove the rediscretization plateau with the true Galerkin coarse equation.

    This is an outer defect-correction stage, not a nested nonlinear
    preconditioner.  The 118k rediscretized compatible block preconditions a
    matrix-free solve of ``P.T A_f P``.  The prolongated coarse correction is
    line-searched against the original fine residual, then a short fine
    two-level LGMRES polish removes the high-frequency remainder.
    """
    x = np.asarray(field, complex).reshape(-1).copy()
    current = float(field_residual)
    rhs = np.asarray(rhs, complex).reshape(-1)
    P = two_level.prolongation
    Ac = two_level.galerkin_operator()
    Mc = two_level.galerkin_preconditioner()

    cycles = int(cfg.get("linear_two_level_galerkin_polish_cycles", 3))
    coarse_rtol = float(cfg.get("linear_two_level_galerkin_coarse_rtol", 1e-4))
    coarse_maxiter = int(cfg.get("linear_two_level_galerkin_coarse_maxiter", 8))
    coarse_inner_m = int(cfg.get("linear_two_level_galerkin_coarse_inner_m", 20))
    fine_maxiter = int(cfg.get("linear_two_level_galerkin_fine_maxiter", 12))
    fine_inner_m = int(cfg.get("linear_two_level_galerkin_fine_inner_m", 20))
    if (
        cycles < 1
        or not 0.0 < coarse_rtol < 1.0
        or coarse_maxiter < 1
        or coarse_inner_m < 2
        or fine_maxiter < 1
        or fine_inner_m < 2
    ):
        raise ValueError("two-level Galerkin polish settings are invalid")

    fine_target = max(float(residual_tolerance) * 0.2, 1e-12)
    history = []
    tiny = np.finfo(float).tiny

    for cycle in range(1, cycles + 1):
        fine_defect = np.asarray(rhs - A @ x, complex).reshape(-1)
        coarse_rhs = np.asarray(P.T @ fine_defect, complex).reshape(-1)
        coarse_norm = float(np.linalg.norm(coarse_rhs))
        if not np.isfinite(coarse_norm) or coarse_norm <= tiny:
            break

        started = time.perf_counter()
        coarse_delta, coarse_info = local_solver_module._lgmres(
            Ac,
            coarse_rhs,
            x0=None,
            M=Mc,
            rtol=coarse_rtol,
            maxiter=coarse_maxiter,
            inner_m=coarse_inner_m,
        )
        coarse_delta = np.asarray(coarse_delta, complex).reshape(-1)
        coarse_relative = float(
            np.linalg.norm(coarse_rhs - Ac @ coarse_delta) / max(coarse_norm, tiny)
        )
        correction = np.asarray(P @ coarse_delta, complex).reshape(-1)
        action = np.asarray(A @ correction, complex).reshape(-1)
        denominator = complex(np.vdot(action, action))
        if (
            np.any(~np.isfinite(coarse_delta))
            or np.any(~np.isfinite(correction))
            or not np.isfinite(coarse_relative)
            or abs(denominator) <= tiny
        ):
            break

        # Exact one-dimensional least-squares minimizer for
        # ||fine_defect - alpha * A*correction||_2.  This makes acceptance
        # independent of the inner coarse solve's stopping details.
        alpha = complex(np.vdot(action, fine_defect) / denominator)
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
            break
        candidate = x + alpha * correction
        candidate_residual = float(local_solver_module._relative_residual(A, candidate, rhs))
        elapsed = float(time.perf_counter() - started)
        accepted = bool(np.isfinite(candidate_residual) and candidate_residual < current)
        history.append({
            "solver": f"two-level-galerkin-coarse-defect-{cycle}",
            "krylov_info": int(coarse_info),
            "coarse_relative_residual": coarse_relative,
            "alpha_real": float(alpha.real),
            "alpha_imag": float(alpha.imag),
            "seconds": elapsed,
            "starting_relative_residual": current,
            "relative_residual": candidate_residual,
            "accepted": accepted,
        })
        print(
            f"local Maxwell two-level Galerkin coarse defect-{cycle}: "
            f"fine_residual={candidate_residual:.3e}, start={current:.3e}, "
            f"coarse_residual={coarse_relative:.3e}, info={int(coarse_info)}, "
            f"alpha={alpha.real:.3e}{alpha.imag:+.3e}j, "
            f"accepted={'yes' if accepted else 'no'}, time={elapsed:.1f}s",
            flush=True,
        )
        if not accepted:
            break
        x = candidate
        current = candidate_residual
        if current <= residual_tolerance:
            return x, current, history

        if fine_M is None:
            continue
        started = time.perf_counter()
        polished, polish_info = local_solver_module._lgmres(
            A,
            rhs,
            x0=x,
            M=fine_M,
            rtol=fine_target,
            maxiter=fine_maxiter,
            inner_m=fine_inner_m,
        )
        polished = np.asarray(polished, complex).reshape(-1)
        polished_residual = float(local_solver_module._relative_residual(A, polished, rhs))
        elapsed = float(time.perf_counter() - started)
        polish_accepted = bool(np.isfinite(polished_residual) and polished_residual < current)
        history.append({
            "solver": f"two-level-galerkin-fine-polish-{cycle}",
            "krylov_info": int(polish_info),
            "seconds": elapsed,
            "starting_relative_residual": current,
            "relative_residual": polished_residual,
            "accepted": polish_accepted,
        })
        print(
            f"local Maxwell two-level Galerkin fine polish-{cycle}: "
            f"residual={polished_residual:.3e}, start={current:.3e}, "
            f"info={int(polish_info)}, accepted={'yes' if polish_accepted else 'no'}, "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        if polish_accepted:
            x = polished
            current = polished_residual
        if current <= residual_tolerance:
            return x, current, history

    return x, current, history


def install(local_solver_module):
    original_iterative = local_solver_module._iterative_solve

    def iterative_solve(A, rhs, x0, cfg, residual_tolerance, **kwargs):
        background = kwargs.get("background") or getattr(A, "_sdfmpneo_background", None)
        context = kwargs.get("context") or getattr(A, "_sdfmpneo_context", None)
        minimum_dofs = _two_level_minimum_dofs(cfg)
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
        galerkin_start = float(cfg.get("linear_two_level_galerkin_polish_start_residual", 1e-4))
        if maxiter < 1 or inner_m < 2 or pilot_maxiter < 1 or pilot_inner_m < 2:
            raise ValueError("two-level Maxwell iteration limits are invalid")
        if not 0.0 < pilot_accept_ratio < 1.0 or galerkin_start <= 0.0:
            raise ValueError("two-level Maxwell screening settings are invalid")

        zero = np.zeros_like(rhs)
        best = zero.copy()
        best_residual = 1.0
        best_M = None
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

        # Coarse-only is useful as a cheap low-frequency pilot but the production
        # log showed that running it for all 40 outer iterations costs minutes
        # and barely improves the field.  Only V-cycles with fine smoothing get
        # a full solve budget.
        attempts = (
            (1, 0, "two-level-hcurl-coarse-only", False),
            (1, 1, "two-level-hcurl-fast", True),
            (2, 1, "two-level-hcurl-coarse2", True),
            (2, 2, "two-level-hcurl-strong", True),
        )
        galerkin_polished = False
        for coarse_corrections, smoother_sweeps, label, run_full in attempts:
            M = two_level.operator(
                coarse_corrections=coarse_corrections,
                smoother_sweeps=smoother_sweeps,
            )
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
                best_M = M
            if best_residual <= residual_tolerance:
                return best, best_residual, history
            if not run_full:
                continue

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
                best_M = M
            if best_residual <= residual_tolerance:
                return best, best_residual, history

            # Once the V-cycle reaches the observed ~1e-6 plateau, solve the
            # actual Galerkin coarse defect instead of trying more rediscretized
            # Richardson corrections.  This directly removes the measured
            # coarse-consistency mismatch.
            if (
                not galerkin_polished
                and best_M is not None
                and np.isfinite(best_residual)
                and best_residual <= galerkin_start
            ):
                polished, polished_residual, extra = _galerkin_defect_polish(
                    local_solver_module,
                    two_level,
                    A,
                    rhs,
                    best,
                    best_residual,
                    best_M,
                    residual_tolerance,
                    cfg,
                )
                history.extend(extra)
                galerkin_polished = True
                if np.isfinite(polished_residual) and polished_residual < best_residual:
                    best = polished
                    best_residual = polished_residual
                if best_residual <= residual_tolerance:
                    return best, best_residual, history

        if (
            not galerkin_polished
            and best_M is not None
            and np.isfinite(best_residual)
            and best_residual <= galerkin_start
        ):
            polished, polished_residual, extra = _galerkin_defect_polish(
                local_solver_module,
                two_level,
                A,
                rhs,
                best,
                best_residual,
                best_M,
                residual_tolerance,
                cfg,
            )
            history.extend(extra)
            if np.isfinite(polished_residual) and polished_residual < best_residual:
                best = polished
                best_residual = polished_residual
            if best_residual <= residual_tolerance:
                return best, best_residual, history

        if best_M is not None and best_residual <= 1e-8:
            refined, refined_residual, extra = local_solver_module._defect_refine(
                A,
                rhs,
                best,
                best_M,
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


__all__ = ["_galerkin_defect_polish", "_relative_pilot", "_two_level_minimum_dofs", "install"]
