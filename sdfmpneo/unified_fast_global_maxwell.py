"""Fast certified multi-port Maxwell solves for global truth and Physics Gates.

Large open-domain systems use the exact compatible scalar-gradient block for
the longitudinal response and a transverse-only stabilized ILU for the remaining
Maxwell correction.  One scalar factor and one edge preconditioner are shared by
all port RHS for a geometry.  No source projection, physical-matrix shift, or
tolerance relaxation is used; acceptance always checks the original Maxwell
matrix and RHS.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse.linalg as spla

from .unified_gradient_block_maxwell import (
    build_gradient_block,
    compose_block_preconditioner,
)
from .unified_transverse_ilu import build_transverse_ilu


def _cfg(background):
    root = dict(getattr(background, "background_config", {}) or {})
    cfg = dict(root.get("linear_solver", {}) or {})
    cfg.setdefault("relative_residual_tolerance", 1e-9)
    cfg.setdefault("direct_max_dofs", 60000)
    cfg.setdefault("iterative_maxiter", 24)
    cfg.setdefault("iterative_inner_m", 24)
    cfg.setdefault("ilu_drop_tolerance", 5e-3)
    cfg.setdefault("ilu_fill_factor", 4.0)
    cfg.setdefault("ilu_strong_drop_tolerance", 1e-3)
    cfg.setdefault("ilu_strong_fill_factor", 8.0)
    cfg.setdefault("ilu_shift_factor", 3e-2)
    cfg.setdefault("ilu_strong_shift_factor", 1e-1)
    cfg.setdefault("transverse_stabilization_factor", 3e-2)
    cfg.setdefault("transverse_strong_stabilization_factor", 1e-1)
    # Keep compatibility with run.py's iterative_defect_* names.  These are
    # solver-policy aliases only; the physical residual certificate is unchanged.
    cfg.setdefault("defect_steps", cfg.get("iterative_defect_steps", 2))
    cfg.setdefault("defect_maxiter", cfg.get("iterative_defect_maxiter", 12))
    cfg.setdefault("defect_inner_m", cfg.get("iterative_defect_inner_m", 20))
    cfg.setdefault(
        "defect_start_residual",
        cfg.get("iterative_defect_start_residual", 1e-8),
    )
    # A final true-residual rescue is allowed from a wider basin only after
    # every primary preconditioned solve has failed.  This changes solver
    # policy, not the physical operator or the 1e-9 acceptance certificate.
    cfg.setdefault(
        "defect_rescue_start_residual",
        cfg.get("iterative_defect_rescue_start_residual", 2e-2),
    )
    cfg.setdefault(
        "defect_rescue_steps",
        cfg.get("iterative_defect_rescue_steps", max(4, int(cfg["defect_steps"]))),
    )
    cfg.setdefault(
        "defect_rescue_maxiter",
        cfg.get("iterative_defect_rescue_maxiter", max(24, int(cfg["defect_maxiter"]))),
    )
    cfg.setdefault(
        "defect_rescue_inner_m",
        cfg.get("iterative_defect_rescue_inner_m", max(30, int(cfg["defect_inner_m"]))),
    )
    return cfg


def _true_residuals(A, X, B):
    B = np.asarray(B, complex)
    X = np.asarray(X, complex)
    scale = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
    return np.asarray(np.linalg.norm(B - A @ X, axis=0) / scale, float)


def _direct(A, B):
    started = time.perf_counter()
    try:
        lu = spla.splu(
            A.tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        lu = spla.splu(A.tocsc())
    X = np.column_stack([lu.solve(B[:, p]) for p in range(B.shape[1])])
    return np.asarray(X, complex), float(time.perf_counter() - started)


def _defect_cleanup(
    A,
    B,
    X,
    M,
    local_solver_module,
    tolerance,
    steps,
    *,
    maxiter,
    inner_m,
    label_prefix="compatible-transverse-defect",
):
    X = np.asarray(X, complex).copy()
    history = []
    for sweep in range(int(steps)):
        residuals = _true_residuals(A, X, B)
        if float(np.max(residuals)) <= tolerance:
            break
        started = time.perf_counter()
        infos = []
        for p in range(B.shape[1]):
            if residuals[p] <= tolerance:
                infos.append(0)
                continue
            defect = np.asarray(B[:, p] - A @ X[:, p], complex).reshape(-1)
            eta = min(
                2e-2,
                max(1e-5, 0.25 * tolerance / max(float(residuals[p]), np.finfo(float).tiny)),
            )
            delta, info = local_solver_module._lgmres(
                A,
                defect,
                x0=None,
                M=M,
                rtol=eta,
                maxiter=int(maxiter),
                inner_m=int(inner_m),
            )
            candidate = X[:, p] + np.asarray(delta, complex).reshape(-1)
            old = float(residuals[p])
            new = float(_true_residuals(A, candidate[:, None], B[:, p : p + 1])[0])
            if np.isfinite(new) and new < old:
                X[:, p] = candidate
            infos.append(int(info))
        after = _true_residuals(A, X, B)
        elapsed = time.perf_counter() - started
        history.append(
            {
                "solver": f"{label_prefix}-{sweep+1}",
                "krylov_info": infos,
                "seconds": float(elapsed),
                "maximum_relative_residual": float(np.max(after)),
            }
        )
        print(
            f"global Maxwell {label_prefix}-{sweep+1}: "
            f"residual={float(np.max(after)):.3e}, info={infos}, time={elapsed:.1f}s",
            flush=True,
        )
    return X, float(np.max(_true_residuals(A, X, B))), history


def solve_multi_rhs(background, A, B, local_solver_module):
    """Solve all port RHS with shared compatible preconditioners and certify them."""
    B = np.asarray(B, complex)
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("global Maxwell RHS must have shape (n_edges, n_ports)")
    cfg = _cfg(background)
    tolerance = float(cfg["relative_residual_tolerance"])
    direct_max = int(cfg["direct_max_dofs"])
    if tolerance <= 0.0 or direct_max < 1:
        raise ValueError("invalid global Maxwell linear solver configuration")

    if A.shape[0] <= direct_max:
        X, elapsed = _direct(A, B)
        residuals = _true_residuals(A, X, B)
        print(
            f"global Maxwell sparse-direct: edges={A.shape[0]}, ports={B.shape[1]}, "
            f"residual={float(np.max(residuals)):.3e}, time={elapsed:.1f}s",
            flush=True,
        )
        if float(np.max(residuals)) > tolerance:
            raise RuntimeError(
                "global Maxwell direct solve failed residual certificate: "
                f"{float(np.max(residuals)):.3e} > {tolerance:.3e}"
            )
        return X, float(np.max(residuals)), ({
            "solver": "sparse-direct",
            "seconds": elapsed,
            "maximum_relative_residual": float(np.max(residuals)),
        },)

    context = getattr(A, "_sdfmpneo_context", None)
    tagged_background = getattr(A, "_sdfmpneo_background", None)
    mqs = bool(getattr(A, "_sdfmpneo_mqs", False))
    mqs_admittance = getattr(A, "_sdfmpneo_mqs_admittance", None)
    gradient_block = None
    if context is not None and tagged_background is not None:
        gradient_block = build_gradient_block(
            tagged_background,
            context,
            mqs=mqs,
            mqs_admittance=mqs_admittance,
            check_topology=True,
        )

    maxiter = int(cfg["iterative_maxiter"])
    inner_m = int(cfg["iterative_inner_m"])
    target = max(0.2 * tolerance, 1e-12)
    fast_drop = float(cfg["ilu_drop_tolerance"])
    fast_fill = float(cfg["ilu_fill_factor"])
    strong_drop = float(cfg["ilu_strong_drop_tolerance"])
    strong_fill = float(cfg["ilu_strong_fill_factor"])
    if gradient_block is not None:
        attempts = (
            (
                fast_drop,
                fast_fill,
                float(cfg["transverse_stabilization_factor"]),
                "compatible-transverse-ilu-fast",
                "transverse",
            ),
            (
                strong_drop,
                strong_fill,
                float(cfg["transverse_strong_stabilization_factor"]),
                "compatible-transverse-ilu-strong",
                "transverse",
            ),
            (
                strong_drop,
                strong_fill,
                max(5e-3, 0.5 * float(cfg["ilu_shift_factor"])),
                "compatible-shifted-ilu-tight-recovery",
                "shifted",
            ),
        )
    else:
        attempts = (
            (fast_drop, fast_fill, float(cfg["ilu_shift_factor"]), "shifted-ilu-fast", "shifted"),
            (
                strong_drop,
                strong_fill,
                float(cfg["ilu_strong_shift_factor"]),
                "shifted-ilu-strong",
                "shifted",
            ),
        )

    history = []
    best_X = None
    best_M = None
    best_residual = float("inf")

    for drop_tol, fill_factor, stabilization, label, mode in attempts:
        started = time.perf_counter()
        preconditioner_stats = {}
        try:
            if mode == "transverse":
                edge_M, preconditioner_stats = build_transverse_ilu(
                    A,
                    tagged_background,
                    context,
                    gradient_block,
                    drop_tol=drop_tol,
                    fill_factor=fill_factor,
                    stabilization_factor=stabilization,
                    mqs=mqs,
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
                compose_block_preconditioner(A, edge_M, gradient_block, post_correct=True)
                if gradient_block is not None
                else edge_M
            )
        except (RuntimeError, ValueError, MemoryError) as exc:
            history.append({
                "solver": label,
                "preconditioner_failed": True,
                "error": type(exc).__name__,
                "seconds": float(time.perf_counter() - started),
            })
            continue

        preconditioner_seconds = time.perf_counter() - started
        columns = []
        infos = []
        for p in range(B.shape[1]):
            x0 = None if best_X is None else best_X[:, p]
            candidate, info = local_solver_module._lgmres(
                A,
                B[:, p],
                x0=x0,
                M=M,
                rtol=target,
                maxiter=maxiter,
                inner_m=inner_m,
            )
            columns.append(np.asarray(candidate, complex).reshape(-1))
            infos.append(int(info))
        X = np.column_stack(columns)
        residuals = _true_residuals(A, X, B)
        worst = float(np.max(residuals))
        elapsed = time.perf_counter() - started
        row = {
            "solver": label,
            "drop_tolerance": drop_tol,
            "fill_factor": fill_factor,
            "krylov_info": infos,
            "preconditioner_seconds": float(preconditioner_seconds),
            "gradient_scalar_dofs": 0 if gradient_block is None else gradient_block.scalar_dofs,
            "gradient_factor_seconds": 0.0 if gradient_block is None else gradient_block.build_seconds,
            "seconds": float(elapsed),
            "maximum_relative_residual": worst,
        }
        if mode == "transverse":
            row.update({f"transverse_{k}": v for k, v in preconditioner_stats.items()})
        else:
            row["shift_factor"] = float(stabilization)
        history.append(row)
        print(
            f"global Maxwell {label}: edges={A.shape[0]}, ports={B.shape[1]}, "
            f"residual={worst:.3e}, info={infos}, time={elapsed:.1f}s",
            flush=True,
        )
        if np.isfinite(worst) and worst < best_residual:
            best_X = X
            best_M = M
            best_residual = worst
        if best_X is not None and best_residual <= tolerance:
            return best_X, best_residual, tuple(history)

        defect_start = max(
            float(cfg["defect_start_residual"]),
            float(tolerance),
        )
        if (
            best_X is not None
            and best_residual <= defect_start
            and int(cfg["defect_steps"]) > 0
        ):
            refined, refined_residual, extra = _defect_cleanup(
                A,
                B,
                best_X,
                M,
                local_solver_module,
                tolerance,
                int(cfg["defect_steps"]),
                maxiter=int(cfg["defect_maxiter"]),
                inner_m=int(cfg["defect_inner_m"]),
            )
            history.extend(extra)
            if refined_residual < best_residual:
                best_X = refined
                best_M = M
                best_residual = refined_residual
            if best_residual <= tolerance:
                return best_X, best_residual, tuple(history)

    rescue_start = max(
        float(cfg["defect_rescue_start_residual"]),
        float(tolerance),
    )
    if (
        best_X is not None
        and best_M is not None
        and np.isfinite(best_residual)
        and best_residual <= rescue_start
        and int(cfg["defect_rescue_steps"]) > 0
    ):
        rescued, rescued_residual, extra = _defect_cleanup(
            A,
            B,
            best_X,
            best_M,
            local_solver_module,
            tolerance,
            int(cfg["defect_rescue_steps"]),
            maxiter=int(cfg["defect_rescue_maxiter"]),
            inner_m=int(cfg["defect_rescue_inner_m"]),
            label_prefix="certified-defect-rescue",
        )
        history.extend(extra)
        if rescued_residual < best_residual:
            best_X = rescued
            best_residual = rescued_residual
        if best_residual <= tolerance:
            return best_X, best_residual, tuple(history)

    preconditioner_name = (
        "compatible gradient/transverse"
        if gradient_block is not None
        else "shifted-ILU fallback"
    )
    attempt_summary = []
    for row in history:
        label = str(row.get("solver", "unknown"))
        if row.get("preconditioner_failed", False):
            attempt_summary.append(f"{label}=preconditioner_failed")
        elif "maximum_relative_residual" in row:
            attempt_summary.append(
                f"{label}={float(row['maximum_relative_residual']):.3e}"
            )
    raise RuntimeError(
        "large global Maxwell iterative solve did not reach the certified residual; "
        f"edges={A.shape[0]}, residual={best_residual:.3e}, tolerance={tolerance:.3e}, "
        f"attempts=[{'; '.join(attempt_summary)}]. "
        f"The {preconditioner_name} preconditioner was unable to certify this system; "
        "do not relax the Gate."
    )


def install(physics_gate_module, tensor_surrogate_module, local_solver_module):
    if bool(getattr(physics_gate_module, "_fast_global_maxwell_installed", False)):
        return

    def solve_fields(background, geometry, *, mqs=False):
        context = background.geometry_context(geometry, assemble_thermal=False)
        A = physics_gate_module._operator(background, context, mqs=mqs)
        A._sdfmpneo_background = background
        A._sdfmpneo_context = context
        A._sdfmpneo_mqs = bool(mqs)
        if mqs:
            A._sdfmpneo_mqs_admittance = physics_gate_module._mqs_boundary_admittance(background)
        B = np.asarray(background.rhs_matrix(context), complex)
        X, _residual, _history = solve_multi_rhs(background, A, B, local_solver_module)
        if np.any(~np.isfinite(X)):
            raise FloatingPointError("Physics Gate Maxwell solve produced non-finite fields")
        source = np.asarray(context.source_shape, float)
        reaction = -source.T @ X
        z = 0.5 * (reaction + reaction.T)
        sigma = np.asarray(background.cell_properties(context, None, em=True)[0], float)
        edge_loss = np.asarray(background.edge_cell_hodge @ sigma).reshape(-1)
        raw_d = X.conj().T @ (edge_loss[:, None] * X)
        d = 0.5 * (raw_d + raw_d.conj().T)
        admittance = (
            physics_gate_module._mqs_boundary_admittance(background)
            if mqs
            else background.boundary_admittance()
        )
        outward = float(admittance.real) * np.asarray(background.boundary_edge_hodge, float)
        raw_out = X.conj().T @ (outward[:, None] * X)
        d_out = 0.5 * (raw_out + raw_out.conj().T)
        return context, X, sigma, z, d, d_out

    def solve_port_fields(background, context):
        A = background.em_operator(context, None)
        # Thermal-anchor/tensor truth must use the same certified compatible
        # gradient+transverse solver path as the Physics Gate.  solve_multi_rhs
        # discovers that path from operator metadata; without these tags this
        # entry point silently falls back to the older shifted-ILU preconditioner.
        A._sdfmpneo_background = background
        A._sdfmpneo_context = context
        A._sdfmpneo_mqs = False
        B = np.asarray(background.rhs_matrix(context), complex)
        X, residual, _history = solve_multi_rhs(background, A, B, local_solver_module)
        if np.any(~np.isfinite(X)):
            raise FloatingPointError("Maxwell truth solve produced non-finite fields")
        return X, float(residual)

    physics_gate_module._solve_fields = solve_fields
    tensor_surrogate_module._solve_port_fields = solve_port_fields
    physics_gate_module._fast_global_maxwell_installed = True
    tensor_surrogate_module._fast_global_maxwell_installed = True


__all__ = ["install", "solve_multi_rhs"]
