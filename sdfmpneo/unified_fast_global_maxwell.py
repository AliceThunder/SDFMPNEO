"""Fast certified multi-port Maxwell solves for global truth and Physics Gates.

Large open-domain matrices use one shared Maxwell-aware shifted-ILU per attempt
and LGMRES for every port.  If the main Krylov solve reaches the 1e-6--1e-7
range but not the final certificate, true-residual defect correction reuses the
same ILU instead of building an expensive sparse LU.  Acceptance always checks
the original unmodified physical Maxwell operator.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse.linalg as spla


def _cfg(background):
    root = dict(getattr(background, "background_config", {}) or {})
    cfg = dict(root.get("linear_solver", {}) or {})
    cfg.setdefault("relative_residual_tolerance", 1e-9)
    cfg.setdefault("direct_max_dofs", 60000)
    cfg.setdefault("iterative_maxiter", 40)
    cfg.setdefault("iterative_inner_m", 30)
    cfg.setdefault("iterative_defect_steps", 3)
    cfg.setdefault("iterative_defect_maxiter", 16)
    cfg.setdefault("iterative_defect_inner_m", 20)
    cfg.setdefault("iterative_defect_start_residual", 5e-6)
    cfg.setdefault("ilu_drop_tolerance", 5e-3)
    cfg.setdefault("ilu_fill_factor", 4.0)
    cfg.setdefault("ilu_strong_drop_tolerance", 1e-3)
    cfg.setdefault("ilu_strong_fill_factor", 8.0)
    cfg.setdefault("ilu_shift_factor", 3e-2)
    cfg.setdefault("ilu_strong_shift_factor", 1e-1)
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


def solve_multi_rhs(background, A, B, local_solver_module):
    """Solve all port RHS with one shared preconditioner and certify true residuals."""
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

    maxiter = int(cfg["iterative_maxiter"])
    inner_m = int(cfg["iterative_inner_m"])
    defect_steps = int(cfg["iterative_defect_steps"])
    defect_maxiter = int(cfg["iterative_defect_maxiter"])
    defect_inner_m = int(cfg["iterative_defect_inner_m"])
    defect_start = float(cfg["iterative_defect_start_residual"])
    target = max(0.2 * tolerance, 1e-12)
    attempts = (
        (
            float(cfg["ilu_drop_tolerance"]),
            float(cfg["ilu_fill_factor"]),
            float(cfg["ilu_shift_factor"]),
            "shifted-ilu-fast",
        ),
        (
            float(cfg["ilu_strong_drop_tolerance"]),
            float(cfg["ilu_strong_fill_factor"]),
            float(cfg["ilu_strong_shift_factor"]),
            "shifted-ilu-strong",
        ),
        (
            float(cfg["ilu_strong_drop_tolerance"]),
            float(cfg["ilu_strong_fill_factor"]),
            max(5e-3, 0.5 * float(cfg["ilu_shift_factor"])),
            "shifted-ilu-tight",
        ),
    )
    history = []
    best_X = None
    best_residual = float("inf")

    for drop_tol, fill_factor, shift_factor, label in attempts:
        started = time.perf_counter()
        try:
            M = local_solver_module._ilu_preconditioner(
                A,
                drop_tol=drop_tol,
                fill_factor=fill_factor,
                shift_factor=shift_factor,
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
        history.append({
            "solver": label,
            "drop_tolerance": drop_tol,
            "fill_factor": fill_factor,
            "shift_factor": shift_factor,
            "krylov_info": infos,
            "preconditioner_seconds": float(preconditioner_seconds),
            "seconds": float(elapsed),
            "maximum_relative_residual": worst,
        })
        print(
            f"global Maxwell {label}: edges={A.shape[0]}, ports={B.shape[1]}, "
            f"residual={worst:.3e}, info={infos}, time={elapsed:.1f}s",
            flush=True,
        )
        if np.isfinite(worst) and worst < best_residual:
            best_X = X
            best_residual = worst
        if best_X is not None and best_residual <= tolerance:
            return best_X, best_residual, tuple(history)

        if (
            defect_steps > 0
            and best_X is not None
            and np.isfinite(best_residual)
            and best_residual <= defect_start
        ):
            refined = best_X.copy()
            correction_rows = []
            for p in range(B.shape[1]):
                rp = float(_true_residuals(A, refined[:, [p]], B[:, [p]])[0])
                if rp <= tolerance:
                    continue
                field, rp_new, rows = local_solver_module._defect_refine(
                    A,
                    B[:, p],
                    refined[:, p],
                    M,
                    local_solver_module._lgmres,
                    local_solver_module._relative_residual,
                    tolerance,
                    steps=defect_steps,
                    maxiter=defect_maxiter,
                    inner_m=defect_inner_m,
                    label=f"global-{label}-p{p+1}-defect",
                )
                refined[:, p] = field
                correction_rows.extend(rows)
            history.extend(correction_rows)
            refined_residuals = _true_residuals(A, refined, B)
            refined_worst = float(np.max(refined_residuals))
            if np.isfinite(refined_worst) and refined_worst < best_residual:
                best_X = refined
                best_residual = refined_worst
            if best_residual <= tolerance:
                print(
                    f"global Maxwell {label} defect-corrected: residual={best_residual:.3e}",
                    flush=True,
                )
                return best_X, best_residual, tuple(history)

    raise RuntimeError(
        "large global Maxwell iterative solve did not reach the certified residual after "
        "true-residual defect correction; "
        f"edges={A.shape[0]}, residual={best_residual:.3e}, tolerance={tolerance:.3e}. "
        "Increase Krylov/ILU strength rather than using an hours-long full sparse LU."
    )


def install(physics_gate_module, tensor_surrogate_module, local_solver_module):
    if bool(getattr(physics_gate_module, "_fast_global_maxwell_installed", False)):
        return

    def solve_fields(background, geometry, *, mqs=False):
        context = background.geometry_context(geometry, assemble_thermal=False)
        A = physics_gate_module._operator(background, context, mqs=mqs)
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
