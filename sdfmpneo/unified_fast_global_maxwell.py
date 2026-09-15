"""Fast certified multi-port Maxwell solves for global truth and Physics Gates.

The global open-boundary background may contain O(1e5) edge unknowns once the
artificial boundary is far enough away to pass the domain-convergence Gate.
Factoring every geometry with SuperLU causes the same 3-D fill-in problem as the
canonical local-self solve.  This module changes only the linear algebra:

* small systems keep the sparse direct path;
* large systems use the Maxwell-aware shifted-ILU preconditioner already used by
  the certified local solver;
* one ILU is shared by every port right-hand side for a given Maxwell matrix;
* acceptance always uses the true residual of the unmodified physical operator.

No constitutive law, source, boundary form, tolerance, or physics Gate is changed.
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
            max(1e-2, 0.5 * float(cfg["ilu_shift_factor"])),
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

    raise RuntimeError(
        "large global Maxwell iterative solve did not reach the certified residual; "
        f"edges={A.shape[0]}, residual={best_residual:.3e}, tolerance={tolerance:.3e}. "
        "Increase iterative/ILU strength rather than using an hours-long full sparse LU."
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
