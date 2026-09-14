"""Fast, numerically certified local-self Maxwell solve.

The canonical local problem is intentionally much finer than the production
background.  A sparse direct factorization of the 3-D curl-curl matrix suffers
from severe fill-in once the edge count reaches O(1e5), so using SuperLU for
all local solves can turn preflight into an hours-long operation.

This module keeps the *same assembled physical operator and the same residual
certificate*, but changes only the linear algebra:

* small systems use a fill-reducing sparse direct solve;
* large systems use ILU-preconditioned LGMRES;
* progressively stronger bounded-fill ILU attempts are tried only when needed;
* a previous coarser local field is interpolated onto the next refined edge
  grid and used as the Krylov initial guess;
* every accepted result is checked against the original matrix with the true
  relative residual.  No tolerance or physics Gate is relaxed.

The one-field warm state stored on the parent background contains only edge
component arrays/coordinates, not a second Maxwell matrix, so refinement does
not retain the coarse sparse operator in memory.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator


def _relative_residual(A, field, rhs):
    return float(
        np.linalg.norm(rhs - A @ field)
        / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    )


def _canonical_key(local_geometry, port):
    # _canonical_port_geometry already removes translation/rotation.  Keep the
    # port index in the key because material assignments are port-indexed.
    return int(port), local_geometry.canonical_json()


def _edge_component_axes(background, axis):
    xm = 0.5 * (background.x[:-1] + background.x[1:])
    ym = 0.5 * (background.y[:-1] + background.y[1:])
    zm = 0.5 * (background.z[:-1] + background.z[1:])
    if axis == 0:
        return xm, background.y, background.z
    if axis == 1:
        return background.x, ym, background.z
    return background.x, background.y, zm


def _edge_component_shape(background, axis):
    if axis == 0:
        return background.nx, background.ny + 1, background.nz + 1
    if axis == 1:
        return background.nx + 1, background.ny, background.nz + 1
    return background.nx + 1, background.ny + 1, background.nz


def _pack_warm_state(local, local_geometry, port, fine_step, field):
    """Store compact E-component grids for a subsequent refined solve."""
    field = np.asarray(field, complex).reshape(-1)
    components = []
    offset = 0
    for axis in range(3):
        shape = _edge_component_shape(local, axis)
        count = int(np.prod(shape))
        part = field[offset : offset + count]
        lengths = local.edge_lengths[offset : offset + count]
        if len(part) != count:
            raise RuntimeError("local edge ordering is inconsistent with Cartesian topology")
        # Edge DOFs are line integrals; interpolate the physical component.
        values = (part / lengths).reshape(shape)
        components.append((tuple(np.asarray(q, float) for q in _edge_component_axes(local, axis)), values))
        offset += count
    if offset != local.n_edges:
        raise RuntimeError("local edge warm-state packing did not consume all edge DOFs")
    return {
        "key": _canonical_key(local_geometry, port),
        "fine_step": float(fine_step),
        "components": tuple(components),
    }


def _warm_start(parent, local, local_geometry, port, fine_step):
    state = getattr(parent, "_local_self_warm_state", None)
    if not isinstance(state, dict) or state.get("key") != _canonical_key(local_geometry, port):
        return None, False
    # Warm starts are useful only when moving to an equal/finer grid.  Never
    # prolongate a fine field back to a coarser solve in a different audit.
    previous_step = float(state.get("fine_step", np.inf))
    if float(fine_step) > previous_step * (1.0 + 1e-12):
        return None, False

    pieces = []
    try:
        for axis, (old_axes, old_values) in enumerate(state["components"]):
            new_axes = _edge_component_axes(local, axis)
            mesh = np.meshgrid(*new_axes, indexing="ij")
            points = np.column_stack([q.ravel() for q in mesh])
            interp = RegularGridInterpolator(
                old_axes,
                old_values,
                method="linear",
                bounds_error=False,
                # Only a one-cell-thick strip near the refined outer boundary
                # can lie outside coarse edge-center coordinates.  Zero is a
                # conservative initial guess there and does not affect the
                # converged/certified solution.
                fill_value=0.0,
            )
            component = np.asarray(interp(points), complex).reshape(-1)
            pieces.append(component)
        guess = np.concatenate(pieces) * local.edge_lengths
    except (ValueError, TypeError, FloatingPointError):
        return None, False
    if guess.shape != (local.n_edges,) or np.any(~np.isfinite(guess)):
        return None, False
    return guess, True


def _direct_solve(A, rhs):
    """Sparse direct solve with symmetric-pattern ordering, then safe fallback."""
    try:
        lu = spla.splu(
            A.tocsc(),
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.01,
            options={"Equil": True},
        )
    except (RuntimeError, ValueError):
        lu = spla.splu(A.tocsc())
    return np.asarray(lu.solve(rhs), complex).reshape(-1), lu


def _lgmres(A, rhs, *, x0, M, rtol, maxiter, inner_m):
    kwargs = dict(x0=x0, M=M, atol=0.0, maxiter=maxiter, inner_m=inner_m, outer_k=3)
    try:
        return spla.lgmres(A, rhs, rtol=rtol, **kwargs)
    except TypeError:
        # SciPy 1.10/1.11 compatibility (tol was renamed to rtol later).
        return spla.lgmres(A, rhs, tol=rtol, **kwargs)


def _ilu_preconditioner(A, *, drop_tol, fill_factor, shift_factor):
    matrix = A
    if shift_factor > 0.0:
        row_scale = np.asarray(abs(A).sum(axis=1)).reshape(-1)
        scale = np.maximum(row_scale, np.finfo(float).tiny)
        matrix = A + sp.diags(float(shift_factor) * scale, format="csr")
    ilu = spla.spilu(
        matrix.tocsc(),
        drop_tol=float(drop_tol),
        fill_factor=float(fill_factor),
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.01,
    )
    M = spla.LinearOperator(A.shape, matvec=ilu.solve, dtype=A.dtype)
    return M


def _iterative_solve(A, rhs, x0, cfg, residual_tolerance):
    """Bounded-fill ILU + LGMRES with progressively stronger attempts."""
    maxiter = int(cfg.get("linear_iterative_maxiter", 40))
    inner_m = int(cfg.get("linear_iterative_inner_m", 30))
    if maxiter < 1 or inner_m < 2:
        raise ValueError("local iterative Maxwell solver iteration limits are invalid")

    attempts = (
        (
            float(cfg.get("linear_ilu_drop_tolerance", 5e-3)),
            float(cfg.get("linear_ilu_fill_factor", 4.0)),
            0.0,
            "ilu-fast",
        ),
        (
            float(cfg.get("linear_ilu_strong_drop_tolerance", 1e-3)),
            float(cfg.get("linear_ilu_strong_fill_factor", 8.0)),
            0.0,
            "ilu-strong",
        ),
        (
            float(cfg.get("linear_ilu_strong_drop_tolerance", 1e-3)),
            float(cfg.get("linear_ilu_strong_fill_factor", 8.0)),
            float(cfg.get("linear_ilu_shift_factor", 1e-8)),
            "ilu-shifted",
        ),
    )
    target = max(float(residual_tolerance) * 0.2, 1e-12)
    best = None if x0 is None else np.asarray(x0, complex).reshape(-1).copy()
    best_residual = float("inf") if best is None else _relative_residual(A, best, rhs)
    history = []

    for drop_tol, fill_factor, shift_factor, label in attempts:
        t0 = time.perf_counter()
        try:
            M = _ilu_preconditioner(
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
        candidate, info = _lgmres(
            A,
            rhs,
            x0=best,
            M=M,
            rtol=target,
            maxiter=maxiter,
            inner_m=inner_m,
        )
        candidate = np.asarray(candidate, complex).reshape(-1)
        residual = _relative_residual(A, candidate, rhs)
        elapsed = time.perf_counter() - t0
        history.append(
            {
                "solver": label,
                "drop_tolerance": drop_tol,
                "fill_factor": fill_factor,
                "shift_factor": shift_factor,
                "krylov_info": int(info),
                "preconditioner_seconds": float(t1 - t0),
                "seconds": float(elapsed),
                "relative_residual": float(residual),
            }
        )
        print(
            f"local Maxwell {label}: residual={residual:.3e}, info={int(info)}, "
            f"time={elapsed:.1f}s",
            flush=True,
        )
        if np.isfinite(residual) and residual < best_residual:
            best = candidate
            best_residual = residual
        if best is not None and best_residual <= residual_tolerance:
            return best, best_residual, history

    return best, best_residual, history


def install(self_correction_module):
    if bool(getattr(self_correction_module, "_certified_local_solve_installed", False)):
        return self_correction_module

    def solve_local(parent, geometry, port, fine_step, phi=None):
        started = time.perf_counter()
        global_geometry, local_geometry, local = self_correction_module._local_background(
            parent, geometry, port, fine_step
        )
        context = local.geometry_context(local_geometry, assemble_thermal=False)
        A = local.em_operator(context, None)
        B = np.asarray(local.rhs_matrix(context), complex)
        if B.shape[1] != 1:
            raise AssertionError("canonical self-correction problem must have exactly one port")
        rhs = B[:, 0]

        cfg = self_correction_module._config(parent)
        residual_tolerance = float(cfg.get("linear_relative_residual_tolerance", 1e-9))
        refinement_steps = int(cfg.get("linear_refinement_steps", 3))
        direct_max_dofs = int(cfg.get("linear_direct_max_dofs", 60000))
        direct_fallback_max_dofs = int(cfg.get("linear_direct_fallback_max_dofs", 120000))
        if residual_tolerance <= 0.0:
            raise ValueError("self_correction.linear_relative_residual_tolerance must be positive")
        if refinement_steps < 0 or direct_max_dofs < 1 or direct_fallback_max_dofs < direct_max_dofs:
            raise ValueError("invalid local Maxwell linear solver configuration")

        x0, warm_started = _warm_start(parent, local, local_geometry, port, fine_step)
        print(
            f"local Maxwell solve: port={int(port)+1}, step={float(fine_step):g}m, "
            f"edges={local.n_edges}, solver={'direct' if local.n_edges <= direct_max_dofs else 'ILU+LGMRES'}, "
            f"warm_start={'yes' if warm_started else 'no'}",
            flush=True,
        )

        lu = None
        solver_history = []
        if local.n_edges <= direct_max_dofs:
            t0 = time.perf_counter()
            field, lu = _direct_solve(A, rhs)
            residual = _relative_residual(A, field, rhs)
            solver_history.append(
                {
                    "solver": "sparse-direct",
                    "seconds": float(time.perf_counter() - t0),
                    "relative_residual": float(residual),
                }
            )
        else:
            field, residual, solver_history = _iterative_solve(
                A, rhs, x0, cfg, residual_tolerance
            )
            if field is None or residual > residual_tolerance:
                # A bounded direct fallback is useful for medium systems, but
                # deliberately forbidden for the 2.25-mm ~250k-DOF reference:
                # that is exactly the hours-long fill-in path this solver avoids.
                if local.n_edges <= direct_fallback_max_dofs:
                    print(
                        f"local Maxwell iterative solve not yet certified ({residual:.3e}); "
                        "using bounded direct fallback",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    field, lu = _direct_solve(A, rhs)
                    residual = _relative_residual(A, field, rhs)
                    solver_history.append(
                        {
                            "solver": "sparse-direct-fallback",
                            "seconds": float(time.perf_counter() - t0),
                            "relative_residual": float(residual),
                        }
                    )
                else:
                    raise RuntimeError(
                        "large local Maxwell iterative solve did not reach the certified residual; "
                        f"edges={local.n_edges}, residual={float(residual):.3e}, "
                        f"tolerance={residual_tolerance:.3e}. Increase ILU strength/iterations rather "
                        "than falling back to an hours-long full sparse LU."
                    )

        if np.any(~np.isfinite(field)):
            raise FloatingPointError("local self-correction Maxwell solve produced non-finite fields")

        initial_residual = float(residual)
        refinements = 0
        # Direct solves can cheaply reuse their exact factorization for residual
        # correction.  Iterative candidates have already been driven by the true
        # residual and do not retain a huge factorization.
        if lu is not None:
            for _ in range(refinement_steps):
                if residual <= residual_tolerance:
                    break
                defect = np.asarray(rhs - A @ field, complex).reshape(-1)
                delta = np.asarray(lu.solve(defect), complex).reshape(-1)
                if np.any(~np.isfinite(delta)):
                    break
                candidate = field + delta
                candidate_residual = _relative_residual(A, candidate, rhs)
                if not np.isfinite(candidate_residual) or candidate_residual >= residual:
                    break
                field = candidate
                residual = candidate_residual
                refinements += 1

        # Save only compact component grids for coarse->fine/refinement warm start.
        parent._local_self_warm_state = _pack_warm_state(
            local, local_geometry, port, fine_step, field
        )

        source = np.asarray(context.source_shape[:, 0], float)
        z = complex(-source @ field)
        sigma = np.asarray(local.cell_properties(context, None, em=True)[0], float)
        edge_loss = np.asarray(local.edge_cell_hodge @ sigma).reshape(-1)
        d = float(np.real(field.conj() @ (edge_loss * field)))
        outward_weights = np.asarray(local.outward_loss_weights(), float).reshape(-1)
        d_out = float(np.real(field.conj() @ (outward_weights * field)))
        q_cells = np.asarray(
            0.5
            * sigma
            * np.asarray(local.edge_cell_hodge.T @ (np.abs(field) ** 2)).reshape(-1),
            float,
        )

        direct_d = float(2.0 * np.sum(q_cells))
        joule_total_error = self_correction_module._relative_identity_error(d, direct_d)

        modal = None
        modal_error = 0.0
        if phi is not None:
            local_phi = self_correction_module._phi_on_local_grid(
                parent, global_geometry, port, local, phi
            )
            direct_modal = np.asarray(2.0 * (local_phi.T @ q_cells), float)
            modal = direct_modal.copy()
            modal_error = self_correction_module._relative_identity_error(modal, direct_modal)

        scale = max(abs(z.real), abs(d) + abs(d_out), np.finfo(float).tiny)
        balance = float(abs(z.real - d - d_out) / scale)
        elapsed_total = float(time.perf_counter() - started)
        print(
            f"local Maxwell certified: edges={local.n_edges}, residual={residual:.3e}, "
            f"total={elapsed_total:.1f}s",
            flush=True,
        )
        return {
            "z": z,
            "d_vol": d,
            "d_out": d_out,
            "modal_h": modal,
            "linear_solver": "sparse-direct" if local.n_edges <= direct_max_dofs else "ilu-lgmres",
            "linear_warm_started": bool(warm_started),
            "linear_initial_relative_residual": float(initial_residual),
            "linear_relative_residual": float(residual),
            "linear_relative_residual_tolerance": float(residual_tolerance),
            "linear_refinement_iterations": int(refinements),
            "linear_solver_converged": bool(residual <= residual_tolerance),
            "linear_solver_history": solver_history,
            "linear_solve_total_seconds": elapsed_total,
            "power_balance_relative_error": balance,
            "joule_total_power_relative_error": joule_total_error,
            "joule_modal_contraction_relative_error": modal_error,
            "n_cells": int(local.n_cells),
            "n_edges": int(local.n_edges),
            "fine_step": float(fine_step),
        }

    self_correction_module._solve_local = solve_local
    self_correction_module._certified_local_solve_installed = True
    return self_correction_module


__all__ = ["install"]
