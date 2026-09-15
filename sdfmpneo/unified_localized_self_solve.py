"""Install localized local-self truth extraction on top of the certified solver.

The certified Maxwell solve remains unchanged and always uses the full physical
open two-terminal source.  This wrapper changes only the *local defect truth*
returned to the fine-minus-coarse correction: the pure scalar-gradient terminal
self energy is kept at global scale, while localizable transverse/cross response
is eligible for canonical local refinement.
"""
from __future__ import annotations

import time

import numpy as np

from .unified_compensated_field import (
    collapsed_field,
    field_abs2,
    field_is_finite,
    field_linear_dot,
)


def install(self_correction_module, certified_module):
    if bool(getattr(self_correction_module, "_localized_self_solve_installed", False)):
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

        x0, warm_started = certified_module._warm_start(
            parent, local, local_geometry, port, fine_step
        )
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
            field, lu = certified_module._direct_solve(A, rhs)
            residual = certified_module._relative_residual(A, field, rhs)
            solver_history.append({
                "solver": "sparse-direct",
                "seconds": float(time.perf_counter() - t0),
                "relative_residual": float(residual),
            })
        else:
            field, residual, solver_history = certified_module._iterative_solve(
                A, rhs, x0, cfg, residual_tolerance,
                background=local,
                context=context,
            )
            if field is None or residual > residual_tolerance:
                if local.n_edges <= direct_fallback_max_dofs:
                    print(
                        f"local Maxwell iterative solve not yet certified ({residual:.3e}); "
                        "using bounded direct fallback",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    field, lu = certified_module._direct_solve(A, rhs)
                    residual = certified_module._relative_residual(A, field, rhs)
                    solver_history.append({
                        "solver": "sparse-direct-fallback",
                        "seconds": float(time.perf_counter() - t0),
                        "relative_residual": float(residual),
                    })
                else:
                    raise RuntimeError(
                        "large local Maxwell iterative solve did not reach the certified residual; "
                        f"edges={local.n_edges}, residual={float(residual):.3e}, "
                        f"tolerance={residual_tolerance:.3e}."
                    )

        if not field_is_finite(field):
            raise FloatingPointError("local self-correction Maxwell solve produced non-finite fields")

        initial_residual = float(residual)
        refinements = 0
        if lu is not None:
            for _ in range(refinement_steps):
                if residual <= residual_tolerance:
                    break
                field_array = collapsed_field(field)
                defect = np.asarray(rhs - A @ field_array, complex).reshape(-1)
                delta = np.asarray(lu.solve(defect), complex).reshape(-1)
                if np.any(~np.isfinite(delta)):
                    break
                candidate = field_array + delta
                candidate_residual = certified_module._relative_residual(A, candidate, rhs)
                if not np.isfinite(candidate_residual) or candidate_residual >= residual:
                    break
                field = candidate
                residual = candidate_residual
                refinements += 1

        # Warm state is an initial guess only.  The currently certified high/low
        # field remains intact for all truth contractions below.
        parent._local_self_warm_state = certified_module._pack_warm_state(
            local, local_geometry, port, fine_step, field
        )

        source = np.asarray(context.source_shape[:, 0], float)
        z = -field_linear_dot(source, field)
        sigma = np.asarray(local.cell_properties(context, None, em=True)[0], float)
        edge_loss = np.asarray(local.edge_cell_hodge @ sigma).reshape(-1)
        abs2 = field_abs2(field)
        d = float(np.sum(edge_loss * abs2))
        outward_weights = np.asarray(local.outward_loss_weights(), float).reshape(-1)
        d_out = float(np.sum(outward_weights * abs2))
        q_cells = np.asarray(
            0.5 * sigma * np.asarray(local.edge_cell_hodge.T @ abs2).reshape(-1),
            float,
        )

        direct_d = float(2.0 * np.sum(q_cells))
        joule_total_error = self_correction_module._relative_identity_error(d, direct_d)

        local_phi = None
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
        localized = self_correction_module._localized_self_response(
            local,
            context,
            A,
            rhs,
            field,
            source,
            sigma,
            edge_loss,
            outward_weights,
            local_phi=local_phi,
        )
        print(
            "local self localized response: "
            f"raw_z=({z.real:.3e},{z.imag:.3e}), "
            f"local_z=({localized['localized_z'].real:.3e},{localized['localized_z'].imag:.3e}), "
            f"raw_d={d:.3e}, local_d={localized['localized_d_vol']:.3e}, "
            f"long_z_im={localized['longitudinal_z'].imag:.3e}",
            flush=True,
        )

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
            **localized,
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
    self_correction_module._localized_self_solve_installed = True
    return self_correction_module


__all__ = ["install"]
