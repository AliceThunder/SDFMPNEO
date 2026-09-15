"""Install localized local-self truth extraction on top of the certified solver.

The physical local Maxwell equation is unchanged and keeps the full open
source.  For numerical stability the compatible decomposition is now used as an
*exact solve split*, not only as a post-processing subtraction.

With

    A = C.T H_mu C + D,    C G = 0,

first solve the scalar-gradient block

    G.T D G phi = G.T b,    E_L = G phi,

then form the transverse right-hand side

    b_T = b - D E_L

with compensated accumulation and solve

    A E_T = b_T.

Therefore ``E = E_L + E_T`` satisfies the original full-source equation while
the Krylov certificate is applied directly to the small transverse field.  This
avoids recovering a O(1) localizable response by subtracting fields whose
longitudinal terminal response can be O(1e6) larger on the finest grid.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

from .unified_accurate_residual import accurate_residual_vector
from .unified_compensated_field import (
    as_compensated_field,
    collapsed_field,
    compensated_add,
    field_abs2,
    field_is_finite,
    field_linear_dot,
    field_parts,
)
from .unified_gradient_block_maxwell import _edge_mass_diagonal, build_gradient_block
from .unified_refined_gradient_projection import refined_gradient_projection


def _combine_split_field(longitudinal, transverse):
    out = as_compensated_field(longitudinal, copy=True)
    high, low = field_parts(transverse)
    out = compensated_add(out, high)
    out = compensated_add(out, low)
    return out


def _certify_transverse(A, field, rhs_transverse, rhs_full, tolerance):
    defect, diagnostics = accurate_residual_vector(
        A,
        field,
        rhs_transverse,
        target_relative=float(tolerance),
    )
    defect_norm = float(np.linalg.norm(defect))
    transverse_norm = max(float(np.linalg.norm(rhs_transverse)), np.finfo(float).tiny)
    full_norm = max(float(np.linalg.norm(rhs_full)), np.finfo(float).tiny)
    return (
        defect,
        float(defect_norm / transverse_norm),
        float(defect_norm / full_norm),
        diagnostics,
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

        # Exact compatible split.  The scalar solve retains the genuine terminal
        # divergence of the open source; it does not project the source away.
        gradient = build_gradient_block(local, context, check_topology=True)
        longitudinal, projection = refined_gradient_projection(
            local,
            gradient,
            rhs,
            relative_tolerance=5e-13,
            maximum_refinements=5,
        )
        diagonal = _edge_mass_diagonal(local, context)
        mass = sp.diags(diagonal, format="csr")
        split_target = min(5e-13, max(1e-15, 0.1 * residual_tolerance))
        transverse_rhs, split_diagnostics = accurate_residual_vector(
            mass,
            longitudinal,
            rhs,
            target_relative=split_target,
        )
        transverse_rhs = np.asarray(transverse_rhs, complex).reshape(-1)
        full_rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
        transverse_rhs_norm = max(float(np.linalg.norm(transverse_rhs)), np.finfo(float).tiny)
        rhs_ratio = float(transverse_rhs_norm / full_rhs_norm)
        scalar_rhs = np.asarray(gradient.gradient.T @ rhs, complex).reshape(-1)
        scalar_transverse = np.asarray(
            gradient.gradient.T @ transverse_rhs, complex
        ).reshape(-1)
        gradient_leakage = float(
            np.linalg.norm(scalar_transverse)
            / max(float(np.linalg.norm(scalar_rhs)), np.finfo(float).tiny)
        )
        print(
            "local Maxwell compatible split: "
            f"transverse_rhs={rhs_ratio:.3e} of full, "
            f"gradient_leakage={gradient_leakage:.3e}, "
            f"scalar_residual={projection['relative_residual']:.3e}, "
            f"mode={split_diagnostics.get('accumulation_mode', 'unknown')}",
            flush=True,
        )

        # Warm state is now the transverse field.  The H(curl) transfer and
        # two-level coarse-space construction therefore operate on the component
        # that the refined solve must actually resolve.
        x0, warm_started = certified_module._warm_start(
            parent, local, local_geometry, port, fine_step
        )
        print(
            f"local Maxwell transverse solve: port={int(port)+1}, step={float(fine_step):g}m, "
            f"edges={local.n_edges}, solver={'direct' if local.n_edges <= direct_max_dofs else 'ILU+LGMRES'}, "
            f"warm_start={'yes' if warm_started else 'no'}",
            flush=True,
        )

        lu = None
        solver_history = [{
            "solver": "compatible-longitudinal-transverse-split",
            "transverse_rhs_relative_norm": rhs_ratio,
            "gradient_leakage_relative_norm": gradient_leakage,
            "gradient_projection_relative_residual": float(projection["relative_residual"]),
            "split_accumulation_mode": str(split_diagnostics.get("accumulation_mode", "unknown")),
        }]

        if local.n_edges <= direct_max_dofs:
            t0 = time.perf_counter()
            transverse, lu = certified_module._direct_solve(A, transverse_rhs)
            defect, transverse_residual, physical_residual, residual_diagnostics = _certify_transverse(
                A, transverse, transverse_rhs, rhs, residual_tolerance
            )
            solver_history.append({
                "solver": "sparse-direct-transverse",
                "seconds": float(time.perf_counter() - t0),
                "relative_residual": float(transverse_residual),
                "physical_relative_residual": float(physical_residual),
                "accumulation_mode": str(residual_diagnostics.get("accumulation_mode", "unknown")),
            })
        else:
            transverse, reported_residual, iterative_history = certified_module._iterative_solve(
                A,
                transverse_rhs,
                x0,
                cfg,
                residual_tolerance,
                background=local,
                context=context,
            )
            solver_history.extend(iterative_history)
            if transverse is None:
                transverse_residual = float("inf")
                physical_residual = float("inf")
                defect = None
                residual_diagnostics = {}
            else:
                defect, transverse_residual, physical_residual, residual_diagnostics = _certify_transverse(
                    A, transverse, transverse_rhs, rhs, residual_tolerance
                )
                solver_history.append({
                    "solver": "accurate-transverse-certificate",
                    "reported_relative_residual": float(reported_residual),
                    "relative_residual": float(transverse_residual),
                    "physical_relative_residual": float(physical_residual),
                    "accumulation_mode": str(residual_diagnostics.get("accumulation_mode", "unknown")),
                })

            if transverse is None or transverse_residual > residual_tolerance:
                if local.n_edges <= direct_fallback_max_dofs:
                    print(
                        "local Maxwell transverse iterative solve not yet certified "
                        f"({transverse_residual:.3e}); using bounded direct fallback",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    transverse, lu = certified_module._direct_solve(A, transverse_rhs)
                    defect, transverse_residual, physical_residual, residual_diagnostics = _certify_transverse(
                        A, transverse, transverse_rhs, rhs, residual_tolerance
                    )
                    solver_history.append({
                        "solver": "sparse-direct-transverse-fallback",
                        "seconds": float(time.perf_counter() - t0),
                        "relative_residual": float(transverse_residual),
                        "physical_relative_residual": float(physical_residual),
                        "accumulation_mode": str(residual_diagnostics.get("accumulation_mode", "unknown")),
                    })
                else:
                    raise RuntimeError(
                        "large local transverse Maxwell solve did not reach the certified residual; "
                        f"edges={local.n_edges}, residual={float(transverse_residual):.3e}, "
                        f"tolerance={residual_tolerance:.3e}."
                    )

        if not field_is_finite(transverse):
            raise FloatingPointError("local self-correction transverse solve produced non-finite fields")

        initial_transverse_residual = float(transverse_residual)
        initial_physical_residual = float(physical_residual)
        refinements = 0
        if lu is not None:
            for _ in range(refinement_steps):
                if transverse_residual <= residual_tolerance:
                    break
                delta = np.asarray(lu.solve(defect), complex).reshape(-1)
                if np.any(~np.isfinite(delta)):
                    break
                candidate = compensated_add(transverse, delta)
                candidate_defect, candidate_transverse, candidate_physical, candidate_diag = (
                    _certify_transverse(A, candidate, transverse_rhs, rhs, residual_tolerance)
                )
                if not np.isfinite(candidate_transverse) or candidate_transverse >= transverse_residual:
                    break
                transverse = candidate
                defect = candidate_defect
                transverse_residual = candidate_transverse
                physical_residual = candidate_physical
                residual_diagnostics = candidate_diag
                refinements += 1

        if transverse_residual > residual_tolerance:
            raise RuntimeError(
                "local transverse Maxwell field failed the certified residual after refinement; "
                f"residual={transverse_residual:.3e}, tolerance={residual_tolerance:.3e}"
            )

        parent._local_self_warm_state = certified_module._pack_warm_state(
            local, local_geometry, port, fine_step, transverse
        )

        # Reconstruct the original physical field only after both compatible
        # components are independently resolved.  Keep the high/low expansion
        # for raw full-field contractions; localized truth is contracted from
        # the directly solved transverse field below.
        field = _combine_split_field(longitudinal, transverse)
        if not field_is_finite(field):
            raise FloatingPointError("local self-correction reconstructed full field is invalid")

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
            longitudinal=longitudinal,
            transverse=transverse,
            projection=projection,
        )
        print(
            "local self localized response: "
            f"raw_z=({z.real:.3e},{z.imag:.3e}), "
            f"local_z=({localized['localized_z'].real:.3e},{localized['localized_z'].imag:.3e}), "
            f"raw_d={d:.3e}, local_d={localized['localized_d_vol']:.3e}, "
            f"long_z_im={localized['longitudinal_z'].imag:.3e}, path=direct-transverse",
            flush=True,
        )

        elapsed_total = float(time.perf_counter() - started)
        print(
            "local Maxwell certified split: "
            f"edges={local.n_edges}, transverse_residual={transverse_residual:.3e}, "
            f"physical_residual={physical_residual:.3e}, total={elapsed_total:.1f}s",
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
            "linear_initial_relative_residual": float(initial_physical_residual),
            "linear_relative_residual": float(physical_residual),
            "linear_transverse_initial_relative_residual": float(initial_transverse_residual),
            "linear_transverse_relative_residual": float(transverse_residual),
            "linear_transverse_rhs_relative_norm": rhs_ratio,
            "linear_split_gradient_leakage_relative_norm": gradient_leakage,
            "linear_split_accumulation_mode": str(split_diagnostics.get("accumulation_mode", "unknown")),
            "linear_residual_accumulation_mode": str(residual_diagnostics.get("accumulation_mode", "unknown")),
            "linear_relative_residual_tolerance": float(residual_tolerance),
            "linear_refinement_iterations": int(refinements),
            "linear_solver_converged": bool(transverse_residual <= residual_tolerance),
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
