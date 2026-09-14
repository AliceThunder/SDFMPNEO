"""Numerically certified local-self Maxwell solve.

This patch keeps the existing local-self geometry and postprocessing but refuses
to silently treat a poorly solved refined Maxwell system as a mesh-convergence
reference.  SuperLU is reused for residual-correction iterations and every
result carries an explicit linear-solve certificate consumed by the audit.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse.linalg as spla


def _relative_residual(A, field, rhs):
    return float(
        np.linalg.norm(rhs - A @ field)
        / max(float(np.linalg.norm(rhs)), np.finfo(float).tiny)
    )


def install(self_correction_module):
    if bool(getattr(self_correction_module, "_certified_local_solve_installed", False)):
        return self_correction_module

    def solve_local(parent, geometry, port, fine_step, phi=None):
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
        refinement_steps = int(cfg.get("linear_refinement_steps", 5))
        if residual_tolerance <= 0.0:
            raise ValueError("self_correction.linear_relative_residual_tolerance must be positive")
        if refinement_steps < 0:
            raise ValueError("self_correction.linear_refinement_steps must be nonnegative")

        lu = None
        try:
            lu = spla.splu(A.tocsc())
            field = np.asarray(lu.solve(rhs), complex).reshape(-1)
        except RuntimeError:
            field = np.asarray(spla.spsolve(A, rhs), complex).reshape(-1)
        if np.any(~np.isfinite(field)):
            raise FloatingPointError("local self-correction Maxwell solve produced non-finite fields")

        initial_residual = _relative_residual(A, field, rhs)
        residual = initial_residual
        refinements = 0
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
                if not np.isfinite(candidate_residual):
                    break
                # Accept any real improvement.  Stop instead of adding a correction
                # that increases backward error on an extremely ill-conditioned grid.
                if candidate_residual >= residual:
                    break
                field = candidate
                residual = candidate_residual
                refinements += 1

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
        return {
            "z": z,
            "d_vol": d,
            "d_out": d_out,
            "modal_h": modal,
            "linear_initial_relative_residual": float(initial_residual),
            "linear_relative_residual": float(residual),
            "linear_relative_residual_tolerance": float(residual_tolerance),
            "linear_refinement_iterations": int(refinements),
            "linear_solver_converged": bool(residual <= residual_tolerance),
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
