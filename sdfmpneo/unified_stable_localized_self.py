"""Stable localized self-response contractions for high-dynamic-range port fields.

This patch preserves the v2 physical definition exactly.  The full open-port
Maxwell field and compatible longitudinal component are unchanged.  It only
changes how the localizable remainder is evaluated numerically: instead of
subtracting two large port impedances or two large squared field magnitudes, it
forms the transverse remainder with the existing high/low compensated field
representation and contracts that remainder directly.
"""
from __future__ import annotations

import numpy as np

from .unified_compensated_field import (
    compensated_add,
    field_abs2,
    field_linear_dot,
    field_norm,
    field_parts,
)
from .unified_gradient_block_maxwell import build_gradient_block


def install(self_correction_module):
    if bool(getattr(self_correction_module, "_stable_localized_self_installed", False)):
        return self_correction_module

    def localized_self_response(
        local,
        context,
        A,
        rhs,
        field,
        source,
        sigma,
        edge_loss,
        outward_weights,
        *,
        local_phi=None,
    ):
        """Evaluate the v2 localizable response without catastrophic subtraction.

        Let ``E = E_L + E_T`` where ``E_L`` is the exact compatible gradient
        component.  The v2 correction removes only the pure longitudinal self
        term.  Therefore

            Z_local = -S^T E_T

        and

            |E|^2 - |E_L|^2 = |E_T|^2 + 2 Re(E_L^* E_T).

        These are algebraically identical to the previous v2 formulas, but they
        remain well conditioned when the terminal field is many orders of
        magnitude larger than the localizable remainder.
        """
        del A  # The decomposition uses the already-certified physical field.
        gradient = build_gradient_block(local, context, check_topology=True)
        longitudinal = np.asarray(gradient.solve(rhs), complex).reshape(-1)
        if longitudinal.shape != (local.n_edges,) or np.any(~np.isfinite(longitudinal)):
            raise FloatingPointError("local Maxwell longitudinal field is invalid")

        # Error-free high/low subtraction of the dominant longitudinal field.
        transverse = compensated_add(field, longitudinal, scale=-1.0)
        if not np.isfinite(field_norm(transverse)):
            raise FloatingPointError("local Maxwell transverse remainder is invalid")

        # Port impedance is a linear output.  Contract the small remainder
        # directly rather than evaluating full_z - longitudinal_z.
        refinable_z = complex(-field_linear_dot(source, transverse))
        longitudinal_z = complex(-np.asarray(source, float) @ longitudinal)

        # Preserve the L/T cross term required by the v2 physical definition,
        # but avoid |E|^2 - |E_L|^2 cancellation.
        t_high, t_low = field_parts(transverse)
        transverse_abs2 = field_abs2(transverse)
        cross = 2.0 * (
            longitudinal.real * t_high.real
            + longitudinal.imag * t_high.imag
            + longitudinal.real * t_low.real
            + longitudinal.imag * t_low.imag
        )
        refinable_abs2 = np.asarray(transverse_abs2 + cross, float)
        if np.any(~np.isfinite(refinable_abs2)):
            raise FloatingPointError("localized Maxwell energy remainder is invalid")

        longitudinal_abs2 = np.abs(longitudinal) ** 2
        edge_loss = np.asarray(edge_loss, float)
        outward_weights = np.asarray(outward_weights, float)
        refinable_d = float(np.dot(edge_loss, refinable_abs2))
        longitudinal_d = float(np.dot(edge_loss, longitudinal_abs2))
        refinable_out = float(np.dot(outward_weights, refinable_abs2))
        longitudinal_out = float(np.dot(outward_weights, longitudinal_abs2))

        q_refinable = np.asarray(
            0.5
            * np.asarray(sigma, float)
            * np.asarray(local.edge_cell_hodge.T @ refinable_abs2).reshape(-1),
            float,
        )
        modal_refinable = None
        if local_phi is not None:
            modal_refinable = np.asarray(2.0 * (local_phi.T @ q_refinable), float)

        scale = max(
            abs(float(np.real(refinable_z))),
            abs(refinable_d) + abs(refinable_out),
            np.finfo(float).tiny,
        )
        balance = float(abs(refinable_z.real - refinable_d - refinable_out) / scale)
        full_norm = max(field_norm(field), np.finfo(float).tiny)
        return {
            "localized_z": refinable_z,
            "localized_d_vol": refinable_d,
            "localized_d_out": refinable_out,
            "localized_modal_h": modal_refinable,
            "localized_power_balance_relative_error": balance,
            "localized_contraction": "compensated_transverse_remainder_v2",
            "longitudinal_z": longitudinal_z,
            "longitudinal_d_vol": longitudinal_d,
            "longitudinal_d_out": longitudinal_out,
            "longitudinal_field_relative_norm": float(
                np.linalg.norm(longitudinal) / full_norm
            ),
            "transverse_field_relative_norm": float(field_norm(transverse) / full_norm),
        }

    self_correction_module._localized_self_response = localized_self_response
    self_correction_module._stable_localized_self_installed = True
    return self_correction_module


__all__ = ["install"]
