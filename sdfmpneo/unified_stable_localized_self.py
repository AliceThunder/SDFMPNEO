"""Stable localized self-response contractions for high-dynamic-range port fields.

This patch preserves the v2 physical definition exactly.  The full open-port
Maxwell field and compatible longitudinal component are unchanged.  Localized
truth extraction is made numerically robust in two places:

* the compatible scalar-gradient projection is certified and iteratively
  refined before the tiny transverse remainder is formed;
* the localizable port/loss outputs are contracted directly from a compensated
  transverse remainder instead of subtracting two large full-field quantities.
"""
from __future__ import annotations

import numpy as np

from .unified_compensated_field import (
    as_compensated_field,
    compensated_add,
    field_abs2,
    field_linear_dot,
    field_norm,
    field_parts,
)
from .unified_gradient_block_maxwell import build_gradient_block
from .unified_refined_gradient_projection import refined_gradient_projection


def _subtract_compensated(left, right):
    out = as_compensated_field(left, copy=True)
    right_high, right_low = field_parts(right)
    out = compensated_add(out, right_high, scale=-1.0)
    out = compensated_add(out, right_low, scale=-1.0)
    return out


def _cross_real(left, right):
    """Return 2 Re(left^H right) edgewise from two high/low expansions."""
    lh, ll = field_parts(left)
    rh, rl = field_parts(right)
    return 2.0 * np.asarray(
        lh.real * rh.real
        + lh.imag * rh.imag
        + lh.real * rl.real
        + lh.imag * rl.imag
        + ll.real * rh.real
        + ll.imag * rh.imag
        + ll.real * rl.real
        + ll.imag * rl.imag,
        dtype=float,
    )


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
        term.  For an exact Galerkin longitudinal projection,

            Z_local = -S^T E_T

        and

            |E|^2 - |E_L|^2 = |E_T|^2 + 2 Re(E_L^* E_T).

        The finest validation grid can have ``||E_T||/||E||`` of only a few
        parts per million, so the scalar projection itself is certified before
        these equivalent remainder contractions are evaluated.
        """
        del A  # Full-field physics has already been certified before extraction.
        gradient = build_gradient_block(local, context, check_topology=True)
        longitudinal, projection = refined_gradient_projection(
            local,
            gradient,
            rhs,
            relative_tolerance=5e-13,
            maximum_refinements=5,
        )
        if not np.isfinite(field_norm(longitudinal)):
            raise FloatingPointError("local Maxwell longitudinal field is invalid")

        transverse = _subtract_compensated(field, longitudinal)
        if not np.isfinite(field_norm(transverse)):
            raise FloatingPointError("local Maxwell transverse remainder is invalid")

        # The source contraction is algebraically equal to subtracting the pure
        # longitudinal impedance when the compatible projection is solved.  It
        # is far better conditioned because only the small remainder is paired.
        refinable_z = complex(-field_linear_dot(source, transverse))
        longitudinal_z = complex(-field_linear_dot(source, longitudinal))

        # Preserve the L/T cross term required by the v2 physical definition,
        # but avoid |E|^2 - |E_L|^2 cancellation.
        refinable_abs2 = np.asarray(
            field_abs2(transverse) + _cross_real(longitudinal, transverse),
            float,
        )
        if np.any(~np.isfinite(refinable_abs2)):
            raise FloatingPointError("localized Maxwell energy remainder is invalid")

        longitudinal_abs2 = field_abs2(longitudinal)
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
            "localized_contraction": "certified_gradient_compensated_transverse_v2",
            "localized_gradient_projection_initial_relative_residual": float(
                projection["initial_relative_residual"]
            ),
            "localized_gradient_projection_relative_residual": float(
                projection["relative_residual"]
            ),
            "localized_gradient_projection_refinements": int(projection["refinements"]),
            "localized_gradient_projection_edge_low_relative_norm": float(
                projection["edge_low_relative_norm"]
            ),
            "longitudinal_z": longitudinal_z,
            "longitudinal_d_vol": longitudinal_d,
            "longitudinal_d_out": longitudinal_out,
            "longitudinal_field_relative_norm": float(field_norm(longitudinal) / full_norm),
            "transverse_field_relative_norm": float(field_norm(transverse) / full_norm),
        }

    self_correction_module._localized_self_response = localized_self_response
    self_correction_module._stable_localized_self_installed = True
    return self_correction_module


__all__ = ["install"]
