"""Activation shortcuts for funnel-shaped response layers."""
from __future__ import annotations

import numpy as np

from .fixed_response_basis import _ExpPoly


class _LayerActivationMixin:
    def _layer_is_zero(self, layer):
        layer = int(layer)
        if np.any(self._array(f"bias_{layer}")):
            return False
        if layer == 0:
            names = ("input_linear_out", "quadratic_out", "square_out")
        else:
            names = (
                f"hidden_linear_out_{layer}",
                f"cross_out_{layer}",
                f"state_out_{layer}",
            )
        return all(not np.any(self._array(name)) for name in names)

    def _make_layer(self, base, static, previous, layer, derivative_kind, nd):
        # Parameter derivatives of a zero-amplitude layer are not zero, so only
        # value/input-Jacobian paths may skip feature construction. High-rank
        # training uses the dedicated layer-amplitude Jacobian for activation.
        if derivative_kind != "parameter" and self._layer_is_zero(layer):
            return tuple(
                _ExpPoly.zero(self.n_modes, nd)
                for _ in range(self.layer_widths[layer])
            )
        return super()._make_layer(
            base, static, previous, layer, derivative_kind, nd
        )


__all__ = ["_LayerActivationMixin"]
