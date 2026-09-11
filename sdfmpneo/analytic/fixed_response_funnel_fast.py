"""Fast-path routing for funnel-shaped multilayer response networks."""
from __future__ import annotations

import numpy as np


class _FunnelFastMixin:
    def _deeper_layers_zero(self):
        return self.depth <= 1 or all(
            self._layer_is_zero(layer) for layer in range(1, self.depth)
        )

    def evaluate_with_jacobians(self, t, *, a0, operating, derivative_kind):
        # While residual-correction layers are still zero, the depth-three model
        # is exactly the first response layer. Preserve the vectorized high-rank
        # evaluator for values and input sensitivities. Full parameter tangents
        # are excluded because zero deeper amplitudes still have nonzero tangent.
        if (
            self.depth > 1
            and derivative_kind != "parameter"
            and self._deeper_layers_zero()
        ):
            return self._depth_one_evaluate(
                t, a0=a0, operating=operating, derivative_kind=derivative_kind
            )
        return super().evaluate_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind=derivative_kind
        )

    def evaluate_layer_amplitude_jacobian(self, t, *, a0, operating, layer):
        layer = int(layer)
        if layer == 0 and self.depth > 1 and self._deeper_layers_zero():
            a, da, ja, jda = self._depth_one_evaluate(
                t, a0=a0, operating=operating, derivative_kind="parameter"
            )
            ids = np.asarray(self.layer_amplitude_parameter_indices(0), dtype=int)
            return a, da, ja[:, ids], jda[:, ids], ids
        return super().evaluate_layer_amplitude_jacobian(
            t, a0=a0, operating=operating, layer=layer
        )


__all__ = ["_FunnelFastMixin"]
