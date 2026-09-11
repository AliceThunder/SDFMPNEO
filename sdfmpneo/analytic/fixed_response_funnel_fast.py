"""Fast-path routing for funnel-shaped multilayer response networks."""
from __future__ import annotations

import numpy as np


class _FunnelFastMixin:
    def _deeper_layers_zero(self):
        return self.depth <= 1 or all(
            self._layer_is_zero(layer) for layer in range(1, self.depth)
        )

    def evaluate_with_jacobians(self, t, *, a0, operating, derivative_kind):
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

    def _depth_one_amplitude_jacobian(self, t, *, a0, operating):
        """Vectorized exact tangent of Layer 1's linear amplitude block only."""
        data = self._depth_one_data(float(t), a0, operating)
        initial = data["initial"]
        target = self.layer_targets[0]
        gate = np.asarray(self._array("channel_gate")[0, :self.layer_widths[0]], float)
        decay = np.exp(-self.lambdas * float(t))
        a = initial * decay
        da = -self.lambdas * a
        np.add.at(a, target, gate * data["y"])
        np.add.at(da, target, gate * data["dy"])

        ids = np.asarray(self.layer_amplitude_parameter_indices(0), dtype=int)
        p = len(ids)
        ja = np.zeros((self.n_modes, p), dtype=float)
        jda = np.zeros_like(ja)
        width = self.layer_widths[0]
        cursor = 0

        # bias_0
        for c in range(width):
            mode = int(target[c])
            ja[mode, cursor + c] += gate[c] * data["rzero"][c]
            jda[mode, cursor + c] += gate[c] * data["drzero"][c]
        cursor += width

        # input_linear_out
        for c in range(width):
            mode = int(target[c])
            sl = slice(cursor + c * self.linear_rank, cursor + (c + 1) * self.linear_rank)
            ja[mode, sl] += gate[c] * data["lin_resp"][c]
            jda[mode, sl] += gate[c] * data["lin_slope"][c]
        cursor += width * self.linear_rank

        # quadratic_out; feature gates are frozen but still part of the source.
        qgate = np.asarray(data["qgate"], float)
        for c in range(width):
            mode = int(target[c])
            sl = slice(cursor + c * self.quadratic_rank, cursor + (c + 1) * self.quadratic_rank)
            ja[mode, sl] += gate[c] * qgate * data["qresp"][c]
            jda[mode, sl] += gate[c] * qgate * data["qslope"][c]
        cursor += width * self.quadratic_rank

        # square_out
        sgate = np.asarray(data["sgate"], float)
        for c in range(width):
            mode = int(target[c])
            sl = slice(cursor + c * self.square_rank, cursor + (c + 1) * self.square_rank)
            ja[mode, sl] += gate[c] * sgate * data["sresp"][c]
            jda[mode, sl] += gate[c] * sgate * data["sslope"][c]
        cursor += width * self.square_rank
        if cursor != p:
            raise RuntimeError("Layer-1 amplitude layout no longer matches fast Jacobian")
        return a, da, ja, jda, ids

    def evaluate_layer_amplitude_jacobian(self, t, *, a0, operating, layer):
        layer = int(layer)
        if layer == 0 and self.depth > 1 and self._deeper_layers_zero():
            return self._depth_one_amplitude_jacobian(
                t, a0=a0, operating=operating
            )
        return super().evaluate_layer_amplitude_jacobian(
            t, a0=a0, operating=operating, layer=layer
        )


__all__ = ["_FunnelFastMixin"]
