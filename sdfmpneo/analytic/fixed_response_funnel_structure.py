"""Activity semantics for frozen-gate funnel response networks."""
from __future__ import annotations

import numpy as np


class _FunnelStructureMixin:
    def _channel_amplitude(self, layer, channel):
        values = []
        if layer == 0:
            values.append(float(self._array("bias_0")[channel]))
            names = ("input_linear_out", "quadratic_out", "square_out")
        else:
            names = (
                f"hidden_linear_out_{layer}",
                f"cross_out_{layer}",
                f"state_out_{layer}",
            )
        for name in names:
            values.extend(np.asarray(self._array(name)[channel], float).reshape(-1))
        return float(np.linalg.norm(values))

    @property
    def response_nodes(self):
        gates = self._array("channel_gate")
        result = []
        offset = 0
        for layer, width in enumerate(self.layer_widths):
            for channel in range(width):
                if (
                    gates[layer, channel] != 0.0
                    and self._channel_amplitude(layer, channel) != 0.0
                ):
                    result.append(self._logical_nodes[offset + channel])
            offset += width
        return tuple(result)

    def structure_summary(self, threshold=0.0):
        summary = dict(super().structure_summary(threshold))
        threshold = float(threshold)
        gates = self._array("channel_gate")
        by_layer = []
        per_mode = [0] * self.n_modes
        for layer, width in enumerate(self.layer_widths):
            count = 0
            for channel in range(width):
                active = (
                    abs(float(gates[layer, channel])) > threshold
                    and self._channel_amplitude(layer, channel) > threshold
                )
                if active:
                    count += 1
                    per_mode[int(self.layer_targets[layer][channel])] += 1
            by_layer.append(count)
        active_layers = [i for i, count in enumerate(by_layer) if count]
        summary.update(
            effective_depth=0 if not active_layers else max(active_layers) + 1,
            active_channels_by_layer=by_layer,
            active_channels_per_mode=per_mode,
            active_response_channels=int(sum(by_layer)),
        )
        return summary


__all__ = ["_FunnelStructureMixin"]
