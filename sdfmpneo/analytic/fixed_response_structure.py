"""Parameter access and continuous structure controls."""
from __future__ import annotations

import math
import numpy as np


class _NetworkStructureMixin:
    @property
    def n_modes(self):
        return int(self.lambdas.size)

    def _channel_amplitude_strength(self, layer, channel):
        layer = int(layer); channel = int(channel)
        values = [float(self._array(f"bias_{layer}")[channel])]
        if layer == 0:
            names = ("input_linear_out", "quadratic_out", "square_out")
        else:
            names = (
                f"hidden_linear_out_{layer}",
                f"cross_out_{layer}",
                f"state_out_{layer}",
            )
        for name in names:
            values.extend(np.asarray(self._array(name)[channel], float).reshape(-1).tolist())
        return float(np.linalg.norm(values))

    def _channel_is_active(self, layer, channel, threshold=0.0):
        if self._array("channel_gate")[int(layer), int(channel)] == 0.0:
            return False
        return self._channel_amplitude_strength(layer, channel) > float(threshold)

    @property
    def response_nodes(self):
        result = []
        offset = 0
        for layer, width in enumerate(self.layer_widths):
            for channel in range(width):
                if self._channel_is_active(layer, channel, 0.0):
                    result.append(self._logical_nodes[offset + channel])
            offset += width
        return tuple(result)

    def _array(self, name):
        sl, shape = self._slices[name]
        return self.parameters[sl].reshape(shape)

    def _indices(self, name):
        sl, shape = self._slices[name]
        return np.arange(sl.start, sl.stop).reshape(shape)

    def with_parameters(self, parameters):
        return type(self)(
            self.lambdas, self.operating_names,
            max_response_time=self.max_response_time,
            input_center=self.input_center,
            input_scale=self.input_scale,
            depth=self.depth,
            channels_per_mode=self.channels_per_mode,
            linear_rank=self.linear_rank,
            hidden_rank=self.hidden_rank,
            quadratic_rank=self.quadratic_rank,
            square_rank=self.square_rank,
            cross_rank=self.cross_rank,
            state_rank=self.state_rank,
            layer_widths=self.layer_widths,
            layer_targets=tuple(v.copy() for v in self.layer_targets),
            layer_hidden_ranks=self.layer_hidden_ranks,
            layer_cross_ranks=self.layer_cross_ranks,
            layer_state_ranks=self.layer_state_ranks,
            state_feature_term_budget=self.state_feature_term_budget,
            parameters=parameters,
        )

    def with_layer_targets(self, layer, targets):
        """Return an identical network with one residual-correction target set replaced."""
        layer = int(layer)
        if layer <= 0 or layer >= self.depth:
            raise ValueError("only deeper residual-correction targets may be replaced")
        values = np.asarray(targets, dtype=int).reshape(-1)
        if values.shape != (self.layer_widths[layer],):
            raise ValueError("target count does not match response layer width")
        if np.any(values < 0) or np.any(values >= self.n_modes):
            raise ValueError("response target mode is out of range")
        layers = [v.copy() for v in self.layer_targets]
        layers[layer] = values
        return type(self)(
            self.lambdas, self.operating_names,
            max_response_time=self.max_response_time,
            input_center=self.input_center,
            input_scale=self.input_scale,
            depth=self.depth,
            channels_per_mode=self.channels_per_mode,
            linear_rank=self.linear_rank,
            hidden_rank=self.hidden_rank,
            quadratic_rank=self.quadratic_rank,
            square_rank=self.square_rank,
            cross_rank=self.cross_rank,
            state_rank=self.state_rank,
            layer_widths=self.layer_widths,
            layer_targets=tuple(layers),
            layer_hidden_ranks=self.layer_hidden_ranks,
            layer_cross_ranks=self.layer_cross_ranks,
            layer_state_ranks=self.layer_state_ranks,
            state_feature_term_budget=self.state_feature_term_budget,
            parameters=self.parameters,
        )

    def layer_amplitude_parameter_indices(self, layer):
        """Return the stable linear-amplitude block for one response layer.

        Factor projections and all gates stay frozen during residual training.
        Every layer trains its constant source bias plus feature-output amplitudes;
        only the input/factor directions themselves are frozen after Stage 0.
        """
        layer = int(layer)
        if layer < 0 or layer >= self.depth:
            raise ValueError("response layer index is out of range")
        names = [f"bias_{layer}"]
        if layer == 0:
            names.extend(("input_linear_out", "quadratic_out", "square_out"))
        else:
            names.extend((
                f"hidden_linear_out_{layer}",
                f"cross_out_{layer}",
                f"state_out_{layer}",
            ))
        ids = []
        for name in names:
            sl, _ = self._slices[name]
            ids.extend(range(sl.start, sl.stop))
        return np.asarray(ids, dtype=int)

    def trainable_parameter_indices(self):
        """All trainable response amplitudes; structure/factor parameters stay frozen."""
        blocks = [self.layer_amplitude_parameter_indices(layer) for layer in range(self.depth)]
        return np.concatenate(blocks) if blocks else np.empty(0, dtype=int)

    def amplitude_parameter_indices(self):
        return self.trainable_parameter_indices()

    def zero_layer_amplitudes(self, layer):
        layer = int(layer)
        theta = self.parameters.copy()
        theta[self.layer_amplitude_parameter_indices(layer)] = 0.0
        return self.with_parameters(theta)

    def structure_gate_entries(self):
        entries = []
        gates = self._array("channel_gate")
        gate_ids = self._indices("channel_gate")
        for layer, width in enumerate(self.layer_widths):
            for c in range(width):
                entries.append({
                    "kind": "channel", "layer": layer, "index": c,
                    "target_mode": int(self.layer_targets[layer][c]),
                    "parameter_index": int(gate_ids[layer, c]),
                    "value": float(gates[layer, c]),
                })
        spec = [("quadratic", 0, self.quadratic_rank), ("square", 0, self.square_rank)]
        for layer in range(1, self.depth):
            spec.extend((
                ("cross", layer, self.layer_cross_ranks[layer - 1]),
                ("state", layer, self.layer_state_ranks[layer - 1]),
            ))
        for kind, layer, rank in spec:
            if rank == 0:
                continue
            name = f"{kind}_gate" if kind in {"quadratic", "square"} else f"{kind}_gate_{layer}"
            values, ids = self._array(name), self._indices(name)
            for r in range(rank):
                entries.append({
                    "kind": kind, "layer": layer, "index": r,
                    "parameter_index": int(ids[r]), "value": float(values[r]),
                })
        return tuple(entries)

    def structure_summary(self, threshold=0.0):
        threshold = float(threshold)
        by_layer = []
        for layer, width in enumerate(self.layer_widths):
            by_layer.append(sum(
                1 for channel in range(width)
                if self._channel_is_active(layer, channel, threshold)
            ))
        active_layers = [i for i, count in enumerate(by_layer) if count]
        per_mode = []
        for mode in range(self.n_modes):
            count = 0
            for layer, width in enumerate(self.layer_widths):
                for channel in range(width):
                    if (
                        int(self.layer_targets[layer][channel]) == mode
                        and self._channel_is_active(layer, channel, threshold)
                    ):
                        count += 1
            per_mode.append(count)
        return {
            "maximum_depth": self.depth,
            "effective_depth": 0 if not active_layers else max(active_layers) + 1,
            "maximum_channels_per_mode": self.channels_per_mode,
            "layer_widths": list(self.layer_widths),
            "layer_targets": [v.tolist() for v in self.layer_targets],
            "linear_rank": self.linear_rank,
            "hidden_rank": self.hidden_rank,
            "hidden_rank_by_layer": list(self.layer_hidden_ranks),
            "active_channels_by_layer": by_layer,
            "active_channels_per_mode": per_mode,
            "quadratic_rank": int(np.count_nonzero(np.abs(self._array("quadratic_gate")) > threshold)),
            "square_rank": int(np.count_nonzero(np.abs(self._array("square_gate")) > threshold)),
            "cross_rank_by_layer": [
                int(np.count_nonzero(np.abs(self._array(f"cross_gate_{l}")) > threshold))
                for l in range(1, self.depth)
            ],
            "state_rank_by_layer": [
                int(np.count_nonzero(np.abs(self._array(f"state_gate_{l}")) > threshold))
                for l in range(1, self.depth)
            ],
            "state_feature_term_budget": self.state_feature_term_budget,
            "parameter_count": self.parameter_count,
            "trainable_amplitude_parameter_count": int(self.trainable_parameter_indices().size),
            "active_response_channels": int(sum(by_layer)),
            "max_response_time": self.max_response_time,
        }

    def soft_threshold_structure(self, amount):
        amount = float(amount)
        if not np.isfinite(amount) or amount < 0:
            raise ValueError("invalid structure shrink")
        theta = self.parameters.copy()
        for entry in self.structure_gate_entries():
            i = entry["parameter_index"]
            theta[i] = math.copysign(max(0.0, abs(theta[i]) - amount), theta[i])
        return self.with_parameters(theta)


__all__ = ["_NetworkStructureMixin"]
