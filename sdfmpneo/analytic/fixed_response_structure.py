"""Parameter access and continuous structure controls."""
from __future__ import annotations

import math
import numpy as np


class _NetworkStructureMixin:
    @property
    def n_modes(self):
        return int(self.lambdas.size)

    @property
    def response_nodes(self):
        gates = self._array("channel_gate")
        return tuple(
            node for i, node in enumerate(self._logical_nodes)
            if gates[i // self.width, i % self.width] != 0.0
        )

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
            parameters=parameters,
        )

    def amplitude_parameter_indices(self):
        """Indices of coefficients linear in the current feature realization.

        These parameters form a well-conditioned warm-start block: biases and
        output amplitudes can fit the dominant forcing before nonlinear factor
        directions are allowed to move.  Structure gates stay frozen in this
        block to avoid gate/output scale degeneracy.
        """
        ids = []
        for name, (sl, _shape) in self._slices.items():
            if name.startswith("bias_") or name.endswith("_out"):
                ids.extend(range(sl.start, sl.stop))
        return np.asarray(ids, dtype=int)

    def structure_gate_entries(self):
        entries = []
        for layer in range(self.depth):
            values = self._array("channel_gate")[layer]
            ids = self._indices("channel_gate")[layer]
            for c in range(self.width):
                entries.append({
                    "kind": "channel", "layer": layer, "index": c,
                    "target_mode": int(self.targets[c]),
                    "parameter_index": int(ids[c]), "value": float(values[c]),
                })
        spec = [("quadratic", 0, self.quadratic_rank), ("square", 0, self.square_rank)] + [
            (kind, layer, rank)
            for layer in range(1, self.depth)
            for kind, rank in (("cross", self.cross_rank), ("state", self.state_rank))
        ]
        for kind, layer, rank in spec:
            if kind in {"quadratic", "square"}:
                name = f"{kind}_gate"
            else:
                name = f"{kind}_gate_{layer}"
            values, ids = self._array(name), self._indices(name)
            for r in range(rank):
                entries.append({
                    "kind": kind, "layer": layer, "index": r,
                    "parameter_index": int(ids[r]), "value": float(values[r]),
                })
        return tuple(entries)

    def structure_summary(self, threshold=0.0):
        threshold = float(threshold)
        channel = np.abs(self._array("channel_gate")) > threshold
        by_layer = [int(np.count_nonzero(row)) for row in channel]
        active_layers = [i for i, count in enumerate(by_layer) if count]
        per_mode = [
            int(np.count_nonzero(channel[:, self.targets == mode]))
            for mode in range(self.n_modes)
        ]
        return {
            "maximum_depth": self.depth,
            "effective_depth": 0 if not active_layers else max(active_layers) + 1,
            "maximum_channels_per_mode": self.channels_per_mode,
            "linear_rank": self.linear_rank,
            "hidden_rank": self.hidden_rank,
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
            "parameter_count": self.parameter_count,
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
