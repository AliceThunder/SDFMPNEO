"""Network allocation and initialization for the fixed analytic response model."""
from __future__ import annotations

import math
import numpy as np

from .fixed_response_basis import _LogicalNode


class _NetworkInitMixin:
    def __init__(
        self,
        lambdas,
        operating_names,
        *,
        max_response_time,
        input_center=None,
        input_scale=None,
        depth=3,
        channels_per_mode=1,
        linear_rank=2,
        hidden_rank=2,
        quadratic_rank=3,
        square_rank=2,
        cross_rank=2,
        state_rank=2,
        layer_widths=None,
        layer_targets=None,
        layer_hidden_ranks=None,
        layer_cross_ranks=None,
        layer_state_ranks=None,
        state_feature_term_budget=24,
        parameters=None,
        seed=23,
    ):
        self.lambdas = np.asarray(lambdas, float).reshape(-1)
        if self.lambdas.size == 0 or np.any(~np.isfinite(self.lambdas)) or np.any(self.lambdas <= 0):
            raise ValueError("thermal decay rates must be finite and positive")
        self.max_response_time = float(max_response_time)
        if not np.isfinite(self.max_response_time) or self.max_response_time <= 0:
            raise ValueError("max_response_time must be finite and positive")
        self.operating_names = tuple(str(v) for v in operating_names)
        self.initial_names = tuple(f"a0_{i}" for i in range(self.lambdas.size))
        self.channels_per_mode = int(channels_per_mode)
        self.linear_rank = int(linear_rank)
        self.quadratic_rank = int(quadratic_rank)
        self.square_rank = int(square_rank)
        if min(self.channels_per_mode, self.linear_rank, self.quadratic_rank, self.square_rank) < 1:
            raise ValueError("first-layer response ranks must be positive")

        requested_depth = int(depth)
        if requested_depth < 1:
            raise ValueError("network depth must be positive")
        full_width = self.n_modes * self.channels_per_mode
        if layer_widths is None:
            widths = (full_width,) * requested_depth
        else:
            widths = tuple(int(v) for v in layer_widths)
            if len(widths) != requested_depth:
                raise ValueError("layer_widths must contain one width per response layer")
            if not widths or any(v < 1 or v > full_width for v in widths):
                raise ValueError("invalid response layer width")
            if widths[0] != full_width:
                raise ValueError("first response layer must cover every thermal target channel")
        self.depth = len(widths)
        self.layer_widths = widths
        # ``width`` remains the first/full response width for v6 compatibility.
        self.width = widths[0]

        def _ranks(values, fallback, name):
            if self.depth == 1:
                return ()
            if values is None:
                result = (int(fallback),) * (self.depth - 1)
            else:
                result = tuple(int(v) for v in values)
                if len(result) != self.depth - 1:
                    raise ValueError(f"{name} must contain depth-1 entries")
            if any(v < 0 for v in result):
                raise ValueError(f"{name} entries must be non-negative")
            return result

        self.layer_hidden_ranks = _ranks(layer_hidden_ranks, hidden_rank, "layer_hidden_ranks")
        self.layer_cross_ranks = _ranks(layer_cross_ranks, cross_rank, "layer_cross_ranks")
        self.layer_state_ranks = _ranks(layer_state_ranks, state_rank, "layer_state_ranks")
        # Legacy scalar attributes are retained for metadata/tests that inspect them.
        self.hidden_rank = int(max(self.layer_hidden_ranks, default=int(hidden_rank)))
        self.cross_rank = int(max(self.layer_cross_ranks, default=int(cross_rank)))
        self.state_rank = int(max(self.layer_state_ranks, default=int(state_rank)))
        self.state_feature_term_budget = int(state_feature_term_budget)
        if self.state_feature_term_budget < 1:
            raise ValueError("state_feature_term_budget must be positive")

        if layer_targets is None:
            targets = [np.repeat(np.arange(self.n_modes, dtype=int), self.channels_per_mode)]
            for width in widths[1:]:
                # Deeper layers are residual correctors. Their initial targets are
                # placeholders only; fresh training replaces them using modal residual
                # energy before the layer is activated.
                targets.append(np.arange(width, dtype=int) % self.n_modes)
        else:
            if len(layer_targets) != self.depth:
                raise ValueError("layer_targets must contain one target list per layer")
            targets = [np.asarray(v, dtype=int).reshape(-1) for v in layer_targets]
        for layer, (target, width) in enumerate(zip(targets, widths)):
            if target.shape != (width,) or np.any(target < 0) or np.any(target >= self.n_modes):
                raise ValueError(f"invalid target modes for response layer {layer}")
            if layer == 0:
                expected = np.repeat(np.arange(self.n_modes, dtype=int), self.channels_per_mode)
                if not np.array_equal(target, expected):
                    raise ValueError("first response layer targets must cover all thermal modes")
        self.layer_targets = tuple(np.asarray(v, dtype=int) for v in targets)
        self.targets = self.layer_targets[0]

        self.input_dimension = self.n_modes + len(self.operating_names)
        # Nonlinear source factors use one constant feature plus only static
        # geometry/operating coordinates on one side. This keeps the source
        # factorization O(r*k), not O(r^2).
        self.static_dimension = 1 + len(self.operating_names)
        if input_center is None:
            input_center = np.zeros(self.input_dimension)
        if input_scale is None:
            input_scale = np.ones(self.input_dimension)
        self.input_center = np.asarray(input_center, float).reshape(-1)
        self.input_scale = np.asarray(input_scale, float).reshape(-1)
        if (
            self.input_center.shape != (self.input_dimension,)
            or self.input_scale.shape != (self.input_dimension,)
            or np.any(~np.isfinite(self.input_center + self.input_scale))
            or np.any(self.input_scale <= 0)
        ):
            raise ValueError("invalid input normalization")

        self._slices = {}
        offset = 0

        def allocate(name, shape):
            nonlocal offset
            shape = tuple(int(v) for v in shape)
            size = int(np.prod(shape))
            self._slices[name] = (slice(offset, offset + size), shape)
            offset += size

        for layer, width in enumerate(self.layer_widths):
            allocate(f"bias_{layer}", (width,))
        allocate("input_linear_in", (self.linear_rank, self.input_dimension))
        allocate("input_linear_out", (self.layer_widths[0], self.linear_rank))
        allocate("quadratic_u", (self.quadratic_rank, self.input_dimension))
        allocate("quadratic_v", (self.quadratic_rank, self.static_dimension))
        allocate("quadratic_out", (self.layer_widths[0], self.quadratic_rank))
        allocate("square_in", (self.square_rank, self.n_modes))
        allocate("square_out", (self.layer_widths[0], self.square_rank))
        for layer in range(1, self.depth):
            previous_width = self.layer_widths[layer - 1]
            current_width = self.layer_widths[layer]
            hr = self.layer_hidden_ranks[layer - 1]
            cr = self.layer_cross_ranks[layer - 1]
            sr = self.layer_state_ranks[layer - 1]
            allocate(f"hidden_linear_in_{layer}", (hr, previous_width))
            allocate(f"hidden_linear_out_{layer}", (current_width, hr))
            allocate(f"cross_input_{layer}", (cr, self.static_dimension))
            allocate(f"cross_hidden_{layer}", (cr, previous_width))
            allocate(f"cross_out_{layer}", (current_width, cr))
            allocate(f"state_u_{layer}", (sr, previous_width))
            allocate(f"state_v_{layer}", (sr, previous_width))
            allocate(f"state_out_{layer}", (current_width, sr))
        # Keep the legacy rectangular gate block so uniform-width v6 parameter
        # vectors retain their exact layout. Entries beyond a funnel layer width
        # are unused and remain one.
        allocate("channel_gate", (self.depth, self.width))
        allocate("quadratic_gate", (self.quadratic_rank,))
        allocate("square_gate", (self.square_rank,))
        for layer in range(1, self.depth):
            allocate(f"cross_gate_{layer}", (self.layer_cross_ranks[layer - 1],))
            allocate(f"state_gate_{layer}", (self.layer_state_ranks[layer - 1],))
        self.parameter_count = offset

        if parameters is None:
            rng = np.random.default_rng(seed)
            theta = np.zeros(offset)
            response_scale = 1e-4 * float(np.min(self.lambdas))
            for name, dimension in (
                ("input_linear_in", self.input_dimension),
                ("quadratic_u", self.input_dimension),
                ("quadratic_v", self.static_dimension),
                ("square_in", self.n_modes),
            ):
                sl, shape = self._slices[name]
                if int(np.prod(shape)):
                    theta[sl] = rng.normal(
                        scale=1 / math.sqrt(max(1, dimension)), size=shape
                    ).reshape(-1)
            # Layer-1 amplitudes start small and are immediately replaced by the
            # t=0 source least-squares prefit during fresh training.
            for name in ("input_linear_out", "quadratic_out", "square_out"):
                sl, shape = self._slices[name]
                theta[sl] = rng.normal(scale=response_scale, size=shape).reshape(-1)
            # Deeper response layers are true residual correctors: their factor
            # banks are initialized, but amplitudes/biases stay exactly zero until
            # the preceding layer has been trained and residual targets selected.
            for layer in range(1, self.depth):
                for name, dimension in (
                    (f"hidden_linear_in_{layer}", self.layer_widths[layer - 1]),
                    (f"cross_input_{layer}", self.static_dimension),
                    (f"cross_hidden_{layer}", self.layer_widths[layer - 1]),
                    (f"state_u_{layer}", self.layer_widths[layer - 1]),
                    (f"state_v_{layer}", self.layer_widths[layer - 1]),
                ):
                    sl, shape = self._slices[name]
                    if int(np.prod(shape)):
                        theta[sl] = rng.normal(
                            scale=1 / math.sqrt(max(1, dimension)), size=shape
                        ).reshape(-1)
                # *_out and bias_* are intentionally left at zero.
            theta[self._slices["channel_gate"][0]] = 1.0
            theta[self._slices["quadratic_gate"][0]] = 1.0
            theta[self._slices["square_gate"][0]] = 1.0
            for layer in range(1, self.depth):
                theta[self._slices[f"cross_gate_{layer}"][0]] = 1.0
                theta[self._slices[f"state_gate_{layer}"][0]] = 1.0
            self.parameters = theta
        else:
            values = np.asarray(parameters, float).reshape(-1)
            if values.shape != (offset,) or np.any(~np.isfinite(values)):
                raise ValueError("network parameter vector has wrong shape or non-finite values")
            self.parameters = values.copy()

        nodes = []
        for layer, (width, targets_l) in enumerate(zip(self.layer_widths, self.layer_targets)):
            for channel in range(width):
                nodes.append(
                    _LogicalNode(
                        f"fixed_l{layer}_c{channel}", int(targets_l[channel])
                    )
                )
        self._logical_nodes = tuple(nodes)


__all__ = ["_NetworkInitMixin"]
