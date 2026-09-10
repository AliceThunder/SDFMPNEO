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
        self.depth = int(depth)
        self.channels_per_mode = int(channels_per_mode)
        self.linear_rank = int(linear_rank)
        self.hidden_rank = int(hidden_rank)
        self.quadratic_rank = int(quadratic_rank)
        self.square_rank = int(square_rank)
        self.cross_rank = int(cross_rank)
        self.state_rank = int(state_rank)
        if min(
            self.depth, self.channels_per_mode, self.linear_rank, self.hidden_rank,
            self.quadratic_rank, self.square_rank, self.cross_rank, self.state_rank,
        ) < 1:
            raise ValueError("network depths/ranks must be positive")
        self.width = self.n_modes * self.channels_per_mode
        self.input_dimension = self.n_modes + len(self.operating_names)
        # Nonlinear source factors use one constant feature plus only static
        # geometry/operating coordinates on one side.  This prevents dense
        # thermal-state projections from multiplying each other and creating
        # O(n_modes^2) exponential terms at high thermal rank.
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
        self.targets = np.repeat(np.arange(self.n_modes, dtype=int), self.channels_per_mode)
        self._slices = {}
        offset = 0

        def allocate(name, shape):
            nonlocal offset
            size = int(np.prod(shape))
            self._slices[name] = (slice(offset, offset + size), tuple(shape))
            offset += size

        for layer in range(self.depth):
            allocate(f"bias_{layer}", (self.width,))
        allocate("input_linear_in", (self.linear_rank, self.input_dimension))
        allocate("input_linear_out", (self.width, self.linear_rank))
        allocate("quadratic_u", (self.quadratic_rank, self.input_dimension))
        allocate("quadratic_v", (self.quadratic_rank, self.static_dimension))
        allocate("quadratic_out", (self.width, self.quadratic_rank))
        allocate("square_in", (self.square_rank, self.n_modes))
        allocate("square_out", (self.width, self.square_rank))
        for layer in range(1, self.depth):
            allocate(f"hidden_linear_in_{layer}", (self.hidden_rank, self.width))
            allocate(f"hidden_linear_out_{layer}", (self.width, self.hidden_rank))
            allocate(f"cross_input_{layer}", (self.cross_rank, self.static_dimension))
            allocate(f"cross_hidden_{layer}", (self.cross_rank, self.width))
            allocate(f"cross_out_{layer}", (self.width, self.cross_rank))
            allocate(f"state_u_{layer}", (self.state_rank, self.width))
            allocate(f"state_v_{layer}", (self.state_rank, self.width))
            allocate(f"state_out_{layer}", (self.width, self.state_rank))
        allocate("channel_gate", (self.depth, self.width))
        allocate("quadratic_gate", (self.quadratic_rank,))
        allocate("square_gate", (self.square_rank,))
        for layer in range(1, self.depth):
            allocate(f"cross_gate_{layer}", (self.cross_rank,))
            allocate(f"state_gate_{layer}", (self.state_rank,))
        self.parameter_count = offset

        if parameters is None:
            rng = np.random.default_rng(seed)
            theta = np.zeros(offset)
            response_scale = 1e-4 * float(np.min(self.lambdas))
            # Factor inputs are order-one normalized projections; output factors
            # carry the small initial source amplitude so every factor has a
            # nonzero tangent without producing a large initial response.
            for name, dimension in (
                ("input_linear_in", self.input_dimension),
                ("quadratic_u", self.input_dimension),
                ("quadratic_v", self.static_dimension),
                ("square_in", self.n_modes),
            ):
                sl, shape = self._slices[name]
                theta[sl] = rng.normal(
                    scale=1 / math.sqrt(max(1, dimension)), size=shape
                ).reshape(-1)
            for name in ("input_linear_out", "quadratic_out", "square_out"):
                sl, shape = self._slices[name]
                theta[sl] = rng.normal(scale=response_scale, size=shape).reshape(-1)
            for layer in range(1, self.depth):
                for name, dimension in (
                    (f"hidden_linear_in_{layer}", self.width),
                    (f"cross_input_{layer}", self.static_dimension),
                    (f"cross_hidden_{layer}", self.width),
                    (f"state_u_{layer}", self.width),
                    (f"state_v_{layer}", self.width),
                ):
                    sl, shape = self._slices[name]
                    theta[sl] = rng.normal(
                        scale=1 / math.sqrt(max(1, dimension)), size=shape
                    ).reshape(-1)
                for name in (
                    f"hidden_linear_out_{layer}", f"cross_out_{layer}",
                    f"state_out_{layer}",
                ):
                    sl, shape = self._slices[name]
                    theta[sl] = rng.normal(scale=response_scale, size=shape).reshape(-1)
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

        self._logical_nodes = tuple(
            _LogicalNode(f"fixed_l{layer}_c{channel}", int(self.targets[channel]))
            for layer in range(self.depth)
            for channel in range(self.width)
        )



__all__ = ["_NetworkInitMixin"]
