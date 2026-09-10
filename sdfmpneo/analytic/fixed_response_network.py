from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class _LogicalNode:
    name: str
    target_mode: int


class _ExpPoly:
    """Exact finite sums of c*t^k*exp(-mu*t), with optional dense derivatives."""
    def __init__(self, n_modes, nd=0):
        self.n_modes = int(n_modes)
        self.nd = int(nd)
        self.terms = {}

    @classmethod
    def term(cls, counts, coefficient, *, nd=0, derivative=None):
        out = cls(len(counts), nd)
        out.add_term(tuple(int(v) for v in counts), 0, coefficient, derivative)
        return out

    @classmethod
    def zero(cls, n_modes, nd=0):
        return cls(n_modes, nd)

    def add_term(self, counts, power, coefficient, derivative=None):
        counts, power, coefficient = tuple(counts), int(power), float(coefficient)
        gradient = np.zeros(self.nd) if derivative is None else np.asarray(derivative, float)
        if len(counts) != self.n_modes or power < 0 or gradient.shape != (self.nd,):
            raise ValueError("invalid exponential-polynomial term")
        key = counts, power
        old_c, old_g = self.terms.get(key, (0.0, np.zeros(self.nd)))
        coefficient += old_c
        gradient = gradient + old_g
        if coefficient == 0.0 and not np.any(gradient):
            self.terms.pop(key, None)
        else:
            self.terms[key] = coefficient, gradient
        return self

    def add_scaled(self, other, scale, parameter_index=None):
        if self.n_modes != other.n_modes or self.nd != other.nd:
            raise ValueError("signal dimensions do not match")
        scale = float(scale)
        for (counts, power), (coefficient, gradient) in other.terms.items():
            local = scale * gradient
            if parameter_index is not None:
                local = local.copy()
                local[int(parameter_index)] += coefficient
            self.add_term(counts, power, scale * coefficient, local)
        return self

    def product(self, other):
        if self.n_modes != other.n_modes or self.nd != other.nd:
            raise ValueError("signal dimensions do not match")
        out = _ExpPoly.zero(self.n_modes, self.nd)
        for (ca, pa), (va, ga) in self.terms.items():
            for (cb, pb), (vb, gb) in other.terms.items():
                out.add_term(
                    tuple(a + b for a, b in zip(ca, cb)),
                    pa + pb,
                    va * vb,
                    ga * vb + gb * va,
                )
        return out

    def response(self, target, lambdas):
        rates = np.asarray(lambdas, float)
        target = int(target)
        lam = float(rates[target])
        target_counts = tuple(1 if i == target else 0 for i in range(self.n_modes))
        out = _ExpPoly.zero(self.n_modes, self.nd)
        for (counts, power), (coefficient, gradient) in self.terms.items():
            mu = float(np.dot(np.asarray(counts, float), rates))
            if lam == mu:
                factor = 1.0 / (power + 1)
                out.add_term(counts, power + 1, factor * coefficient, factor * gradient)
                continue
            delta = lam - mu
            factorial = math.factorial(power)
            for m in range(power + 1):
                p = power - m
                factor = ((-1.0) ** m) * factorial / math.factorial(p) / delta ** (m + 1)
                out.add_term(counts, p, factor * coefficient, factor * gradient)
            tail = -(((-1.0) ** power) * factorial / delta ** (power + 1))
            out.add_term(target_counts, 0, tail * coefficient, tail * gradient)
        return out

    def evaluate(self, t, lambdas):
        t = float(t)
        if not np.isfinite(t) or t < 0:
            raise ValueError("analytic segment time must be finite and non-negative")
        rates = np.asarray(lambdas, float)
        value = slope = 0.0
        gradient = np.zeros(self.nd)
        slope_gradient = np.zeros(self.nd)
        for (counts, power), (coefficient, dcoefficient) in self.terms.items():
            mu = float(np.dot(np.asarray(counts, float), rates))
            if t == 0.0:
                basis = 1.0 if power == 0 else 0.0
                dbasis = 1.0 if power == 1 else (-mu if power == 0 else 0.0)
            else:
                log_basis = power * math.log(t) - mu * t
                basis = 0.0 if log_basis < -745.0 else (
                    math.inf if log_basis > 709.0 else math.exp(log_basis)
                )
                dbasis = (
                    basis * (power / t - mu)
                    if np.isfinite(basis)
                    else math.copysign(math.inf, power / t - mu)
                )
            value += coefficient * basis
            slope += coefficient * dbasis
            gradient += dcoefficient * basis
            slope_gradient += dcoefficient * dbasis
        return value, slope, gradient, slope_gradient


def _weighted_sum(signals, weights, parameter_indices, nd):
    out = _ExpPoly.zero(signals[0].n_modes, nd)
    for i, (signal, weight) in enumerate(zip(signals, np.asarray(weights).reshape(-1))):
        index = None if parameter_indices is None else int(parameter_indices[i])
        out.add_scaled(signal, weight, index)
    return out


def _gated(signal, value, parameter_index, nd):
    out = _ExpPoly.zero(signal.n_modes, nd)
    out.add_scaled(signal, value, parameter_index)
    return out


class FixedAnalyticResponseNetwork:
    """Finite-horizon analytic flow surrogate with continuously gated fixed capacity."""

    kind = "fixed_analytic_response_network"
    format_version = 4

    def __init__(
        self,
        lambdas,
        operating_names,
        *,
        max_response_time,
        input_center=None,
        input_scale=None,
        depth=5,
        channels_per_mode=2,
        quadratic_rank=8,
        cross_rank=4,
        state_rank=3,
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
        self.quadratic_rank = int(quadratic_rank)
        self.cross_rank = int(cross_rank)
        self.state_rank = int(state_rank)
        if min(self.depth, self.channels_per_mode, self.quadratic_rank, self.cross_rank, self.state_rank) < 1:
            raise ValueError("network depths/ranks must be positive")
        self.width = self.n_modes * self.channels_per_mode
        self.input_dimension = self.n_modes + len(self.operating_names)
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
        allocate("input_linear", (self.width, self.input_dimension))
        allocate("quadratic_u", (self.quadratic_rank, self.input_dimension))
        allocate("quadratic_v", (self.quadratic_rank, self.input_dimension))
        allocate("quadratic_out", (self.width, self.quadratic_rank))
        for layer in range(1, self.depth):
            allocate(f"hidden_linear_{layer}", (self.width, self.width))
            allocate(f"cross_input_{layer}", (self.cross_rank, self.input_dimension))
            allocate(f"cross_hidden_{layer}", (self.cross_rank, self.width))
            allocate(f"cross_out_{layer}", (self.width, self.cross_rank))
            allocate(f"state_u_{layer}", (self.state_rank, self.width))
            allocate(f"state_v_{layer}", (self.state_rank, self.width))
            allocate(f"state_out_{layer}", (self.width, self.state_rank))
        allocate("channel_gate", (self.depth, self.width))
        allocate("quadratic_gate", (self.quadratic_rank,))
        for layer in range(1, self.depth):
            allocate(f"cross_gate_{layer}", (self.cross_rank,))
            allocate(f"state_gate_{layer}", (self.state_rank,))
        self.parameter_count = offset
        if parameters is None:
            rng = np.random.default_rng(seed)
            theta = np.zeros(offset)
            sl, shape = self._slices["input_linear"]
            theta[sl] = rng.normal(
                scale=1e-4 * float(np.min(self.lambdas)), size=shape
            ).reshape(-1)
            for name in ("quadratic_u", "quadratic_v"):
                sl, shape = self._slices[name]
                theta[sl] = rng.normal(
                    scale=1 / math.sqrt(max(1, self.input_dimension)), size=shape
                ).reshape(-1)
            for layer in range(1, self.depth):
                for name, dimension in (
                    (f"cross_input_{layer}", self.input_dimension),
                    (f"cross_hidden_{layer}", self.width),
                    (f"state_u_{layer}", self.width),
                    (f"state_v_{layer}", self.width),
                ):
                    sl, shape = self._slices[name]
                    theta[sl] = rng.normal(
                        scale=1 / math.sqrt(max(1, dimension)), size=shape
                    ).reshape(-1)
            theta[self._slices["channel_gate"][0]] = 1.0
            theta[self._slices["quadratic_gate"][0]] = 1.0
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
            _LogicalNode(
                f"fixed_l{layer}_c{channel}",
                int(self.targets[channel]),
            )
            for layer in range(self.depth)
            for channel in range(self.width)
        )

    @property
    def n_modes(self):
        return int(self.lambdas.size)

    @property
    def response_nodes(self):
        gates = self._array("channel_gate")
        return tuple(
            node
            for i, node in enumerate(self._logical_nodes)
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
            self.lambdas,
            self.operating_names,
            max_response_time=self.max_response_time,
            input_center=self.input_center,
            input_scale=self.input_scale,
            depth=self.depth,
            channels_per_mode=self.channels_per_mode,
            quadratic_rank=self.quadratic_rank,
            cross_rank=self.cross_rank,
            state_rank=self.state_rank,
            parameters=parameters,
        )

    def structure_gate_entries(self):
        entries = []
        for layer in range(self.depth):
            values = self._array("channel_gate")[layer]
            ids = self._indices("channel_gate")[layer]
            for c in range(self.width):
                entries.append(
                    {
                        "kind": "channel",
                        "layer": layer,
                        "index": c,
                        "target_mode": int(self.targets[c]),
                        "parameter_index": int(ids[c]),
                        "value": float(values[c]),
                    }
                )
        spec = [("quadratic", 0, self.quadratic_rank)] + [
            (kind, layer, rank)
            for layer in range(1, self.depth)
            for kind, rank in (("cross", self.cross_rank), ("state", self.state_rank))
        ]
        for kind, layer, rank in spec:
            name = "quadratic_gate" if kind == "quadratic" else f"{kind}_gate_{layer}"
            values, ids = self._array(name), self._indices(name)
            for r in range(rank):
                entries.append(
                    {
                        "kind": kind,
                        "layer": layer,
                        "index": r,
                        "parameter_index": int(ids[r]),
                        "value": float(values[r]),
                    }
                )
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
            "active_channels_by_layer": by_layer,
            "active_channels_per_mode": per_mode,
            "quadratic_rank": int(
                np.count_nonzero(np.abs(self._array("quadratic_gate")) > threshold)
            ),
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

    def _base_signals(self, a0, operating, derivative_kind):
        initial = np.asarray(a0, float).reshape(-1)
        u = np.asarray(operating, float).reshape(-1)
        if (
            initial.shape != (self.n_modes,)
            or u.shape != (len(self.operating_names),)
            or np.any(~np.isfinite(initial))
            or np.any(~np.isfinite(u))
        ):
            raise ValueError("invalid initial/operating input")
        normalized = (np.concatenate([initial, u]) - self.input_center) / self.input_scale
        if derivative_kind == "parameter":
            nd = self.parameter_count
        elif derivative_kind == "initial":
            nd = self.n_modes
        elif derivative_kind == "operating":
            nd = len(self.operating_names)
        elif derivative_kind is None:
            nd = 0
        else:
            raise ValueError("unknown derivative kind")
        signals = []
        for mode in range(self.n_modes):
            d = np.zeros(nd)
            if derivative_kind == "initial":
                d[mode] = 1.0 / self.input_scale[mode]
            signals.append(
                _ExpPoly.term(
                    tuple(1 if i == mode else 0 for i in range(self.n_modes)),
                    normalized[mode],
                    nd=nd,
                    derivative=d,
                )
            )
        zero = (0,) * self.n_modes
        for i in range(len(self.operating_names)):
            d = np.zeros(nd)
            if derivative_kind == "operating":
                d[i] = 1.0 / self.input_scale[self.n_modes + i]
            signals.append(
                _ExpPoly.term(
                    zero,
                    normalized[self.n_modes + i],
                    nd=nd,
                    derivative=d,
                )
            )
        return initial, signals, nd

    def _projection(self, signals, name, row, derivative_kind, nd):
        return _weighted_sum(
            signals,
            self._array(name)[row],
            self._indices(name)[row] if derivative_kind == "parameter" else None,
            nd,
        )

    def _bias(self, layer, channel, derivative_kind, nd):
        zero = (0,) * self.n_modes
        d = np.zeros(nd)
        if derivative_kind == "parameter":
            d[int(self._indices(f"bias_{layer}")[channel])] = 1.0
        return _ExpPoly.term(
            zero,
            self._array(f"bias_{layer}")[channel],
            nd=nd,
            derivative=d,
        )

    def _forward_signals(self, a0, operating, derivative_kind=None):
        initial, base, nd = self._base_signals(a0, operating, derivative_kind)
        pm = derivative_kind == "parameter"
        channel_gates = self._array("channel_gate")
        channel_ids = self._indices("channel_gate") if pm else None
        qu = [
            self._projection(base, "quadratic_u", r, derivative_kind, nd)
            for r in range(self.quadratic_rank)
        ]
        qv = [
            self._projection(base, "quadratic_v", r, derivative_kind, nd)
            for r in range(self.quadratic_rank)
        ]
        qg = self._array("quadratic_gate")
        qids = self._indices("quadratic_gate") if pm else None
        quadratic = [
            _gated(
                a.product(b),
                qg[r],
                None if qids is None else qids[r],
                nd,
            )
            for r, (a, b) in enumerate(zip(qu, qv))
        ]
        linear = self._array("input_linear")
        linear_ids = self._indices("input_linear") if pm else None
        qout = self._array("quadratic_out")
        qout_ids = self._indices("quadratic_out") if pm else None
        previous = []
        layers = []
        for c in range(self.width):
            source = self._bias(0, c, derivative_kind, nd)
            source.add_scaled(
                _weighted_sum(
                    base,
                    linear[c],
                    None if linear_ids is None else linear_ids[c],
                    nd,
                ),
                1.0,
            )
            source.add_scaled(
                _weighted_sum(
                    quadratic,
                    qout[c],
                    None if qout_ids is None else qout_ids[c],
                    nd,
                ),
                1.0,
            )
            response = source.response(self.targets[c], self.lambdas)
            previous.append(
                _gated(
                    response,
                    channel_gates[0, c],
                    None if channel_ids is None else channel_ids[0, c],
                    nd,
                )
            )
        layers.append(tuple(previous))
        for layer in range(1, self.depth):
            h = self._array(f"hidden_linear_{layer}")
            hids = self._indices(f"hidden_linear_{layer}") if pm else None
            ci = [
                self._projection(base, f"cross_input_{layer}", r, derivative_kind, nd)
                for r in range(self.cross_rank)
            ]
            ch = [
                self._projection(previous, f"cross_hidden_{layer}", r, derivative_kind, nd)
                for r in range(self.cross_rank)
            ]
            cg = self._array(f"cross_gate_{layer}")
            cgids = self._indices(f"cross_gate_{layer}") if pm else None
            cross = [
                _gated(
                    a.product(b),
                    cg[r],
                    None if cgids is None else cgids[r],
                    nd,
                )
                for r, (a, b) in enumerate(zip(ci, ch))
            ]
            co = self._array(f"cross_out_{layer}")
            coids = self._indices(f"cross_out_{layer}") if pm else None
            su = [
                self._projection(previous, f"state_u_{layer}", r, derivative_kind, nd)
                for r in range(self.state_rank)
            ]
            sv = [
                self._projection(previous, f"state_v_{layer}", r, derivative_kind, nd)
                for r in range(self.state_rank)
            ]
            sg = self._array(f"state_gate_{layer}")
            sgids = self._indices(f"state_gate_{layer}") if pm else None
            state = [
                _gated(
                    a.product(b),
                    sg[r],
                    None if sgids is None else sgids[r],
                    nd,
                )
                for r, (a, b) in enumerate(zip(su, sv))
            ]
            so = self._array(f"state_out_{layer}")
            soids = self._indices(f"state_out_{layer}") if pm else None
            current = []
            for c in range(self.width):
                source = self._bias(layer, c, derivative_kind, nd)
                source.add_scaled(
                    _weighted_sum(
                        previous,
                        h[c],
                        None if hids is None else hids[c],
                        nd,
                    ),
                    1.0,
                )
                source.add_scaled(
                    _weighted_sum(
                        cross,
                        co[c],
                        None if coids is None else coids[c],
                        nd,
                    ),
                    1.0,
                )
                source.add_scaled(
                    _weighted_sum(
                        state,
                        so[c],
                        None if soids is None else soids[c],
                        nd,
                    ),
                    1.0,
                )
                response = source.response(self.targets[c], self.lambdas)
                current.append(
                    _gated(
                        response,
                        channel_gates[layer, c],
                        None if channel_ids is None else channel_ids[layer, c],
                        nd,
                    )
                )
            previous = current
            layers.append(tuple(current))
        return initial, tuple(layers), nd

    def evaluate_with_jacobians(self, t, *, a0, operating, derivative_kind):
        t = float(t)
        if not np.isfinite(t) or t < 0 or t > self.max_response_time:
            raise ValueError(
                f"segment time must be in [0, {self.max_response_time:g}]"
            )
        initial, layers, nd = self._forward_signals(a0, operating, derivative_kind)
        a = np.zeros(self.n_modes)
        da = np.zeros(self.n_modes)
        ja = np.zeros((self.n_modes, nd))
        jda = np.zeros_like(ja)
        for mode in range(self.n_modes):
            decay = math.exp(-self.lambdas[mode] * t)
            a[mode] = initial[mode] * decay
            da[mode] = -self.lambdas[mode] * a[mode]
            if derivative_kind == "initial":
                ja[mode, mode] += decay
                jda[mode, mode] -= self.lambdas[mode] * decay
        for layer in layers:
            for c, signal in enumerate(layer):
                value, slope, g, sg = signal.evaluate(t, self.lambdas)
                target = self.targets[c]
                a[target] += value
                da[target] += slope
                ja[target] += g
                jda[target] += sg
        return a, da, ja, jda

    def evaluate(self, t, *, a0, operating):
        return self.evaluate_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind=None
        )[:2]

    def evaluate_parameter_jacobian(self, t, *, a0, operating):
        return self.evaluate_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind="parameter"
        )

    def evaluate_initial_jacobian(self, t, *, a0, operating):
        return self.evaluate_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind="initial"
        )

    def evaluate_operating_jacobian(self, t, *, a0, operating):
        return self.evaluate_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind="operating"
        )

    def to_metadata(self):
        return {
            "kind": self.kind,
            "format_version": self.format_version,
            "lambdas": self.lambdas.tolist(),
            "operating_names": list(self.operating_names),
            "max_response_time": self.max_response_time,
            "input_center": self.input_center.tolist(),
            "input_scale": self.input_scale.tolist(),
            "depth": self.depth,
            "channels_per_mode": self.channels_per_mode,
            "quadratic_rank": self.quadratic_rank,
            "cross_rank": self.cross_rank,
            "state_rank": self.state_rank,
            "structure": self.structure_summary(0.0),
        }

    @classmethod
    def from_metadata(cls, metadata, parameters):
        if (
            metadata.get("kind") != cls.kind
            or metadata.get("format_version") != cls.format_version
        ):
            raise ValueError("unsupported fixed analytic response network format")
        return cls(
            metadata["lambdas"],
            metadata["operating_names"],
            max_response_time=metadata["max_response_time"],
            input_center=metadata["input_center"],
            input_scale=metadata["input_scale"],
            depth=metadata["depth"],
            channels_per_mode=metadata["channels_per_mode"],
            quadratic_rank=metadata["quadratic_rank"],
            cross_rank=metadata["cross_rank"],
            state_rank=metadata["state_rank"],
            parameters=parameters,
        )
