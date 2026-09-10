from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class _LogicalNode:
    name: str
    target_mode: int


class _ExpPoly:
    """Exact exponential-polynomial signal with optional dense sensitivities.

    A term is c * t**k * exp(-<counts, lambdas>*t). Sums, products and the
    first-order thermal response operator remain closed in this representation.
    """

    def __init__(self, n_modes: int, nd: int = 0):
        self.n_modes = int(n_modes)
        self.nd = int(nd)
        self.terms: dict[tuple[tuple[int, ...], int], tuple[float, np.ndarray]] = {}

    @classmethod
    def term(cls, counts, coefficient: float, *, nd=0, derivative=None):
        counts = tuple(int(v) for v in counts)
        out = cls(len(counts), nd)
        out.add_term(counts, 0, coefficient, derivative)
        return out

    @classmethod
    def zero(cls, n_modes, nd=0):
        return cls(n_modes, nd)

    def copy(self):
        out = _ExpPoly(self.n_modes, self.nd)
        out.terms = {key: (float(c), g.copy()) for key, (c, g) in self.terms.items()}
        return out

    def add_term(self, counts, power, coefficient, derivative=None):
        counts = tuple(int(v) for v in counts)
        power = int(power)
        coefficient = float(coefficient)
        if len(counts) != self.n_modes or power < 0:
            raise ValueError("invalid exponential-polynomial term")
        if derivative is None:
            gradient = np.zeros(self.nd, dtype=float)
        else:
            gradient = np.asarray(derivative, dtype=float)
            if gradient.shape != (self.nd,):
                raise ValueError("term derivative dimension mismatch")
        key = (counts, power)
        if key in self.terms:
            old_c, old_g = self.terms[key]
            coefficient += old_c
            gradient = gradient + old_g
        if coefficient == 0.0 and not np.any(gradient):
            self.terms.pop(key, None)
        else:
            self.terms[key] = (coefficient, gradient)
        return self

    def add_scaled(self, other: "_ExpPoly", scale: float, parameter_index=None):
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

    def product(self, other: "_ExpPoly"):
        if self.n_modes != other.n_modes or self.nd != other.nd:
            raise ValueError("signal dimensions do not match")
        out = _ExpPoly(self.n_modes, self.nd)
        for (ca, pa), (va, ga) in self.terms.items():
            for (cb, pb), (vb, gb) in other.terms.items():
                counts = tuple(a + b for a, b in zip(ca, cb))
                out.add_term(counts, pa + pb, va * vb, ga * vb + gb * va)
        return out

    def response(self, target: int, lambdas):
        """Exact zero-initial solution of y' + lambda_target*y = self."""
        target = int(target)
        rates = np.asarray(lambdas, dtype=float)
        if rates.shape != (self.n_modes,):
            raise ValueError("thermal spectrum dimension mismatch")
        lam = float(rates[target])
        if lam <= 0.0:
            raise ValueError("fixed analytic network requires positive thermal decay rates")
        target_counts = tuple(1 if i == target else 0 for i in range(self.n_modes))
        out = _ExpPoly(self.n_modes, self.nd)
        for (counts, power), (coefficient, gradient) in self.terms.items():
            mu = float(np.dot(np.asarray(counts, dtype=float), rates))
            if counts == target_counts or lam == mu:
                factor = 1.0 / float(power + 1)
                out.add_term(counts, power + 1, factor * coefficient, factor * gradient)
                continue
            delta = lam - mu
            factorial = math.factorial(power)
            for m in range(power + 1):
                p = power - m
                factor = ((-1.0) ** m) * factorial / math.factorial(p) / (delta ** (m + 1))
                out.add_term(counts, p, factor * coefficient, factor * gradient)
            tail = -(((-1.0) ** power) * factorial / (delta ** (power + 1)))
            out.add_term(target_counts, 0, tail * coefficient, tail * gradient)
        return out

    def evaluate(self, t: float, lambdas):
        t = float(t)
        if np.isnan(t) or t < 0.0:
            raise ValueError("time must be non-negative or positive infinity")
        rates = np.asarray(lambdas, dtype=float)
        value = 0.0
        slope = 0.0
        gradient = np.zeros(self.nd, dtype=float)
        slope_gradient = np.zeros(self.nd, dtype=float)
        for (counts, power), (coefficient, dcoefficient) in self.terms.items():
            mu = float(np.dot(np.asarray(counts, dtype=float), rates))
            if np.isposinf(t):
                basis = 1.0 if mu == 0.0 and power == 0 else 0.0
                dbasis = 0.0
            elif t == 0.0:
                basis = 1.0 if power == 0 else 0.0
                dbasis = 1.0 if power == 1 else (-mu if power == 0 else 0.0)
            else:
                log_basis = power * math.log(t) - mu * t
                if log_basis < -745.0:
                    basis = 0.0
                elif log_basis > 709.0:
                    basis = math.inf
                else:
                    basis = math.exp(log_basis)
                dbasis = (
                    basis * (power / t - mu)
                    if np.isfinite(basis)
                    else math.copysign(math.inf, power / t - mu)
                )
            value += coefficient * basis
            slope += coefficient * dbasis
            if self.nd:
                gradient += dcoefficient * basis
                slope_gradient += dcoefficient * dbasis
        return value, slope, gradient, slope_gradient


def _weighted_sum(signals, weights, parameter_indices, nd):
    if len(signals) != len(weights):
        raise ValueError("projection dimensions do not match")
    out = _ExpPoly.zero(signals[0].n_modes if signals else 0, nd)
    for index, (signal, weight) in enumerate(zip(signals, np.asarray(weights).reshape(-1))):
        parameter_index = None if parameter_indices is None else int(parameter_indices[index])
        out.add_scaled(signal, float(weight), parameter_index)
    return out


def _gated(signal, value, parameter_index, nd):
    out = _ExpPoly.zero(signal.n_modes, nd)
    out.add_scaled(signal, float(value), parameter_index)
    return out


class FixedAnalyticResponseNetwork:
    """Fixed maximum analytic network with continuously trainable structure gates.

    Every hidden channel obeys h' + lambda_j*h = source. The maximum topology is
    created once; no Grow/Split/candidate enumeration occurs during training.
    Channel and low-rank component gates are ordinary continuous parameters.
    Residual-driven training can drive them to zero and validated pruning then
    removes inactive capacity without changing the analytic time representation.
    """

    kind = "fixed_analytic_response_network"
    format_version = 2

    def __init__(
        self,
        lambdas,
        operating_names,
        *,
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
        self.lambdas = np.asarray(lambdas, dtype=float).reshape(-1)
        if (
            self.lambdas.size == 0
            or np.any(~np.isfinite(self.lambdas))
            or np.any(self.lambdas <= 0.0)
        ):
            raise ValueError("thermal decay rates must be finite and positive")
        self.operating_names = tuple(str(value) for value in operating_names)
        self.initial_names = tuple(f"a0_{i}" for i in range(self.lambdas.size))
        self.depth = int(depth)
        self.channels_per_mode = int(channels_per_mode)
        self.quadratic_rank = int(quadratic_rank)
        self.cross_rank = int(cross_rank)
        self.state_rank = int(state_rank)
        if min(
            self.depth,
            self.channels_per_mode,
            self.quadratic_rank,
            self.cross_rank,
            self.state_rank,
        ) < 1:
            raise ValueError("fixed analytic network depths/ranks must be positive")
        self.width = self.n_modes * self.channels_per_mode
        self.input_dimension = 1 + self.n_modes + len(self.operating_names)
        physical_dimension = self.n_modes + len(self.operating_names)
        if input_center is None:
            input_center = np.zeros(physical_dimension, dtype=float)
        if input_scale is None:
            input_scale = np.ones(physical_dimension, dtype=float)
        self.input_center = np.asarray(input_center, dtype=float).reshape(-1)
        self.input_scale = np.asarray(input_scale, dtype=float).reshape(-1)
        if (
            self.input_center.shape != (physical_dimension,)
            or self.input_scale.shape != (physical_dimension,)
        ):
            raise ValueError("input normalization dimension mismatch")
        if (
            np.any(~np.isfinite(self.input_center + self.input_scale))
            or np.any(self.input_scale <= 0.0)
        ):
            raise ValueError("input normalization must be finite with positive scale")

        self.targets = np.repeat(
            np.arange(self.n_modes, dtype=int), self.channels_per_mode
        )
        self._slices = {}
        offset = 0

        def allocate(name, shape):
            nonlocal offset
            size = int(np.prod(shape))
            self._slices[name] = (
                slice(offset, offset + size),
                tuple(int(v) for v in shape),
            )
            offset += size

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
        self.legacy_parameter_count = offset

        for layer in range(1, self.depth):
            allocate(f"hidden_bias_{layer}", (self.width,))
        allocate("channel_gate", (self.depth, self.width))
        allocate("quadratic_gate", (self.quadratic_rank,))
        for layer in range(1, self.depth):
            allocate(f"cross_gate_{layer}", (self.cross_rank,))
            allocate(f"state_gate_{layer}", (self.state_rank,))
        self.parameter_count = offset

        if parameters is None:
            rng = np.random.default_rng(int(seed))
            theta = np.zeros(self.parameter_count, dtype=float)
            sl, shape = self._slices["input_linear"]
            initial_source_scale = 1.0e-4 * float(np.min(self.lambdas))
            theta[sl] = rng.normal(scale=initial_source_scale, size=shape).reshape(-1)
            for name in ("quadratic_u", "quadratic_v"):
                sl, shape = self._slices[name]
                theta[sl] = rng.normal(scale=1.0 / math.sqrt(self.input_dimension), size=shape).reshape(-1)
            for layer in range(1, self.depth):
                for name, dimension in (
                    (f"cross_input_{layer}", self.input_dimension),
                    (f"cross_hidden_{layer}", self.width),
                    (f"state_u_{layer}", self.width),
                    (f"state_v_{layer}", self.width),
                ):
                    sl, shape = self._slices[name]
                    theta[sl] = rng.normal(scale=1.0 / math.sqrt(dimension), size=shape).reshape(-1)
            theta[self._slices["channel_gate"][0]] = 1.0
            theta[self._slices["quadratic_gate"][0]] = 1.0
            for layer in range(1, self.depth):
                theta[self._slices[f"cross_gate_{layer}"][0]] = 1.0
                theta[self._slices[f"state_gate_{layer}"][0]] = 1.0
            self.parameters = theta
        else:
            values = np.asarray(parameters, dtype=float).reshape(-1)
            if values.shape != (self.parameter_count,) or np.any(~np.isfinite(values)):
                raise ValueError("network parameter vector has wrong shape or non-finite values")
            self.parameters = values.copy()

        self._logical_nodes = tuple(
            _LogicalNode(f"fixed_l{layer}_c{channel}", int(self.targets[channel]))
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
            for index, node in enumerate(self._logical_nodes)
            if gates[index // self.width, index % self.width] != 0.0
        )

    def _array(self, name):
        sl, shape = self._slices[name]
        return self.parameters[sl].reshape(shape)

    def _indices(self, name):
        sl, shape = self._slices[name]
        return np.arange(sl.start, sl.stop, dtype=int).reshape(shape)

    def with_parameters(self, parameters):
        return FixedAnalyticResponseNetwork(
            self.lambdas,
            self.operating_names,
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
        channel = self._array("channel_gate")
        channel_ids = self._indices("channel_gate")
        for layer in range(self.depth):
            for c in range(self.width):
                entries.append({
                    "kind": "channel", "layer": layer, "index": c,
                    "target_mode": int(self.targets[c]),
                    "parameter_index": int(channel_ids[layer, c]),
                    "value": float(channel[layer, c]),
                })
        q = self._array("quadratic_gate")
        qids = self._indices("quadratic_gate")
        for r in range(self.quadratic_rank):
            entries.append({
                "kind": "quadratic", "layer": 0, "index": r,
                "parameter_index": int(qids[r]), "value": float(q[r]),
            })
        for layer in range(1, self.depth):
            for kind, rank in (("cross", self.cross_rank), ("state", self.state_rank)):
                name = f"{kind}_gate_{layer}"
                values = self._array(name)
                ids = self._indices(name)
                for r in range(rank):
                    entries.append({
                        "kind": kind, "layer": layer, "index": r,
                        "parameter_index": int(ids[r]), "value": float(values[r]),
                    })
        return tuple(entries)

    def structure_summary(self, threshold=0.0):
        threshold = float(threshold)
        channel = np.abs(self._array("channel_gate")) > threshold
        channels_by_layer = [int(np.count_nonzero(row)) for row in channel]
        active_per_mode = []
        for mode in range(self.n_modes):
            mask = self.targets == mode
            active_per_mode.append(int(np.count_nonzero(channel[:, mask])))
        active_layers = [i for i, count in enumerate(channels_by_layer) if count]
        cross = []
        state = []
        for layer in range(1, self.depth):
            cross.append(int(np.count_nonzero(np.abs(self._array(f"cross_gate_{layer}")) > threshold)))
            state.append(int(np.count_nonzero(np.abs(self._array(f"state_gate_{layer}")) > threshold)))
        return {
            "maximum_depth": self.depth,
            "effective_depth": 0 if not active_layers else max(active_layers) + 1,
            "maximum_channels_per_mode": self.channels_per_mode,
            "active_channels_by_layer": channels_by_layer,
            "active_channels_per_mode": active_per_mode,
            "quadratic_rank": int(np.count_nonzero(np.abs(self._array("quadratic_gate")) > threshold)),
            "cross_rank_by_layer": cross,
            "state_rank_by_layer": state,
            "parameter_count": self.parameter_count,
            "active_response_channels": int(sum(channels_by_layer)),
        }

    def soft_threshold_structure(self, amount):
        amount = float(amount)
        if not np.isfinite(amount) or amount < 0.0:
            raise ValueError("structure shrink amount must be finite and non-negative")
        if amount == 0.0:
            return self
        theta = self.parameters.copy()
        for entry in self.structure_gate_entries():
            i = entry["parameter_index"]
            value = theta[i]
            theta[i] = math.copysign(max(0.0, abs(value) - amount), value)
        return self.with_parameters(theta)

    def prune_structure(self, threshold):
        threshold = float(threshold)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("pruning threshold must be finite and non-negative")
        theta = self.parameters.copy()
        for entry in self.structure_gate_entries():
            if abs(entry["value"]) <= threshold:
                theta[entry["parameter_index"]] = 0.0
        return self.with_parameters(theta)

    def _base_signals(self, a0, operating, derivative_kind):
        initial = np.asarray(a0, dtype=float).reshape(-1)
        u = np.asarray(operating, dtype=float).reshape(-1)
        if initial.shape != (self.n_modes,) or u.shape != (len(self.operating_names),):
            raise ValueError("initial/operating dimension mismatch")
        if np.any(~np.isfinite(initial)) or np.any(~np.isfinite(u)):
            raise ValueError("initial and operating inputs must be finite")
        normalized = (np.concatenate([initial, u]) - self.input_center) / self.input_scale
        if derivative_kind == "parameter":
            nd = self.parameter_count
        elif derivative_kind == "operating":
            nd = len(self.operating_names)
        elif derivative_kind is None:
            nd = 0
        else:
            raise ValueError("unknown derivative kind")
        zero_counts = (0,) * self.n_modes
        signals = [_ExpPoly.term(zero_counts, 1.0, nd=nd)]
        for mode in range(self.n_modes):
            signals.append(_ExpPoly.term(
                tuple(1 if i == mode else 0 for i in range(self.n_modes)),
                normalized[mode], nd=nd))
        for index in range(len(self.operating_names)):
            derivative = np.zeros(nd, dtype=float)
            if derivative_kind == "operating":
                derivative[index] = 1.0 / self.input_scale[self.n_modes + index]
            signals.append(_ExpPoly.term(
                zero_counts, normalized[self.n_modes + index], nd=nd,
                derivative=derivative))
        return initial, signals, nd

    def _projection(self, signals, matrix_name, row, derivative_kind, nd):
        values = self._array(matrix_name)[row]
        indices = self._indices(matrix_name)[row] if derivative_kind == "parameter" else None
        return _weighted_sum(signals, values, indices, nd)

    def _bias_signal(self, layer, channel, derivative_kind, nd):
        zero_counts = (0,) * self.n_modes
        value = float(self._array(f"hidden_bias_{layer}")[channel])
        derivative = np.zeros(nd, dtype=float)
        if derivative_kind == "parameter":
            derivative[int(self._indices(f"hidden_bias_{layer}")[channel])] = 1.0
        return _ExpPoly.term(zero_counts, value, nd=nd, derivative=derivative)

    def _forward_signals(self, a0, operating, derivative_kind=None):
        initial, base, nd = self._base_signals(a0, operating, derivative_kind)
        parameter_mode = derivative_kind == "parameter"
        channel_gates = self._array("channel_gate")
        channel_gate_ids = self._indices("channel_gate") if parameter_mode else None
        linear = self._array("input_linear")
        linear_ids = self._indices("input_linear") if parameter_mode else None
        qout = self._array("quadratic_out")
        qout_ids = self._indices("quadratic_out") if parameter_mode else None
        qgates = self._array("quadratic_gate")
        qgate_ids = self._indices("quadratic_gate") if parameter_mode else None

        qu = [self._projection(base, "quadratic_u", r, derivative_kind, nd)
              for r in range(self.quadratic_rank)]
        qv = [self._projection(base, "quadratic_v", r, derivative_kind, nd)
              for r in range(self.quadratic_rank)]
        quadratic = []
        for r, (left, right) in enumerate(zip(qu, qv)):
            quadratic.append(_gated(
                left.product(right), qgates[r],
                None if qgate_ids is None else int(qgate_ids[r]), nd))

        previous = []
        layers = []
        for channel in range(self.width):
            source = _weighted_sum(
                base, linear[channel],
                None if linear_ids is None else linear_ids[channel], nd)
            source.add_scaled(_weighted_sum(
                quadratic, qout[channel],
                None if qout_ids is None else qout_ids[channel], nd), 1.0)
            response = source.response(int(self.targets[channel]), self.lambdas)
            previous.append(_gated(
                response, channel_gates[0, channel],
                None if channel_gate_ids is None else int(channel_gate_ids[0, channel]), nd))
        layers.append(tuple(previous))

        for layer in range(1, self.depth):
            hlin = self._array(f"hidden_linear_{layer}")
            hlin_ids = self._indices(f"hidden_linear_{layer}") if parameter_mode else None
            cross_input = [
                self._projection(base, f"cross_input_{layer}", r, derivative_kind, nd)
                for r in range(self.cross_rank)]
            cross_hidden = [
                self._projection(previous, f"cross_hidden_{layer}", r, derivative_kind, nd)
                for r in range(self.cross_rank)]
            cross_gate = self._array(f"cross_gate_{layer}")
            cross_gate_ids = self._indices(f"cross_gate_{layer}") if parameter_mode else None
            cross = [
                _gated(
                    left.product(right), cross_gate[r],
                    None if cross_gate_ids is None else int(cross_gate_ids[r]), nd)
                for r, (left, right) in enumerate(zip(cross_input, cross_hidden))]
            cout = self._array(f"cross_out_{layer}")
            cout_ids = self._indices(f"cross_out_{layer}") if parameter_mode else None

            state_u = [
                self._projection(previous, f"state_u_{layer}", r, derivative_kind, nd)
                for r in range(self.state_rank)]
            state_v = [
                self._projection(previous, f"state_v_{layer}", r, derivative_kind, nd)
                for r in range(self.state_rank)]
            state_gate = self._array(f"state_gate_{layer}")
            state_gate_ids = self._indices(f"state_gate_{layer}") if parameter_mode else None
            state_products = [
                _gated(
                    left.product(right), state_gate[r],
                    None if state_gate_ids is None else int(state_gate_ids[r]), nd)
                for r, (left, right) in enumerate(zip(state_u, state_v))]
            sout = self._array(f"state_out_{layer}")
            sout_ids = self._indices(f"state_out_{layer}") if parameter_mode else None

            current = []
            for channel in range(self.width):
                source = self._bias_signal(layer, channel, derivative_kind, nd)
                source.add_scaled(_weighted_sum(
                    previous, hlin[channel],
                    None if hlin_ids is None else hlin_ids[channel], nd), 1.0)
                source.add_scaled(_weighted_sum(
                    cross, cout[channel],
                    None if cout_ids is None else cout_ids[channel], nd), 1.0)
                source.add_scaled(_weighted_sum(
                    state_products, sout[channel],
                    None if sout_ids is None else sout_ids[channel], nd), 1.0)
                response = source.response(int(self.targets[channel]), self.lambdas)
                current.append(_gated(
                    response, channel_gates[layer, channel],
                    None if channel_gate_ids is None else int(channel_gate_ids[layer, channel]), nd))
            previous = current
            layers.append(tuple(current))
        return initial, tuple(layers), nd

    def evaluate_stable_with_jacobians(self, t, *, a0, operating, derivative_kind="operating"):
        initial, layers, nd = self._forward_signals(a0, operating, derivative_kind)
        t = float(t)
        a = np.zeros(self.n_modes, dtype=float)
        da = np.zeros(self.n_modes, dtype=float)
        ja = np.zeros((self.n_modes, nd), dtype=float)
        jda = np.zeros_like(ja)
        for mode in range(self.n_modes):
            decay = 0.0 if np.isposinf(t) else math.exp(-float(self.lambdas[mode]) * t)
            a[mode] = initial[mode] * decay
            da[mode] = 0.0 if np.isposinf(t) else -self.lambdas[mode] * a[mode]
        for layer in layers:
            for channel, signal in enumerate(layer):
                value, slope, gradient, slope_gradient = signal.evaluate(t, self.lambdas)
                target = int(self.targets[channel])
                a[target] += value
                da[target] += slope
                if nd:
                    ja[target] += gradient
                    jda[target] += slope_gradient
        return a, da, ja, jda

    def evaluate(self, t, *, a0, operating):
        a, da, _, _ = self.evaluate_stable_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind=None)
        return a, da

    def evaluate_parameter_jacobian(self, t, *, a0, operating):
        return self.evaluate_stable_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind="parameter")

    def evaluate_operating_jacobian(self, t, *, a0, operating):
        return self.evaluate_stable_with_jacobians(
            t, a0=a0, operating=operating, derivative_kind="operating")

    def to_metadata(self):
        return {
            "kind": self.kind,
            "format_version": self.format_version,
            "lambdas": self.lambdas.tolist(),
            "operating_names": list(self.operating_names),
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
        if metadata.get("kind") != cls.kind:
            raise ValueError("unsupported fixed analytic response network kind")
        version = int(metadata.get("format_version", 1))
        if version not in (1, cls.format_version):
            raise ValueError("unsupported fixed analytic response network format")
        kwargs = dict(
            input_center=metadata["input_center"],
            input_scale=metadata["input_scale"],
            depth=metadata["depth"],
            channels_per_mode=metadata["channels_per_mode"],
            quadratic_rank=metadata["quadratic_rank"],
            cross_rank=metadata["cross_rank"],
            state_rank=metadata["state_rank"],
        )
        values = np.asarray(parameters, dtype=float).reshape(-1)
        if version == cls.format_version:
            return cls(
                metadata["lambdas"], metadata["operating_names"],
                parameters=values, **kwargs)

        upgraded = cls(metadata["lambdas"], metadata["operating_names"], **kwargs)
        if values.shape != (upgraded.legacy_parameter_count,):
            raise ValueError("legacy fixed-network parameter vector has wrong shape")
        theta = upgraded.parameters.copy()
        theta[: upgraded.legacy_parameter_count] = values
        return upgraded.with_parameters(theta)
