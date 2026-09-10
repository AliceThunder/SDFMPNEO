"""Generic symbolic evaluator for deeper fixed analytic response networks."""
from __future__ import annotations

import math
import numpy as np

from .fixed_response_basis import _ExpPoly, _weighted_sum, _gated


class _GenericAnalyticMixin:
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
        if derivative_kind == "parameter": nd = self.parameter_count
        elif derivative_kind == "initial": nd = self.n_modes
        elif derivative_kind == "operating": nd = len(self.operating_names)
        elif derivative_kind is None: nd = 0
        else: raise ValueError("unknown derivative kind")
        signals = []
        for mode in range(self.n_modes):
            d = np.zeros(nd)
            if derivative_kind == "initial":
                d[mode] = 1.0 / self.input_scale[mode]
            signals.append(_ExpPoly.modal_term(
                self.n_modes, mode, normalized[mode], nd=nd, derivative=d
            ))
        static = [_ExpPoly.constant_term(self.n_modes, 1.0, nd=nd)]
        for i in range(len(self.operating_names)):
            d = np.zeros(nd)
            if derivative_kind == "operating":
                d[i] = 1.0 / self.input_scale[self.n_modes + i]
            signal = _ExpPoly.constant_term(
                self.n_modes, normalized[self.n_modes + i], nd=nd, derivative=d
            )
            signals.append(signal)
            static.append(signal)
        return initial, signals, static, nd

    def _projection(self, signals, name, row, derivative_kind, nd):
        return _weighted_sum(
            signals,
            self._array(name)[row],
            self._indices(name)[row] if derivative_kind == "parameter" else None,
            nd,
        )

    def _bias(self, layer, channel, derivative_kind, nd):
        d = np.zeros(nd)
        if derivative_kind == "parameter":
            d[int(self._indices(f"bias_{layer}")[channel])] = 1.0
        return _ExpPoly.constant_term(
            self.n_modes, self._array(f"bias_{layer}")[channel], nd=nd, derivative=d
        )

    def _forward_signals(self, a0, operating, derivative_kind=None):
        initial, base, static, nd = self._base_signals(a0, operating, derivative_kind)
        pm = derivative_kind == "parameter"
        channel_gates = self._array("channel_gate")
        channel_ids = self._indices("channel_gate") if pm else None

        linear_features = [
            self._projection(base, "input_linear_in", r, derivative_kind, nd)
            for r in range(self.linear_rank)
        ]
        linear_out = self._array("input_linear_out")
        linear_out_ids = self._indices("input_linear_out") if pm else None
        qu = [self._projection(base, "quadratic_u", r, derivative_kind, nd) for r in range(self.quadratic_rank)]
        qv = [self._projection(static, "quadratic_v", r, derivative_kind, nd) for r in range(self.quadratic_rank)]
        qg = self._array("quadratic_gate")
        qids = self._indices("quadratic_gate") if pm else None
        quadratic = [
            _gated(a.product(b), qg[r], None if qids is None else qids[r], nd)
            for r, (a, b) in enumerate(zip(qu, qv))
        ]
        qout = self._array("quadratic_out")
        qout_ids = self._indices("quadratic_out") if pm else None
        thermal_squared = [signal.product(signal) for signal in base[:self.n_modes]]
        square_features = [
            self._projection(thermal_squared, "square_in", r, derivative_kind, nd)
            for r in range(self.square_rank)
        ]
        square_gates = self._array("square_gate")
        square_gate_ids = self._indices("square_gate") if pm else None
        square_features = [
            _gated(feature, square_gates[r], None if square_gate_ids is None else square_gate_ids[r], nd)
            for r, feature in enumerate(square_features)
        ]
        square_out = self._array("square_out")
        square_out_ids = self._indices("square_out") if pm else None

        previous = []
        layers = []
        for c in range(self.width):
            source = self._bias(0, c, derivative_kind, nd)
            source.add_scaled(_weighted_sum(
                linear_features, linear_out[c],
                None if linear_out_ids is None else linear_out_ids[c], nd
            ), 1.0)
            source.add_scaled(_weighted_sum(
                quadratic, qout[c], None if qout_ids is None else qout_ids[c], nd
            ), 1.0)
            source.add_scaled(_weighted_sum(
                square_features, square_out[c],
                None if square_out_ids is None else square_out_ids[c], nd
            ), 1.0)
            response = source.response(self.targets[c], self.lambdas)
            previous.append(_gated(
                response, channel_gates[0, c],
                None if channel_ids is None else channel_ids[0, c], nd
            ))
        layers.append(tuple(previous))

        for layer in range(1, self.depth):
            hidden_features = [
                self._projection(previous, f"hidden_linear_in_{layer}", r, derivative_kind, nd)
                for r in range(self.hidden_rank)
            ]
            hidden_out = self._array(f"hidden_linear_out_{layer}")
            hidden_out_ids = self._indices(f"hidden_linear_out_{layer}") if pm else None
            ci = [self._projection(static, f"cross_input_{layer}", r, derivative_kind, nd) for r in range(self.cross_rank)]
            ch = [self._projection(previous, f"cross_hidden_{layer}", r, derivative_kind, nd) for r in range(self.cross_rank)]
            cg = self._array(f"cross_gate_{layer}")
            cgids = self._indices(f"cross_gate_{layer}") if pm else None
            cross = [
                _gated(a.product(b), cg[r], None if cgids is None else cgids[r], nd)
                for r, (a, b) in enumerate(zip(ci, ch))
            ]
            co = self._array(f"cross_out_{layer}")
            coids = self._indices(f"cross_out_{layer}") if pm else None
            su = [self._projection(previous, f"state_u_{layer}", r, derivative_kind, nd) for r in range(self.state_rank)]
            sv = [self._projection(previous, f"state_v_{layer}", r, derivative_kind, nd) for r in range(self.state_rank)]
            sg = self._array(f"state_gate_{layer}")
            sgids = self._indices(f"state_gate_{layer}") if pm else None
            state = [
                _gated(a.product(b), sg[r], None if sgids is None else sgids[r], nd)
                for r, (a, b) in enumerate(zip(su, sv))
            ]
            so = self._array(f"state_out_{layer}")
            soids = self._indices(f"state_out_{layer}") if pm else None
            current = []
            for c in range(self.width):
                source = self._bias(layer, c, derivative_kind, nd)
                source.add_scaled(_weighted_sum(
                    hidden_features, hidden_out[c],
                    None if hidden_out_ids is None else hidden_out_ids[c], nd
                ), 1.0)
                source.add_scaled(_weighted_sum(
                    cross, co[c], None if coids is None else coids[c], nd
                ), 1.0)
                source.add_scaled(_weighted_sum(
                    state, so[c], None if soids is None else soids[c], nd
                ), 1.0)
                response = source.response(self.targets[c], self.lambdas)
                current.append(_gated(
                    response, channel_gates[layer, c],
                    None if channel_ids is None else channel_ids[layer, c], nd
                ))
            previous = current
            layers.append(tuple(current))
        return initial, tuple(layers), nd

    def evaluate_with_jacobians(self, t, *, a0, operating, derivative_kind):
        t = float(t)
        if not np.isfinite(t) or t < 0 or t > self.max_response_time:
            raise ValueError(f"segment time must be in [0, {self.max_response_time:g}]")
        if self.depth == 1:
            return self._depth_one_evaluate(
                t, a0=a0, operating=operating, derivative_kind=derivative_kind
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
        return self.evaluate_with_jacobians(t, a0=a0, operating=operating, derivative_kind=None)[:2]

    def evaluate_parameter_jacobian(self, t, *, a0, operating):
        return self.evaluate_with_jacobians(t, a0=a0, operating=operating, derivative_kind="parameter")

    def evaluate_initial_jacobian(self, t, *, a0, operating):
        return self.evaluate_with_jacobians(t, a0=a0, operating=operating, derivative_kind="initial")

    def evaluate_operating_jacobian(self, t, *, a0, operating):
        return self.evaluate_with_jacobians(t, a0=a0, operating=operating, derivative_kind="operating")



__all__ = ["_GenericAnalyticMixin"]
