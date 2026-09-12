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

    def _first_layer_groups(self, base, static, derivative_kind, nd):
        pm = derivative_kind == "parameter"
        linear_features = [
            self._projection(base, "input_linear_in", r, derivative_kind, nd)
            for r in range(self.linear_rank)
        ]
        qu = [
            self._projection(base, "quadratic_u", r, derivative_kind, nd)
            for r in range(self.quadratic_rank)
        ]
        qv = [
            self._projection(static, "quadratic_v", r, derivative_kind, nd)
            for r in range(self.quadratic_rank)
        ]
        qg = self._array("quadratic_gate")
        qids = self._indices("quadratic_gate") if pm else None
        quadratic = [
            _gated(a.product(b), qg[r], None if qids is None else qids[r], nd)
            for r, (a, b) in enumerate(zip(qu, qv))
        ]
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
        return (
            ("input_linear_out", linear_features),
            ("quadratic_out", quadratic),
            ("square_out", square_features),
        )

    def _deeper_layer_groups(self, previous, static, layer, derivative_kind, nd):
        pm = derivative_kind == "parameter"
        groups = []
        hr = self.layer_hidden_ranks[layer - 1]
        if hr:
            hidden = [
                self._projection(previous, f"hidden_linear_in_{layer}", r, derivative_kind, nd)
                for r in range(hr)
            ]
            groups.append((f"hidden_linear_out_{layer}", hidden))
        cr = self.layer_cross_ranks[layer - 1]
        if cr:
            ci = [
                self._projection(static, f"cross_input_{layer}", r, derivative_kind, nd)
                for r in range(cr)
            ]
            ch = [
                self._projection(previous, f"cross_hidden_{layer}", r, derivative_kind, nd)
                for r in range(cr)
            ]
            cg = self._array(f"cross_gate_{layer}")
            cgids = self._indices(f"cross_gate_{layer}") if pm else None
            cross = [
                _gated(a.product(b), cg[r], None if cgids is None else cgids[r], nd)
                for r, (a, b) in enumerate(zip(ci, ch))
            ]
            groups.append((f"cross_out_{layer}", cross))
        sr = self.layer_state_ranks[layer - 1]
        if sr:
            su = [
                self._projection(previous, f"state_u_{layer}", r, derivative_kind, nd)
                for r in range(sr)
            ]
            sv = [
                self._projection(previous, f"state_v_{layer}", r, derivative_kind, nd)
                for r in range(sr)
            ]
            budget = self.state_feature_term_budget
            su = [signal.dominant_terms(budget) for signal in su]
            sv = [signal.dominant_terms(budget) for signal in sv]
            sg = self._array(f"state_gate_{layer}")
            sgids = self._indices(f"state_gate_{layer}") if pm else None
            state = [
                _gated(a.product(b), sg[r], None if sgids is None else sgids[r], nd)
                for r, (a, b) in enumerate(zip(su, sv))
            ]
            groups.append((f"state_out_{layer}", state))
        return tuple(groups)

    def _layer_groups(self, base, static, previous, layer, derivative_kind, nd):
        if layer == 0:
            return self._first_layer_groups(base, static, derivative_kind, nd)
        return self._deeper_layer_groups(previous, static, layer, derivative_kind, nd)

    def _layer_amplitudes_are_zero(self, layer):
        """Return whether a residual-correction layer is still completely inactive."""
        if int(layer) <= 0:
            return False
        names = (
            f"bias_{layer}",
            f"hidden_linear_out_{layer}",
            f"cross_out_{layer}",
            f"state_out_{layer}",
        )
        return all(not np.any(self._array(name)) for name in names)

    def _make_layer(self, base, static, previous, layer, derivative_kind, nd):
        # Fresh funnel layers start with exactly zero amplitudes. Avoid building
        # hidden/state ExpPoly products until a layer is actually activated.
        # This keeps initial residual evaluation at essentially first-layer cost.
        if derivative_kind != "parameter" and self._layer_amplitudes_are_zero(layer):
            return tuple(
                _ExpPoly.zero(self.n_modes, nd)
                for _ in range(self.layer_widths[layer])
            )
        pm = derivative_kind == "parameter"
        groups = self._layer_groups(base, static, previous, layer, derivative_kind, nd)
        gate_values = self._array("channel_gate")[layer]
        gate_ids = self._indices("channel_gate")[layer] if pm else None
        current = []
        for c in range(self.layer_widths[layer]):
            source = self._bias(layer, c, derivative_kind, nd)
            for out_name, features in groups:
                if not features:
                    continue
                out = self._array(out_name)
                out_ids = self._indices(out_name) if pm else None
                source.add_scaled(
                    _weighted_sum(
                        features,
                        out[c],
                        None if out_ids is None else out_ids[c],
                        nd,
                    ),
                    1.0,
                )
            target = int(self.layer_targets[layer][c])
            response = source.response(target, self.lambdas)
            current.append(_gated(
                response,
                gate_values[c],
                None if gate_ids is None else gate_ids[c],
                nd,
            ))
        return tuple(current)

    def _forward_signals(self, a0, operating, derivative_kind=None, stop_layer=None):
        initial, base, static, nd = self._base_signals(a0, operating, derivative_kind)
        if stop_layer is None:
            stop_layer = self.depth - 1
        stop_layer = int(stop_layer)
        if stop_layer < 0 or stop_layer >= self.depth:
            raise ValueError("stop_layer is outside the response network")
        previous = None
        layers = []
        for layer in range(stop_layer + 1):
            current = self._make_layer(
                base, static, previous, layer, derivative_kind, nd
            )
            layers.append(current)
            previous = current
        return initial, tuple(layers), nd

    def _evaluate_layers(self, t, initial, layers, nd, derivative_kind):
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
        for layer_index, layer in enumerate(layers):
            targets = self.layer_targets[layer_index]
            for c, signal in enumerate(layer):
                value, slope, g, sg = signal.evaluate(t, self.lambdas)
                target = int(targets[c])
                a[target] += value
                da[target] += slope
                if nd:
                    ja[target] += g
                    jda[target] += sg
        return a, da, ja, jda

    def evaluate_with_jacobians(self, t, *, a0, operating, derivative_kind):
        t = float(t)
        if not np.isfinite(t) or t < 0 or t > self.max_response_time:
            raise ValueError(f"segment time must be in [0, {self.max_response_time:g}]")
        if self.depth == 1:
            return self._depth_one_evaluate(
                t, a0=a0, operating=operating, derivative_kind=derivative_kind
            )
        if derivative_kind == "parameter" and self.n_modes >= 96:
            raise RuntimeError(
                "high-rank multilayer training must use the exact layer-amplitude Jacobian"
            )
        initial, layers, nd = self._forward_signals(a0, operating, derivative_kind)
        return self._evaluate_layers(t, initial, layers, nd, derivative_kind)

    def _layer_context(self, a0, operating, layer):
        initial, base, static, nd = self._base_signals(a0, operating, None)
        previous = None
        layers = []
        for index in range(layer + 1):
            current = self._make_layer(base, static, previous, index, None, nd)
            layers.append(current)
            if index == layer:
                break
            previous = current
        return initial, base, static, previous, tuple(layers)

    def evaluate_layer_amplitude_jacobian(self, t, *, a0, operating, layer):
        """Exact Jacobian for one response layer's linear amplitude block.

        Earlier layers are treated as frozen analytic features and later layers
        are not part of this stage. This is the scalable training primitive for
        high-rank funnel networks: no exponential term carries a dense global
        parameter tangent.
        """
        t = float(t)
        layer = int(layer)
        if not np.isfinite(t) or t < 0 or t > self.max_response_time:
            raise ValueError(f"segment time must be in [0, {self.max_response_time:g}]")
        if layer < 0 or layer >= self.depth:
            raise ValueError("response layer index is out of range")
        initial, base, static, previous, layers = self._layer_context(a0, operating, layer)
        a, da, _, _ = self._evaluate_layers(t, initial, layers, 0, None)
        groups = self._layer_groups(base, static, previous, layer, None, 0)
        ids = self.layer_amplitude_parameter_indices(layer)
        id_to_column = {int(value): i for i, value in enumerate(ids)}
        ja = np.zeros((self.n_modes, len(ids)))
        jda = np.zeros_like(ja)
        gates = self._array("channel_gate")[layer]

        for c in range(self.layer_widths[layer]):
            target = int(self.layer_targets[layer][c])
            gate = float(gates[c])
            pid = int(self._indices(f"bias_{layer}")[c])
            column = id_to_column[pid]
            signal = _ExpPoly.constant_term(
                self.n_modes, 1.0
            ).response(target, self.lambdas)
            value, slope, _, _ = signal.evaluate(t, self.lambdas)
            ja[target, column] += gate * value
            jda[target, column] += gate * slope
            for out_name, features in groups:
                out_ids = self._indices(out_name)
                for r, feature in enumerate(features):
                    pid = int(out_ids[c, r])
                    column = id_to_column.get(pid)
                    if column is None:
                        continue
                    signal = feature.response(target, self.lambdas)
                    value, slope, _, _ = signal.evaluate(t, self.lambdas)
                    ja[target, column] += gate * value
                    jda[target, column] += gate * slope
        return a, da, ja, jda, ids

    def first_layer_source_design(self, *, a0, operating):
        """Return the t=0 source feature vector for least-squares initialization."""
        _, base, static, _ = self._base_signals(a0, operating, None)
        groups = self._first_layer_groups(base, static, None, 0)
        values = [1.0]
        sizes = []
        for out_name, features in groups:
            local = []
            for feature in features:
                value, _, _, _ = feature.evaluate(0.0, self.lambdas)
                local.append(float(value))
            values.extend(local)
            sizes.append((out_name, len(local)))
        return np.asarray(values, dtype=float), tuple(sizes)

    def fit_first_layer_source(self, samples, desired_source):
        """Least-squares fit the complete first response source at t=0."""
        samples = np.asarray(samples, dtype=float)
        desired = np.asarray(desired_source, dtype=float)
        if samples.ndim != 2 or samples.shape[1] != self.input_dimension:
            raise ValueError("source prefit samples do not match network inputs")
        if desired.shape != (len(samples), self.n_modes):
            raise ValueError("source prefit target shape does not match thermal rank")
        if self.channels_per_mode != 1 or self.layer_widths[0] != self.n_modes:
            raise ValueError("source prefit requires one first-layer response channel per thermal mode")
        rows = []
        layout = None
        for row in samples:
            design, local_layout = self.first_layer_source_design(
                a0=row[:self.n_modes], operating=row[self.n_modes:]
            )
            rows.append(design)
            if layout is None:
                layout = local_layout
        X = np.vstack(rows)
        coefficients, *_ = np.linalg.lstsq(X, desired, rcond=None)
        theta = self.parameters.copy()
        targets = self.layer_targets[0]
        theta[self._indices("bias_0")] = coefficients[0, targets]
        offset = 1
        for out_name, width in layout:
            values = coefficients[offset:offset + width]
            target_values = values[:, targets].T
            theta[self._indices(out_name)] = target_values
            offset += width
        fitted = X @ coefficients
        residual = desired - fitted
        return self.with_parameters(theta), residual

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


__all__ = ["_GenericAnalyticMixin"]
