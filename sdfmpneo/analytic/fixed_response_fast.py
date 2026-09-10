from __future__ import annotations

import numpy as np


class _DepthOneFastMixin:
        def _response_kernel(self, t, source_rates):
            """Stable response of exp(-mu*t) through each channel target pole."""
            t = float(t)
            source = np.asarray(source_rates, float).reshape(1, -1)
            target = self.lambdas[self.targets].reshape(-1, 1)
            if t == 0.0:
                value = np.zeros((self.width, source.shape[1]))
                slope = np.ones_like(value)
                return value, slope
            delta = target - source
            a = np.exp(-source * t)
            b = np.exp(-target * t)
            z = delta * t
            value = np.empty_like(delta)
            near = np.abs(z) <= 1e-5
            # exp(-target*t) * expm1((target-source)*t)/(target-source)
            # is cancellation-free near equal poles.  The direct divided difference
            # is safer away from equality because both exponentials are non-growing.
            if np.any(near):
                zz = z[near]
                exprel = 1.0 + zz / 2.0 + zz * zz / 6.0 + zz**3 / 24.0 + zz**4 / 120.0
                value[near] = (b * np.ones_like(source))[near] * t * exprel
            if np.any(~near):
                value[~near] = (a - b)[~near] / delta[~near]
            slope = a - target * value
            return value, slope

        def _depth_one_data(self, t, a0, operating):
            initial = np.asarray(a0, float).reshape(-1)
            op = np.asarray(operating, float).reshape(-1)
            if (
                initial.shape != (self.n_modes,)
                or op.shape != (len(self.operating_names),)
                or np.any(~np.isfinite(initial))
                or np.any(~np.isfinite(op))
            ):
                raise ValueError("invalid initial/operating input")
            normalized = (np.concatenate([initial, op]) - self.input_center) / self.input_scale
            x = normalized[:self.n_modes]
            u = normalized[self.n_modes:]
            static = np.concatenate([[1.0], u])
            rth, drth = self._response_kernel(t, self.lambdas)
            rzero, drzero = self._response_kernel(t, [0.0])
            rzero = rzero[:, 0]; drzero = drzero[:, 0]
            rsq, drsq = self._response_kernel(t, 2.0 * self.lambdas)

            lin_in = self._array("input_linear_in")
            lin_out = self._array("input_linear_out")
            lin_static = lin_in[:, self.n_modes:] @ u
            lin_resp = (rth * x[None, :]) @ lin_in[:, :self.n_modes].T
            lin_resp += rzero[:, None] * lin_static[None, :]
            lin_slope = (drth * x[None, :]) @ lin_in[:, :self.n_modes].T
            lin_slope += drzero[:, None] * lin_static[None, :]

            qu = self._array("quadratic_u")
            qv = self._array("quadratic_v")
            qout = self._array("quadratic_out")
            qgate = self._array("quadratic_gate")
            qstatic = qu[:, self.n_modes:] @ u
            qdyn = (rth * x[None, :]) @ qu[:, :self.n_modes].T
            qdyn += rzero[:, None] * qstatic[None, :]
            qdyn_slope = (drth * x[None, :]) @ qu[:, :self.n_modes].T
            qdyn_slope += drzero[:, None] * qstatic[None, :]
            qv_static = qv @ static
            qresp = qdyn * qv_static[None, :]
            qslope = qdyn_slope * qv_static[None, :]

            sin = self._array("square_in")
            sout = self._array("square_out")
            sgate = self._array("square_gate")
            x2 = x * x
            sresp = (rsq * x2[None, :]) @ sin.T
            sslope = (drsq * x2[None, :]) @ sin.T

            bias = self._array("bias_0")
            y = bias * rzero
            dy = bias * drzero
            y += np.sum(lin_out * lin_resp, axis=1)
            dy += np.sum(lin_out * lin_slope, axis=1)
            y += np.sum(qout * qgate[None, :] * qresp, axis=1)
            dy += np.sum(qout * qgate[None, :] * qslope, axis=1)
            y += np.sum(sout * sgate[None, :] * sresp, axis=1)
            dy += np.sum(sout * sgate[None, :] * sslope, axis=1)
            return {
                "initial": initial, "x": x, "u": u, "static": static,
                "rth": rth, "drth": drth, "rzero": rzero, "drzero": drzero,
                "rsq": rsq, "drsq": drsq,
                "lin_in": lin_in, "lin_out": lin_out,
                "lin_resp": lin_resp, "lin_slope": lin_slope,
                "qu": qu, "qv": qv, "qout": qout, "qgate": qgate,
                "qstatic": qstatic, "qdyn": qdyn, "qdyn_slope": qdyn_slope,
                "qv_static": qv_static, "qresp": qresp, "qslope": qslope,
                "sin": sin, "sout": sout, "sgate": sgate,
                "sresp": sresp, "sslope": sslope,
                "y": y, "dy": dy,
            }

        def _depth_one_evaluate(self, t, *, a0, operating, derivative_kind):
            data = self._depth_one_data(t, a0, operating)
            initial = data["initial"]
            target = self.targets
            channel_gate = self._array("channel_gate")[0]
            decay = np.exp(-self.lambdas * float(t))
            a = initial * decay
            da = -self.lambdas * a
            np.add.at(a, target, channel_gate * data["y"])
            np.add.at(da, target, channel_gate * data["dy"])

            if derivative_kind is None:
                return a, da, np.zeros((self.n_modes, 0)), np.zeros((self.n_modes, 0))

            if derivative_kind == "parameter":
                p = self.parameter_count
                jy = np.zeros((self.width, p))
                jdy = np.zeros((self.width, p))
                rows = np.arange(self.width)
                # Biases.
                ids = self._indices("bias_0")
                jy[rows, ids] = data["rzero"]
                jdy[rows, ids] = data["drzero"]

                # Low-rank linear source.
                lin_out = data["lin_out"]
                out_ids = self._indices("input_linear_out")
                for c in range(self.width):
                    jy[c, out_ids[c]] = data["lin_resp"][c]
                    jdy[c, out_ids[c]] = data["lin_slope"][c]
                in_ids = self._indices("input_linear_in")
                xr = data["rth"] * data["x"][None, :]
                dxr = data["drth"] * data["x"][None, :]
                ur = data["rzero"][:, None] * data["u"][None, :]
                dur = data["drzero"][:, None] * data["u"][None, :]
                for k in range(self.linear_rank):
                    coeff = lin_out[:, k, None]
                    jy[:, in_ids[k, :self.n_modes]] = coeff * xr
                    jdy[:, in_ids[k, :self.n_modes]] = coeff * dxr
                    if len(data["u"]):
                        jy[:, in_ids[k, self.n_modes:]] = coeff * ur
                        jdy[:, in_ids[k, self.n_modes:]] = coeff * dur

                # Dynamic x static quadratic source.
                qout = data["qout"]; qgate = data["qgate"]
                qout_ids = self._indices("quadratic_out")
                qgate_ids = self._indices("quadratic_gate")
                qu_ids = self._indices("quadratic_u")
                qv_ids = self._indices("quadratic_v")
                for q in range(self.quadratic_rank):
                    common = qout[:, q] * qgate[q]
                    jy[rows, qout_ids[:, q]] = qgate[q] * data["qresp"][:, q]
                    jdy[rows, qout_ids[:, q]] = qgate[q] * data["qslope"][:, q]
                    jy[:, qgate_ids[q]] = qout[:, q] * data["qresp"][:, q]
                    jdy[:, qgate_ids[q]] = qout[:, q] * data["qslope"][:, q]
                    coeff = (common * data["qv_static"][q])[:, None]
                    jy[:, qu_ids[q, :self.n_modes]] = coeff * xr
                    jdy[:, qu_ids[q, :self.n_modes]] = coeff * dxr
                    if len(data["u"]):
                        jy[:, qu_ids[q, self.n_modes:]] = coeff * ur
                        jdy[:, qu_ids[q, self.n_modes:]] = coeff * dur
                    vcoeff = common * data["qdyn"][:, q]
                    dvcoeff = common * data["qdyn_slope"][:, q]
                    jy[:, qv_ids[q, 0]] = vcoeff
                    jdy[:, qv_ids[q, 0]] = dvcoeff
                    if len(data["u"]):
                        jy[:, qv_ids[q, 1:]] = vcoeff[:, None] * data["u"][None, :]
                        jdy[:, qv_ids[q, 1:]] = dvcoeff[:, None] * data["u"][None, :]

                # Low-rank map of modewise thermal squares.
                sout = data["sout"]; sgate = data["sgate"]
                sout_ids = self._indices("square_out")
                sgate_ids = self._indices("square_gate")
                sin_ids = self._indices("square_in")
                x2r = data["rsq"] * (data["x"] * data["x"])[None, :]
                dx2r = data["drsq"] * (data["x"] * data["x"])[None, :]
                for q in range(self.square_rank):
                    common = sout[:, q] * sgate[q]
                    jy[rows, sout_ids[:, q]] = sgate[q] * data["sresp"][:, q]
                    jdy[rows, sout_ids[:, q]] = sgate[q] * data["sslope"][:, q]
                    jy[:, sgate_ids[q]] = sout[:, q] * data["sresp"][:, q]
                    jdy[:, sgate_ids[q]] = sout[:, q] * data["sslope"][:, q]
                    jy[:, sin_ids[q]] = common[:, None] * x2r
                    jdy[:, sin_ids[q]] = common[:, None] * dx2r

                # Apply channel gates, then expose gate derivatives themselves.
                jy *= channel_gate[:, None]
                jdy *= channel_gate[:, None]
                gate_ids = self._indices("channel_gate")[0]
                jy[rows, gate_ids] += data["y"]
                jdy[rows, gate_ids] += data["dy"]
                ja = np.zeros((self.n_modes, p)); jda = np.zeros_like(ja)
                for c in range(self.width):
                    ja[target[c]] += jy[c]
                    jda[target[c]] += jdy[c]
                return a, da, ja, jda

            if derivative_kind == "initial":
                scale = self.input_scale[:self.n_modes]
                lin = (data["lin_out"] @ data["lin_in"][:, :self.n_modes])
                dya = lin * data["rth"] / scale[None, :]
                ddya = lin * data["drth"] / scale[None, :]
                qmix = (data["qout"] * data["qgate"][None, :] * data["qv_static"][None, :]) @ data["qu"][:, :self.n_modes]
                dya += qmix * data["rth"] / scale[None, :]
                ddya += qmix * data["drth"] / scale[None, :]
                smix = (data["sout"] * data["sgate"][None, :]) @ data["sin"]
                square_factor = 2.0 * data["x"] / scale
                dya += smix * data["rsq"] * square_factor[None, :]
                ddya += smix * data["drsq"] * square_factor[None, :]
                dya *= channel_gate[:, None]; ddya *= channel_gate[:, None]
                ja = np.diag(decay); jda = np.diag(-self.lambdas * decay)
                for c in range(self.width):
                    ja[target[c]] += dya[c]
                    jda[target[c]] += ddya[c]
                return a, da, ja, jda

            if derivative_kind == "operating":
                m = len(self.operating_names)
                if m == 0:
                    return a, da, np.zeros((self.n_modes, 0)), np.zeros((self.n_modes, 0))
                scale = self.input_scale[self.n_modes:]
                lin_static = data["lin_out"] @ data["lin_in"][:, self.n_modes:]
                dyu = lin_static * data["rzero"][:, None]
                ddyu = lin_static * data["drzero"][:, None]
                common = data["qout"] * data["qgate"][None, :]
                # derivative through qv(u)
                dyu += (common * data["qdyn"]) @ data["qv"][:, 1:]
                ddyu += (common * data["qdyn_slope"]) @ data["qv"][:, 1:]
                # derivative through the static portion of qu(u)
                qv_common = common * data["qv_static"][None, :]
                dyu += (qv_common @ data["qu"][:, self.n_modes:]) * data["rzero"][:, None]
                ddyu += (qv_common @ data["qu"][:, self.n_modes:]) * data["drzero"][:, None]
                dyu = dyu / scale[None, :]
                ddyu = ddyu / scale[None, :]
                dyu *= channel_gate[:, None]; ddyu *= channel_gate[:, None]
                ja = np.zeros((self.n_modes, m)); jda = np.zeros_like(ja)
                for c in range(self.width):
                    ja[target[c]] += dyu[c]
                    jda[target[c]] += ddyu[c]
                return a, da, ja, jda

            raise ValueError("unknown derivative kind")


__all__ = ["_DepthOneFastMixin"]
