"""Finite-horizon residual optimizer for the fixed analytic network."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np
import scipy.linalg

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from sdfmpneo.em.modal_heat import heat_source_for_reduced_model
from .research_config import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network

_HARD_WEIGHT_STRENGTH = 24.0
_GN_FEEDBACK_SUBSPACE_RANK = 4


def _work(monitor, label, completed, total):
    if monitor is None:
        return
    monitor.checkpoint()
    activity = getattr(monitor, "activity", None)
    if activity is not None:
        activity(label, int(completed), int(total))


def _physics_vector_field(field, state, operating):
    """Evaluate the exact physical RHS without constructing its state Jacobian."""
    a = np.asarray(state, dtype=float)
    u = np.asarray(operating, dtype=float)

    if hasattr(field, "split") and hasattr(field, "thermal_model"):
        context, current = field.split(u)
        rhs = context.rhs.evaluate(current)
        heat = heat_source_for_reduced_model(context.em, a, rhs)
        return np.linalg.solve(context.M, -context.K @ a + heat)

    if hasattr(field, "em_model") and hasattr(field, "thermal_model") and hasattr(field, "rhs"):
        rhs = field.rhs(u if getattr(field, "rhs_map", None) is not None else None)
        heat = heat_source_for_reduced_model(field.em_model, a, rhs)
        forcing = np.asarray(
            getattr(field, "thermal_forcing", np.zeros_like(a)), dtype=float
        )
        return (
            -np.asarray(field.thermal_model.lambdas, dtype=float) * a
            + heat
            + forcing
        )

    if hasattr(field, "vector_field"):
        return np.asarray(field.vector_field(a, u), dtype=float)
    return np.asarray(field.evaluate(a, u).vector_field, dtype=float)


def _gn_field_jacobian(field, state, operating):
    """Cheap dissipative baseline Jacobian used in the inexact GN model."""
    u = np.asarray(operating, dtype=float)
    if hasattr(field, "split") and hasattr(field, "thermal_model"):
        context, _ = field.split(u)
        return np.linalg.solve(context.M, -context.K)
    if hasattr(field, "em_model") and hasattr(field, "thermal_model"):
        return -np.diag(np.asarray(field.thermal_model.lambdas, dtype=float))
    return np.asarray(field.evaluate(state, u).vector_field_jacobian, dtype=float)


def _gn_apply_field_jacobian(
    field,
    state,
    operating,
    state_parameter_jacobian,
    *,
    feedback_rank=_GN_FEEDBACK_SUBSPACE_RANK,
):
    """Approximate ``J_F @ J_a`` in the state-sensitivity subspace.

    Building the complete electromagnetic heat Jacobian is O(r^2) in FE loss
    derivatives.  The network parameter Jacobian only needs its action on
    ``J_a``.  Start from the exact thermal diffusion Jacobian, identify the
    dominant left singular directions of ``J_a``, and correct the field action
    along a few of those directions using central differences of the *exact*
    fast physical residual.  If ``J_a`` is low rank this recovers the complete
    action on its range, while high-rank problems pay only 2*q extra physical
    forward evaluations per hard point (q=4 by default).
    """
    a = np.asarray(state, dtype=float)
    u = np.asarray(operating, dtype=float)
    Ja = np.asarray(state_parameter_jacobian, dtype=float)
    baseline = _gn_field_jacobian(field, a, u)
    action = baseline @ Ja
    if Ja.size == 0 or int(feedback_rank) <= 0 or not np.any(Ja):
        return action

    gram = Ja @ Ja.T
    gram = 0.5 * (gram + gram.T)
    values, vectors = np.linalg.eigh(gram)
    scale = float(np.max(values, initial=0.0))
    if not np.isfinite(scale) or scale <= 0.0:
        return action
    threshold = 1e-12 * scale
    ids = np.flatnonzero(values > threshold)
    if ids.size == 0:
        return action
    ids = ids[-min(int(feedback_rank), ids.size):][::-1]

    base_scale = max(1.0, float(np.linalg.norm(a)))
    step0 = np.cbrt(np.finfo(float).eps) * base_scale
    for idx in ids:
        direction = vectors[:, idx]
        step = step0 / max(1.0, float(np.linalg.norm(direction)))
        plus = _physics_vector_field(field, a + step * direction, u)
        minus = _physics_vector_field(field, a - step * direction, u)
        exact_direction = (plus - minus) / (2.0 * step)
        correction = exact_direction - baseline @ direction
        coordinates = direction @ Ja
        action += correction[:, None] * coordinates[None, :]
    return action


def _evaluate_network(network, field, points, *, jacobian=False, monitor=None, work_label=None):
    points = np.asarray(points, dtype=float)
    n = network.n_modes
    records = []
    total = len(points)
    for i, point in enumerate(points):
        _work(
            monitor,
            work_label or ("physics_jacobian" if jacobian else "physics_residual"),
            i,
            total,
        )
        initial, operating, time = point[:n], point[n:-1], float(point[-1])
        if jacobian:
            a, da, ja, jda = network.evaluate_parameter_jacobian(
                time, a0=initial, operating=operating
            )
            F = _physics_vector_field(field, a, operating)
            JFJa = _gn_apply_field_jacobian(field, a, operating, ja)
            records.append(
                SimpleNamespace(
                    residual=np.asarray(da - F, dtype=float),
                    parameter_jacobian=np.asarray(jda - JFJa, dtype=float),
                )
            )
        else:
            a, da = network.evaluate(time, a0=initial, operating=operating)
            F = _physics_vector_field(field, a, operating)
            records.append(SimpleNamespace(residual=np.asarray(da - F, dtype=float)))
    _work(
        monitor,
        work_label or ("physics_jacobian" if jacobian else "physics_residual"),
        total,
        total,
    )
    return records


def _evaluate_semigroup(network, rows, *, jacobian=False, monitor=None, work_label=None):
    rows = np.asarray(rows, dtype=float)
    n = network.n_modes
    horizon = float(network.max_response_time)
    records = []
    total_rows = len(rows)
    for i, row in enumerate(rows):
        _work(
            monitor,
            work_label or ("restart_jacobian" if jacobian else "restart_residual"),
            i,
            total_rows,
        )
        initial = row[:n]
        operating = row[n:-2]
        t1, t2 = float(row[-2]), float(row[-1])
        total = t1 + t2
        if total > horizon + 64.0 * np.finfo(float).eps * horizon:
            raise ValueError("semigroup sample exceeds max_response_time")
        total = min(total, horizon)
        if jacobian:
            direct, _, Jdirect, _ = network.evaluate_parameter_jacobian(
                total, a0=initial, operating=operating
            )
            first, _, Jfirst, _ = network.evaluate_parameter_jacobian(
                t1, a0=initial, operating=operating
            )
            restarted, _, Jrestart, _ = network.evaluate_parameter_jacobian(
                t2, a0=first, operating=operating
            )
            _, _, Jinitial, _ = network.evaluate_initial_jacobian(
                t2, a0=first, operating=operating
            )
            defect = np.asarray(direct - restarted, dtype=float)
            jac = np.asarray(
                Jdirect - (Jrestart + Jinitial @ Jfirst), dtype=float
            )
            records.append(
                SimpleNamespace(
                    residual=defect / horizon,
                    raw_defect=defect,
                    parameter_jacobian=jac / horizon,
                )
            )
        else:
            direct, _ = network.evaluate(total, a0=initial, operating=operating)
            first, _ = network.evaluate(t1, a0=initial, operating=operating)
            restarted, _ = network.evaluate(t2, a0=first, operating=operating)
            defect = np.asarray(direct - restarted, dtype=float)
            records.append(
                SimpleNamespace(residual=defect / horizon, raw_defect=defect)
            )
    _work(
        monitor,
        work_label or ("restart_jacobian" if jacobian else "restart_residual"),
        total_rows,
        total_rows,
    )
    return records


def _metrics(records):
    norms = np.asarray(
        [np.linalg.norm(record.residual) for record in records], dtype=float
    )
    objective = float(np.mean(norms * norms)) if norms.size else 0.0
    return objective, float(np.max(norms, initial=0.0)), norms


def _combined_metrics(physics_records, semigroup_records):
    records = list(physics_records) + list(semigroup_records)
    objective, maximum, norms = _metrics(records)
    _, physics_max, physics_norms = _metrics(physics_records)
    _, semigroup_max, semigroup_norms = _metrics(semigroup_records)
    return SimpleNamespace(
        records=records,
        objective=objective,
        maximum=maximum,
        norms=norms,
        physics_max=physics_max,
        physics_norms=physics_norms,
        semigroup_max=semigroup_max,
        semigroup_norms=semigroup_norms,
    )


def _evaluate_all(network, field, physics_points, semigroup_points, *, jacobian=False, monitor=None):
    physics = _evaluate_network(
        network, field, physics_points, jacobian=jacobian, monitor=monitor
    )
    semigroup = _evaluate_semigroup(
        network, semigroup_points, jacobian=jacobian, monitor=monitor
    )
    return _combined_metrics(physics, semigroup)


def _hard_weights(norms, tolerance):
    values = np.asarray(norms, dtype=float)
    excess = np.maximum(values - float(tolerance), 0.0)
    peak = float(np.max(excess, initial=0.0))
    if peak <= 0.0 or not np.isfinite(peak):
        return np.ones_like(values)
    ratio = excess / peak
    return 1.0 + _HARD_WEIGHT_STRENGTH * ratio * ratio


def _accept(old_norms, new_norms, tolerance, weights):
    old = np.asarray(old_norms, dtype=float)
    new = np.asarray(new_norms, dtype=float)
    protected = old <= tolerance
    margin = 64.0 * np.finfo(float).eps * max(float(tolerance), 1e-30)
    if np.any(new[protected] > float(tolerance) + margin):
        return False
    old_max = float(np.max(old, initial=0.0))
    new_max = float(np.max(new, initial=0.0))
    numerical = 64.0 * np.finfo(float).eps * max(
        old_max, float(tolerance), 1e-30
    )
    if new_max < old_max - numerical:
        return True
    if new_max <= old_max + numerical:
        return float(np.dot(weights, new * new)) < float(
            np.dot(weights, old * old)
        ) - numerical
    return False


def _solve_direction(records, weights, damping, network, parameter_indices=None):
    residual = np.vstack([record.residual for record in records])
    jacobian = np.stack([record.parameter_jacobian for record in records])
    if parameter_indices is None:
        parameter_indices = np.arange(network.parameter_count, dtype=int)
    else:
        parameter_indices = np.asarray(parameter_indices, dtype=int).reshape(-1)
    if parameter_indices.size == 0:
        return np.zeros(network.parameter_count)
    jacobian = jacobian[:, :, parameter_indices]
    root = np.sqrt(np.asarray(weights, dtype=float))[:, None]
    rw = (root * residual).reshape(-1)
    Jw = (root[:, :, None] * jacobian).reshape(-1, parameter_indices.size)
    scales = np.linalg.norm(Jw, axis=0)
    scales[scales < 1e-14] = 1.0
    Jn = Jw / scales
    m, p = Jn.shape
    mu = max(float(damping), 1e-12)
    if m <= p:
        system = Jn @ Jn.T
        system.flat[::m + 1] += mu
        y = scipy.linalg.solve(
            system, -rw, assume_a="pos", check_finite=False
        )
        q = Jn.T @ y
    else:
        system = Jn.T @ Jn
        system.flat[::p + 1] += mu
        q = scipy.linalg.solve(
            system, -(Jn.T @ rw), assume_a="pos", check_finite=False
        )
    local_delta = q / scales
    delta = np.zeros(network.parameter_count)
    delta[parameter_indices] = local_delta
    trust = 2.0 * max(
        1.0, float(np.linalg.norm(network.parameters[parameter_indices]))
    )
    norm = float(np.linalg.norm(local_delta))
    if norm > trust:
        delta *= trust / norm
    return delta


def _unique_rows(*arrays, width):
    rows, seen = [], set()
    for array in arrays:
        values = np.asarray(array, dtype=float)
        if values.size == 0:
            continue
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError("collocation dimensions do not match")
        for row in values:
            key = tuple(float(v) for v in row)
            if key not in seen:
                seen.add(key)
                rows.append(row.copy())
    return np.vstack(rows) if rows else np.empty((0, width), dtype=float)


def _hard_subset(points, norms, budget):
    points = np.asarray(points, dtype=float)
    norms = np.asarray(norms, dtype=float)
    if len(points) != len(norms):
        raise ValueError("residual norms do not match collocation points")
    budget = min(len(points), max(0, int(budget)))
    if budget == 0:
        return points[:0]
    if budget == len(points):
        return points
    ids = np.argpartition(norms, -budget)[-budget:]
    ids = ids[np.argsort(norms[ids])[::-1]]
    return points[ids]


def _linearization(network, field, points, semigroup_points, evaluated, config, monitor):
    physics_budget = min(
        config.jacobian_point_budget,
        max(1, 768 // max(1, network.n_modes)),
    )
    physics_subset = _hard_subset(
        points, evaluated.physics_norms, physics_budget
    )
    physics = _evaluate_network(
        network,
        field,
        physics_subset,
        jacobian=True,
        monitor=monitor,
        work_label="physics_jacobian",
    )
    semigroup = []
    if len(semigroup_points):
        sg_subset = _hard_subset(
            semigroup_points,
            evaluated.semigroup_norms,
            min(
                config.semigroup_jacobian_point_budget,
                max(1, 384 // max(1, network.n_modes)),
            ),
        )
        semigroup = _evaluate_semigroup(
            network,
            sg_subset,
            jacobian=True,
            monitor=monitor,
            work_label="restart_jacobian",
        )
    return _combined_metrics(physics, semigroup)


def _prune(network, field, physics_points, semigroup_points, tolerance, relative_budget, rounds, monitor, config):
    current = network
    for _ in range(rounds):
        entries = [
            entry
            for entry in current.structure_gate_entries()
            if entry["value"] != 0.0
        ]
        if not entries:
            break
        full = _evaluate_all(
            current, field, physics_points, semigroup_points, monitor=monitor
        )
        evaluated = _linearization(
            current,
            field,
            physics_points,
            semigroup_points,
            full,
            config,
            monitor,
        )
        jac = np.stack(
            [record.parameter_jacobian for record in evaluated.records]
        )
        scored = []
        for entry in entries:
            column = jac[:, :, entry["parameter_index"]]
            effect = abs(entry["value"]) * float(
                np.max(np.linalg.norm(column, axis=1), initial=0.0)
            )
            scored.append((effect, entry["parameter_index"]))
        scored.sort()
        budget = max(0.0, tolerance - full.maximum) + relative_budget * tolerance
        chosen, used = [], 0.0
        for effect, index in scored:
            if effect == 0.0 or used + effect <= budget:
                chosen.append(index)
                used += effect
            else:
                break
        if not chosen:
            break
        accepted = None
        count = len(chosen)
        while count:
            theta = current.parameters.copy()
            theta[np.asarray(chosen[:count], dtype=int)] = 0.0
            trial = current.with_parameters(theta)
            result = _evaluate_all(
                trial, field, physics_points, semigroup_points, monitor=monitor
            )
            if result.maximum <= tolerance:
                accepted = trial
                break
            count //= 2
        if accepted is None:
            break
        current = accepted
    return current


__all__ = []
