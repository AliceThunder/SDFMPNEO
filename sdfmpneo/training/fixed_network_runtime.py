from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import os

import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork

_ORIGINAL_TRAIN = None
_ORIGINAL_STABLE = None
_ORIGINAL_STABLE_JAC = None
_ORIGINAL_SAVE = None
_ORIGINAL_LOAD = None


def _setting(config, attr, env, default):
    value = getattr(config, attr, None)
    if value is not None:
        return int(value)
    return int(os.environ.get(env, default))


def _setting_float(config, attr, env, default):
    value = getattr(config, attr, None)
    if value is not None:
        return float(value)
    return float(os.environ.get(env, default))


def _capacity_setting(config, new_attr, legacy_attr, env, default):
    value = getattr(config, new_attr, None)
    if value is None:
        value = getattr(config, legacy_attr, None)
    if value is not None:
        return int(value)
    return int(os.environ.get(env, default))


def _default_capacity(n_modes, n_operating):
    """Problem-size dependent safety caps; effective capacity is pruned later."""
    input_dimension = 1 + int(n_modes) + int(n_operating)
    depth = 5 if n_modes <= 8 else 4
    channels = 2
    quadratic = min(12, max(4, (input_dimension + 1) // 2))
    cross = min(6, max(2, int(np.ceil(np.sqrt(input_dimension)))))
    state = min(4, max(2, int(np.ceil(np.sqrt(max(1, n_modes))))))
    return depth, channels, quadratic, cross, state


def _make_network(field, config, graph):
    if isinstance(graph, FixedAnalyticResponseNetwork):
        return graph
    if graph is not None and getattr(graph, "response_nodes", ()):
        return None
    lambdas = np.asarray(field.thermal_model.lambdas, dtype=float)
    if graph is not None and hasattr(graph, "operating_names"):
        operating_names = tuple(graph.operating_names)
    else:
        operating_names = tuple(f"u{i}" for i in range(len(config.operating_lower)))
    lo = np.asarray(config.initial_lower + config.operating_lower, dtype=float)
    hi = np.asarray(config.initial_upper + config.operating_upper, dtype=float)
    center = 0.5 * (lo + hi)
    scale = 0.5 * (hi - lo)
    scale[scale <= 0.0] = 1.0

    defaults = _default_capacity(len(lambdas), len(operating_names))
    depth = _capacity_setting(
        config, "fixed_network_max_depth", "fixed_network_depth",
        "SDFMPNEO_FIXED_NETWORK_DEPTH", defaults[0])
    channels = _capacity_setting(
        config, "fixed_network_max_channels_per_mode", "fixed_network_channels_per_mode",
        "SDFMPNEO_FIXED_NETWORK_CHANNELS_PER_MODE", defaults[1])
    quadratic = _capacity_setting(
        config, "fixed_network_max_quadratic_rank", "fixed_network_quadratic_rank",
        "SDFMPNEO_FIXED_NETWORK_QUADRATIC_RANK", defaults[2])
    cross = _capacity_setting(
        config, "fixed_network_max_cross_rank", "fixed_network_cross_rank",
        "SDFMPNEO_FIXED_NETWORK_CROSS_RANK", defaults[3])
    state = _capacity_setting(
        config, "fixed_network_max_state_rank", "fixed_network_state_rank",
        "SDFMPNEO_FIXED_NETWORK_STATE_RANK", defaults[4])

    return FixedAnalyticResponseNetwork(
        lambdas, operating_names, input_center=center, input_scale=scale,
        depth=depth, channels_per_mode=channels, quadratic_rank=quadratic,
        cross_rank=cross, state_rank=state)


def _prepare(field, points):
    prepare = getattr(field, "prepare_training_contexts", None)
    if prepare is not None:
        prepare(points)


def _evaluate_network(network, field, points, *, jacobian, monitor=None):
    from .parallel_runtime import _ordered_map

    values = np.ascontiguousarray(points, dtype=float)
    n = network.n_modes

    def one(point):
        if monitor is not None:
            monitor.checkpoint()
        initial = point[:n]
        u = point[n:-1]
        if jacobian:
            a, da, ja, jda = network.evaluate_parameter_jacobian(
                float(point[-1]), a0=initial, operating=u)
            physical = field.evaluate(a, u)
            residual = da - physical.vector_field
            residual_jacobian = jda - physical.vector_field_jacobian @ ja
            return SimpleNamespace(
                a=a, residual=np.asarray(residual, dtype=float),
                J=np.asarray(physical.vector_field_jacobian, dtype=float),
                parameter_jacobian=np.asarray(residual_jacobian, dtype=float),
                initial=np.asarray(initial, dtype=float), u=np.asarray(u, dtype=float),
                time=float(point[-1]))
        a, da = network.evaluate(float(point[-1]), a0=initial, operating=u)
        if hasattr(field, "vector_field"):
            F = field.vector_field(a, u)
        else:
            F = field.evaluate(a, u).vector_field
        return SimpleNamespace(
            a=a, residual=np.asarray(da - F, dtype=float), J=None,
            initial=np.asarray(initial, dtype=float), u=np.asarray(u, dtype=float),
            time=float(point[-1]))

    return _ordered_map(one, values, monitor=monitor)


def _metrics(records):
    norms = np.asarray([np.linalg.norm(record.residual) for record in records], dtype=float)
    return float(np.mean(norms * norms)), float(np.max(norms, initial=0.0)), norms


def _solve_direction(records, point_weights, damping, network):
    residual = np.vstack([record.residual for record in records])
    jacobian = np.stack([record.parameter_jacobian for record in records])
    root = np.sqrt(np.asarray(point_weights, dtype=float))[:, None]
    rw = (root * residual).reshape(-1)
    Jw = (root[:, :, None] * jacobian).reshape(-1, network.parameter_count)
    scales = np.linalg.norm(Jw, axis=0)
    scales[scales < 1.0e-14] = 1.0
    Jn = Jw / scales
    m, p = Jn.shape
    mu = max(float(damping), 1.0e-12)
    if m <= p:
        system = Jn @ Jn.T
        system.flat[::m + 1] += mu
        try:
            y = np.linalg.solve(system, -rw)
        except np.linalg.LinAlgError:
            y = np.linalg.lstsq(system, -rw, rcond=None)[0]
        q = Jn.T @ y
    else:
        system = Jn.T @ Jn
        system.flat[::p + 1] += mu
        rhs = -(Jn.T @ rw)
        try:
            q = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            q = np.linalg.lstsq(system, rhs, rcond=None)[0]
    delta = q / scales
    trust = 2.0 * max(1.0, float(np.linalg.norm(network.parameters)))
    norm = float(np.linalg.norm(delta))
    if norm > trust:
        delta *= trust / norm
    return delta


def _unique_rows(*arrays, width):
    seen = set()
    rows = []
    for array in arrays:
        values = np.asarray(array, dtype=float)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError("collocation point dimensions do not match")
        for row in values:
            key = tuple(float(value) for value in row)
            if key not in seen:
                seen.add(key)
                rows.append(row.copy())
    return np.vstack(rows) if rows else np.empty((0, width), dtype=float)


def _fresh_checks(config, seed, excluded):
    candidate = np.asarray(config.points(validation=True, seed=seed), dtype=float)
    excluded_keys = {
        tuple(float(value) for value in row) for row in np.asarray(excluded, dtype=float)
    }
    rows = [row for row in candidate
            if tuple(float(value) for value in row) not in excluded_keys]
    return np.vstack(rows) if rows else np.empty((0, candidate.shape[1]), dtype=float)


def _zero_gate_indices(network, indices):
    theta = network.parameters.copy()
    theta[np.asarray(indices, dtype=int)] = 0.0
    return network.with_parameters(theta)


def _residual_sensitivity_prune(
    network, field, points, *, tolerance, monitor=None,
    relative_budget=0.10, max_rounds=3,
):
    """Prune existing gates by residual sensitivity, then verify nonlinear residual.

    This is not topology search: the maximum graph is fixed before training. The
    local residual Jacobian ranks already-existing continuous gates, and a small
    number of monotone zeroing trials only compresses a converged network.
    """
    tolerance = float(tolerance)
    relative_budget = max(0.0, float(relative_budget))
    current = network
    if len(points) == 0:
        return current

    for _ in range(max(0, int(max_rounds))):
        entries = [entry for entry in current.structure_gate_entries()
                   if entry["value"] != 0.0]
        if not entries:
            break
        linearized = _evaluate_network(
            current, field, points, jacobian=True, monitor=monitor)
        _, current_max, _ = _metrics(linearized)
        jacobian = np.stack([record.parameter_jacobian for record in linearized])
        scored = []
        for entry in entries:
            index = entry["parameter_index"]
            column = jacobian[:, :, index]
            point_effect = np.linalg.norm(column, axis=1)
            effect = abs(entry["value"]) * float(np.max(point_effect, initial=0.0))
            scored.append((effect, entry))
        scored.sort(key=lambda item: item[0])

        budget = max(0.0, tolerance - current_max) + relative_budget * tolerance
        selected = []
        accumulated = 0.0
        for effect, entry in scored:
            if effect == 0.0 or accumulated + effect <= budget:
                selected.append(entry["parameter_index"])
                accumulated += effect
            else:
                break
        if not selected:
            break

        accepted = None
        count = len(selected)
        while count:
            trial = _zero_gate_indices(current, selected[:count])
            records = _evaluate_network(
                trial, field, points, jacobian=False, monitor=monitor)
            _, maximum, _ = _metrics(records)
            if maximum <= tolerance:
                accepted = trial
                break
            count //= 2
        if accepted is None:
            break
        current = accepted

    return current


def train_fixed_analytic_response_network(
    field, config, *, graph=None, progress=None, monitor=None
):
    """Train one maximum-capacity analytic network using continuous residual optimization."""
    global _ORIGINAL_TRAIN
    from .max_residual_runtime import hard_point_weights_from_norms, max_first_accept
    from .research import ResearchTrainingReport

    network = _make_network(field, config, graph)
    if network is None:
        return _ORIGINAL_TRAIN(
            field, config, graph=graph, progress=progress, monitor=monitor)

    if (len(config.initial_lower) != network.n_modes
            or len(config.operating_lower) != len(network.operating_names)):
        raise ValueError("training domain does not match the fixed analytic network")

    points = np.asarray(config.points(validation=False), dtype=float)
    checks = np.asarray(config.points(validation=True), dtype=float)
    guards = np.empty((0, points.shape[1]), dtype=float)
    _prepare(field, points)
    if monitor is not None:
        monitor.retain(network)
        monitor.phase("initial_residual")
    records = _evaluate_network(network, field, points, jacobian=False, monitor=monitor)
    objective, maximum, norms = _metrics(records)
    initial_rms = float(np.sqrt(objective))
    history = [objective]
    if monitor is not None:
        monitor.record(network, objective, maximum, len(points))

    max_iterations = _setting(
        config, "fixed_network_max_iterations",
        "SDFMPNEO_FIXED_NETWORK_MAX_ITERATIONS", 36)
    max_validation_epochs = _setting(
        config, "fixed_network_validation_epochs",
        "SDFMPNEO_FIXED_NETWORK_VALIDATION_EPOCHS", 5)
    gate_shrink = _setting_float(
        config, "fixed_network_gate_shrink",
        "SDFMPNEO_FIXED_NETWORK_GATE_SHRINK", 0.0)
    prune_budget = _setting_float(
        config, "fixed_network_prune_relative_budget",
        "SDFMPNEO_FIXED_NETWORK_PRUNE_RELATIVE_BUDGET", 0.10)
    prune_rounds = _setting(
        config, "fixed_network_prune_rounds",
        "SDFMPNEO_FIXED_NETWORK_PRUNE_ROUNDS", 3)

    damping = 1.0e-3
    epoch = 0
    status = "stalled"
    validation_max = float("inf")
    iteration = 0
    final_validation_points = np.empty((0, points.shape[1]), dtype=float)

    while True:
        if maximum <= config.residual_tolerance:
            if monitor is not None:
                monitor.phase("validation")
            validation_points = _unique_rows(guards, checks, width=points.shape[1])
            final_validation_points = validation_points
            _prepare(field, validation_points)
            validation_records = _evaluate_network(
                network, field, validation_points, jacobian=False, monitor=monitor)
            _, validation_max, validation_norms = _metrics(validation_records)
            if monitor is not None:
                monitor.validation(validation_max, len(validation_points))
            if validation_max <= config.residual_tolerance:
                status = "numerically_converged"
                break
            if epoch >= max_validation_epochs:
                status = "stalled"
                break
            bad = validation_norms > float(config.residual_tolerance)
            violating = validation_points[bad]
            passed = validation_points[~bad]
            points = _unique_rows(points, violating, width=points.shape[1])
            guards = _unique_rows(passed, width=points.shape[1])
            excluded = _unique_rows(points, guards, width=points.shape[1])
            epoch += 1
            checks = _fresh_checks(config, 2 + epoch, excluded)
            _prepare(field, points)
            records = _evaluate_network(
                network, field, points, jacobian=False, monitor=monitor)
            objective, maximum, norms = _metrics(records)
            history.append(objective)
            iteration = 0
            damping = max(damping, 1.0e-3)
            if monitor is not None:
                monitor.record(network, objective, maximum, len(points), new_points=True)
            continue

        if iteration >= max_iterations:
            status = "stalled"
            break
        iteration += 1
        if monitor is not None:
            monitor.phase("weight_refinement")

        linearized = _evaluate_network(
            network, field, points, jacobian=True, monitor=monitor)
        _, _, old_norms = _metrics(linearized)
        point_weights = hard_point_weights_from_norms(
            old_norms, config.residual_tolerance)
        accepted = False
        local_damping = damping
        for _ in range(3):
            delta = _solve_direction(linearized, point_weights, local_damping, network)
            if not np.all(np.isfinite(delta)) or not np.any(delta):
                local_damping *= 10.0
                continue
            factor = 1.0
            for _ in range(5):
                trial = network.with_parameters(network.parameters + factor * delta)
                if gate_shrink > 0.0:
                    trial = trial.soft_threshold_structure(gate_shrink * factor)
                try:
                    trial_records = _evaluate_network(
                        trial, field, points, jacobian=False, monitor=monitor)
                    trial_objective, trial_max, trial_norms = _metrics(trial_records)
                except (ValueError, FloatingPointError, OverflowError,
                        np.linalg.LinAlgError):
                    trial_records = None
                    trial_norms = np.full_like(old_norms, np.inf)
                    trial_objective = float("inf")
                    trial_max = float("inf")
                if trial_records is not None and max_first_accept(
                    old_norms, trial_norms, config.residual_tolerance, point_weights):
                    network = trial
                    records = trial_records
                    objective = trial_objective
                    maximum = trial_max
                    norms = trial_norms
                    history.append(objective)
                    damping = max(1.0e-8, local_damping * 0.35)
                    accepted = True
                    if monitor is not None:
                        monitor.record(network, objective, maximum, len(points))
                    if progress is not None:
                        progress(len(network.response_nodes), np.sqrt(objective), maximum)
                    break
                factor *= 0.5
            if accepted:
                break
            local_damping *= 10.0
        if not accepted:
            status = "stalled"
            break

    if status == "numerically_converged":
        if monitor is not None:
            monitor.phase("structure_pruning")
        prune_points = _unique_rows(points, final_validation_points, width=points.shape[1])
        _prepare(field, prune_points)
        network = _residual_sensitivity_prune(
            network, field, prune_points, tolerance=config.residual_tolerance,
            monitor=monitor, relative_budget=prune_budget, max_rounds=prune_rounds)
        records = _evaluate_network(
            network, field, points, jacobian=False, monitor=monitor)
        objective, maximum, norms = _metrics(records)
        validation_records = _evaluate_network(
            network, field, final_validation_points, jacobian=False, monitor=monitor)
        _, validation_max, _ = _metrics(validation_records)
        if monitor is not None:
            monitor.record(network, objective, maximum, len(points))
            monitor.validation(validation_max, len(final_validation_points))
        if (maximum > config.residual_tolerance
                or validation_max > config.residual_tolerance):
            raise RuntimeError("validated structure pruning violated the residual tolerance")
    else:
        if monitor is not None:
            monitor.phase("validation")
        validation_points = _unique_rows(guards, checks, width=points.shape[1])
        _prepare(field, validation_points)
        if len(validation_points):
            validation_records = _evaluate_network(
                network, field, validation_points, jacobian=False, monitor=monitor)
            _, validation_max, _ = _metrics(validation_records)
            if monitor is not None:
                monitor.validation(validation_max, len(validation_points))

    return network, ResearchTrainingReport(
        status, len(network.response_nodes), initial_rms,
        float(np.sqrt(objective)), float(maximum), float(validation_max),
        tuple(float(value) for value in history), status == "numerically_converged")


def _install_evaluator_dispatch():
    global _ORIGINAL_STABLE, _ORIGINAL_STABLE_JAC
    from sdfmpneo.analytic import parametric_realization as pr
    import sdfmpneo.analytic as analytic_api
    from sdfmpneo.analytic import operator as operator_module
    from sdfmpneo.analytic import geometry_operator as geometry_operator_module
    from . import research as training_research
    from . import parametric_residual
    import sdfmpneo.research as research_module

    if _ORIGINAL_STABLE is not None:
        return
    _ORIGINAL_STABLE = pr.evaluate_parametric_stable
    _ORIGINAL_STABLE_JAC = pr.evaluate_parametric_stable_with_jacobians

    def stable(graph, t, *, a0, operating):
        if isinstance(graph, FixedAnalyticResponseNetwork):
            return graph.evaluate(float(t), a0=a0, operating=operating)
        return _ORIGINAL_STABLE(graph, t, a0=a0, operating=operating)

    def stable_jac(graph, t, *, a0, operating, weight_derivatives=False,
                   derivative_kind=None):
        if isinstance(graph, FixedAnalyticResponseNetwork):
            kind = derivative_kind
            if kind is None:
                kind = "parameter" if weight_derivatives else "operating"
            return graph.evaluate_stable_with_jacobians(
                float(t), a0=a0, operating=operating, derivative_kind=kind)
        try:
            return _ORIGINAL_STABLE_JAC(
                graph, t, a0=a0, operating=operating,
                weight_derivatives=weight_derivatives, derivative_kind=derivative_kind)
        except TypeError:
            return _ORIGINAL_STABLE_JAC(
                graph, t, a0=a0, operating=operating,
                weight_derivatives=weight_derivatives)

    pr.evaluate_parametric_stable = stable
    pr.evaluate_parametric_stable_with_jacobians = stable_jac
    analytic_api.evaluate_parametric_stable = stable
    operator_module.evaluate_parametric_stable = stable
    geometry_operator_module.evaluate_parametric_stable = stable
    training_research.evaluate_parametric_stable = stable
    research_module.evaluate_parametric_stable = stable
    parametric_residual.evaluate_parametric_stable_with_jacobians = stable_jac

    for module_name in (
        "sdfmpneo.certification.geometry_mass_providers",
        "sdfmpneo.certification.geometry_mass_residual_domain",
        "sdfmpneo.certification.geometry_dynamics",
    ):
        try:
            module = __import__(module_name, fromlist=["evaluate_parametric_stable"])
            if hasattr(module, "evaluate_parametric_stable"):
                module.evaluate_parametric_stable = stable
        except ImportError:
            pass


def _install_persistence(model_class):
    global _ORIGINAL_SAVE, _ORIGINAL_LOAD
    if _ORIGINAL_SAVE is not None:
        return
    _ORIGINAL_SAVE = model_class.save
    _ORIGINAL_LOAD = model_class.load.__func__

    def save(self, path):
        if not isinstance(self.graph, FixedAnalyticResponseNetwork):
            return _ORIGINAL_SAVE(self, path)
        from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph

        network = self.graph
        dummy = ParametricAnalyticEvolutionGraph(network.lambdas, network.operating_names)
        with TemporaryDirectory() as directory:
            temporary = Path(directory) / "base.npz"
            self.graph = dummy
            try:
                _ORIGINAL_SAVE(self, temporary)
            finally:
                self.graph = network
            with np.load(temporary, allow_pickle=False) as data:
                arrays = {key: np.array(data[key]) for key in data.files}
        metadata = json.loads(str(arrays["metadata"]))
        metadata["format_version"] = 3
        metadata["evolution_kind"] = FixedAnalyticResponseNetwork.kind
        metadata["fixed_network"] = network.to_metadata()
        metadata["nodes"] = []
        arrays["fixed_network_parameters"] = np.asarray(network.parameters, dtype=float)
        arrays["metadata"] = np.array(json.dumps(metadata))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("model_type") == "geometry_research":
                return _ORIGINAL_LOAD(cls, path)
            if (int(metadata.get("format_version", 1)) != 3
                    or metadata.get("evolution_kind") != FixedAnalyticResponseNetwork.kind):
                return _ORIGINAL_LOAD(cls, path)
            arrays = {key: np.array(data[key]) for key in data.files}
        compatible = dict(metadata)
        compatible["format_version"] = 1
        compatible["nodes"] = []
        compatible["operating_names"] = list(metadata["fixed_network"]["operating_names"])
        arrays["metadata"] = np.array(json.dumps(compatible))
        with TemporaryDirectory() as directory:
            temporary = Path(directory) / "compat.npz"
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays)
            model = _ORIGINAL_LOAD(cls, temporary)
        model.graph = FixedAnalyticResponseNetwork.from_metadata(
            metadata["fixed_network"], arrays["fixed_network_parameters"])
        return model

    model_class.save = save
    model_class.load = load


def install_fixed_analytic_response_network() -> None:
    """Make fixed continuous analytic-network training the fresh-model default."""
    global _ORIGINAL_TRAIN
    if _ORIGINAL_TRAIN is not None:
        return
    from . import research as training_research
    from . import adaptive_runtime
    import sdfmpneo.research as research_module

    _install_evaluator_dispatch()
    _ORIGINAL_TRAIN = training_research.train_research_graph
    training_research.train_research_graph = train_fixed_analytic_response_network
    adaptive_runtime.adaptive_train_research_graph = train_fixed_analytic_response_network
    research_module.train_research_graph = train_fixed_analytic_response_network
    _install_persistence(research_module.ResearchElectroThermalModel)


__all__ = [
    "train_fixed_analytic_response_network",
    "install_fixed_analytic_response_network",
    "_default_capacity",
    "_residual_sensitivity_prune",
]
