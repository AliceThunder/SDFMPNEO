"""Training loop for finite-horizon multilayer analytic response networks."""
from __future__ import annotations

from types import SimpleNamespace
import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from .research_config import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network
from .research_helpers import (
    _combined_metrics,
    _evaluate_all,
    _evaluate_network,
    _evaluate_semigroup,
    _hard_weights,
    _metrics,
    _physics_vector_field,
    _unique_rows,
)
from .research_multilayer import (
    combined_layer_linearization,
    layer_linearization,
    predicted_layer_metrics,
    residual_target_modes,
    solve_layer_direction,
    source_prefit,
)
from .research_telemetry import trial_event


def _initialize_center_source(network, field, config):
    """Backward-compatible one-point source initializer used by unit tests/tools."""
    if network.channels_per_mode != 1 or network.layer_widths[0] != network.n_modes:
        return network
    a0 = 0.5 * (
        np.asarray(config.initial_lower, dtype=float)
        + np.asarray(config.initial_upper, dtype=float)
    )
    operating = 0.5 * (
        np.asarray(config.operating_lower, dtype=float)
        + np.asarray(config.operating_upper, dtype=float)
    )
    physical = _physics_vector_field(field, a0, operating)
    source = np.asarray(physical, dtype=float) + np.asarray(network.lambdas) * a0
    theta = network.parameters.copy()
    theta[network._indices("bias_0")] = source[network.layer_targets[0]]
    return network.with_parameters(theta)


def _accept_progressive_reason(old_norms, new_norms, tolerance, weights):
    """Physical trust filter without turning temporarily-good points into constraints."""
    old = np.asarray(old_norms, dtype=float)
    new = np.asarray(new_norms, dtype=float)
    tol = float(tolerance)
    if old.shape != new.shape:
        return False, "shape_mismatch"
    if np.any(~np.isfinite(new)):
        return False, "nonfinite_residual"
    old_max = float(np.max(old, initial=0.0))
    new_max = float(np.max(new, initial=0.0))
    w = np.asarray(weights, dtype=float)
    old_merit = float(np.dot(w, old * old))
    new_merit = float(np.dot(w, new * new))
    numerical = 64.0 * np.finfo(float).eps * max(old_merit, 1.0)
    margin = 64.0 * np.finfo(float).eps * max(tol, old_max, 1e-30)

    if old_max <= 20.0 * tol:
        if new_max < old_max - margin and new_merit <= 1.02 * old_merit + numerical:
            return True, "near_target_max_reduced"
        if new_max <= old_max + margin and new_merit < old_merit - numerical:
            return True, "near_target_merit_reduced"
        return False, "near_target_no_progress"

    if new_max < old_max * (1.0 - 1e-6):
        if new_merit <= 1.05 * old_merit + numerical:
            return True, "max_reduced"
        return False, "max_reduced_but_merit_regressed"
    if new_max > 1.10 * old_max + margin:
        return False, "max_filter_exceeded"
    if new_merit < old_merit * (1.0 - 1e-4) - numerical:
        return True, "global_merit_reduced"
    return False, "insufficient_global_progress"


def _accept_progressive(old_norms, new_norms, tolerance, weights):
    return _accept_progressive_reason(old_norms, new_norms, tolerance, weights)[0]


def _validation_proxy(evaluated):
    return SimpleNamespace(
        maximum=float(evaluated.maximum),
        physics_max=float(evaluated.physics_max),
        semigroup_max=float(evaluated.semigroup_max),
    )


def _exact_result(network, field, points, semigroup_points, include_semigroup, monitor):
    if include_semigroup:
        return _evaluate_all(network, field, points, semigroup_points, monitor=monitor)
    physics = _evaluate_network(network, field, points, monitor=monitor)
    return _combined_metrics(physics, [])


def _effective_layer(network):
    depth = int(network.structure_summary(0.0).get("effective_depth", 0))
    return max(0, min(network.depth - 1, depth - 1))


def _verified_channel_prune(
    network,
    field,
    points,
    semigroup_points,
    validation_points,
    validation_semigroup,
    tolerance,
    rounds,
    monitor,
):
    """Post-training pruning only; every removed response neuron is verified."""
    current = network
    training = _evaluate_all(current, field, points, semigroup_points, monitor=monitor)
    validation = _evaluate_all(
        current, field, validation_points, validation_semigroup, monitor=monitor
    )
    for _ in range(int(rounds)):
        gates = current._array("channel_gate")
        candidates = []
        for layer in range(current.depth - 1, -1, -1):
            for channel in range(current.layer_widths[layer]):
                if gates[layer, channel] == 0.0:
                    continue
                strength = current._channel_amplitude_strength(layer, channel)
                candidates.append((strength, layer, channel))
        if not candidates:
            break
        candidates.sort()
        accepted = False
        for _, layer, channel in candidates[:max(1, min(8, len(candidates)))]:
            theta = current.parameters.copy()
            pid = int(current._indices("channel_gate")[layer, channel])
            theta[pid] = 0.0
            trial = current.with_parameters(theta)
            trial_training = _evaluate_all(
                trial, field, points, semigroup_points, monitor=monitor
            )
            if trial_training.maximum > tolerance:
                continue
            trial_validation = _evaluate_all(
                trial, field, validation_points, validation_semigroup, monitor=monitor
            )
            if trial_validation.maximum > tolerance:
                continue
            current, training, validation = trial, trial_training, trial_validation
            accepted = True
            break
        if not accepted:
            break
    return current, training, validation


def train_research_network(
    field,
    config: ResearchTrainingConfig,
    *,
    network=None,
    operating_names=None,
    progress=None,
    monitor=None,
):
    fresh = network is None
    if network is None:
        network = make_fixed_network(field, config, operating_names=operating_names)
    if not isinstance(network, FixedAnalyticResponseNetwork):
        raise TypeError("only FixedAnalyticResponseNetwork checkpoints are supported")
    if (
        len(config.initial_lower) != network.n_modes
        or len(config.operating_lower) != len(network.operating_names)
    ):
        raise ValueError("training domain does not match the fixed analytic network")
    if network.max_response_time != config.max_response_time:
        raise ValueError("training max_response_time does not match the saved network")

    points = np.asarray(config.points(validation=False), dtype=float)
    checks = np.asarray(config.points(validation=True), dtype=float)
    semigroup_points = np.asarray(config.semigroup_points(validation=False), dtype=float)
    semigroup_checks = np.asarray(config.semigroup_points(validation=True), dtype=float)
    source_report = None

    if monitor is not None:
        monitor.retain(network)
    if fresh:
        if monitor is not None:
            monitor.phase("source_prefit")
        network, source_report = source_prefit(network, field, config, monitor)
        if monitor is not None:
            monitor.retain(network)
        trial_event(
            monitor,
            event="source_prefit",
            status="completed",
            actual_rms=float(source_report.rms),
            actual_max=float(source_report.maximum),
            sample_count=int(source_report.sample_count),
        )

    if monitor is not None:
        monitor.phase("initial_residual")
    evaluated = _exact_result(network, field, points, semigroup_points, False, monitor)
    initial_rms = float(np.sqrt(evaluated.objective))
    history = [evaluated.objective]
    if monitor is not None:
        monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

    total_accepted = 0
    trained_depth = max(
        1 if source_report is not None else 0,
        int(network.structure_summary(0.0).get("effective_depth", 0)),
    )
    status = "stalled"
    semigroup_active = False
    validation = None
    validation_performed = False

    def optimize_layer(layer, include_semigroup):
        nonlocal network, evaluated, total_accepted, trained_depth
        damping = 1e-3
        accepted_any = False
        for local_iteration in range(int(config.max_iterations_per_layer)):
            target_met = (
                evaluated.maximum <= config.residual_tolerance
                if include_semigroup
                else evaluated.physics_max <= config.residual_tolerance
            )
            if target_met:
                break
            if monitor is not None:
                monitor.phase(
                    "restart_consistency" if include_semigroup
                    else f"response_layer_{layer + 1}"
                )
            linearized = (
                combined_layer_linearization(
                    network, field, points, semigroup_points, evaluated,
                    layer, config, monitor
                )
                if include_semigroup
                else layer_linearization(
                    network, field, points, evaluated, layer, config, monitor
                )
            )
            ids = np.asarray(linearized.parameter_indices, dtype=int)
            if ids.size == 0:
                trial_event(
                    monitor, event="iteration", layer=layer + 1,
                    iteration=local_iteration + 1, status="rejected",
                    reason="empty_amplitude_block",
                )
                break
            solve_weights = _hard_weights(linearized.norms, config.residual_tolerance)
            acceptance_weights = _hard_weights(evaluated.norms, config.residual_tolerance)
            accepted = False

            for retry in range(int(config.max_damping_retries)):
                delta = solve_layer_direction(
                    linearized.records, solve_weights, damping, network, ids
                )
                if not np.all(np.isfinite(delta)) or not np.any(delta[ids]):
                    trial_event(
                        monitor, event="direction", layer=layer + 1,
                        iteration=local_iteration + 1, retry=retry + 1,
                        status="rejected", reason="zero_or_nonfinite_direction",
                        damping=float(damping),
                    )
                    damping *= 10.0
                    continue

                for backtrack in range(int(config.max_backtracks)):
                    factor = float(config.backtrack_factor) ** backtrack
                    predicted_max, predicted_wrms, _ = predicted_layer_metrics(
                        linearized.records, solve_weights, delta, ids, factor
                    )
                    trial_event(
                        monitor, event="candidate", layer=layer + 1,
                        iteration=local_iteration + 1, retry=retry + 1,
                        backtrack=backtrack, status="evaluating",
                        damping=float(damping), factor=float(factor),
                        predicted_max=float(predicted_max),
                        predicted_weighted_rms=float(predicted_wrms),
                    )
                    trial = network.with_parameters(network.parameters + factor * delta)
                    try:
                        trial_result = _exact_result(
                            trial, field, points, semigroup_points,
                            include_semigroup, monitor
                        )
                    except (
                        ValueError, FloatingPointError, OverflowError,
                        np.linalg.LinAlgError,
                    ) as exc:
                        trial_event(
                            monitor, event="candidate", layer=layer + 1,
                            iteration=local_iteration + 1, retry=retry + 1,
                            backtrack=backtrack, status="error",
                            reason=type(exc).__name__, damping=float(damping),
                            factor=float(factor),
                        )
                        continue

                    accepted_now, reason = _accept_progressive_reason(
                        evaluated.norms,
                        trial_result.norms,
                        config.residual_tolerance,
                        acceptance_weights,
                    )
                    if include_semigroup:
                        physics_guard = max(
                            1.05 * config.residual_tolerance,
                            1.02 * evaluated.physics_max,
                        )
                        if trial_result.physics_max > physics_guard:
                            accepted_now = False
                            reason = "physics_guard_exceeded"
                    trial_event(
                        monitor, event="candidate", layer=layer + 1,
                        iteration=local_iteration + 1, retry=retry + 1,
                        backtrack=backtrack,
                        status="accepted" if accepted_now else "rejected",
                        reason=reason, damping=float(damping), factor=float(factor),
                        predicted_max=float(predicted_max),
                        predicted_weighted_rms=float(predicted_wrms),
                        actual_rms=float(np.sqrt(trial_result.objective)),
                        actual_max=float(trial_result.maximum),
                        actual_physics_max=float(trial_result.physics_max),
                        actual_restart_max=float(trial_result.semigroup_max),
                    )
                    if not accepted_now:
                        continue

                    network = trial
                    evaluated = trial_result
                    history.append(evaluated.objective)
                    accepted = accepted_any = True
                    total_accepted += 1
                    trained_depth = max(trained_depth, layer + 1)
                    damping = max(
                        damping / 3.0 if factor >= 0.5 else damping * 2.0,
                        1e-12,
                    )
                    if monitor is not None:
                        monitor.record(
                            network, evaluated.objective, evaluated.maximum, len(points)
                        )
                    if progress is not None:
                        progress(
                            total_accepted,
                            float(np.sqrt(evaluated.objective)),
                            evaluated.maximum,
                        )
                    break
                if accepted:
                    break
                damping *= 10.0
                trial_event(
                    monitor, event="trust_region", layer=layer + 1,
                    iteration=local_iteration + 1, retry=retry + 1,
                    status="contracted",
                    reason="all_physical_backtracks_rejected",
                    damping=float(damping),
                )
            if not accepted:
                break
        return accepted_any

    # Fresh training starts from Layer 1. Resume continues from the deepest
    # already-active response corrector so previously learned deeper layers are
    # never cleared or silently retargeted.
    start_layer = 0 if fresh else _effective_layer(network)
    for layer in range(start_layer, network.depth):
        if evaluated.physics_max <= config.residual_tolerance:
            break
        if layer > start_layer or (fresh and layer > 0):
            if network._layer_is_zero(layer):
                targets, energy = residual_target_modes(
                    evaluated.records, network.layer_widths[layer]
                )
                network = network.zero_layer_amplitudes(layer)
                network = network.with_layer_targets(layer, targets)
                if monitor is not None:
                    monitor.retain(network)
                trial_event(
                    monitor,
                    event="layer_activation",
                    layer=layer + 1,
                    status="activated",
                    target_modes=targets.tolist(),
                    captured_modal_residual_fraction=float(
                        np.sum(energy[targets])
                        / max(np.sum(energy), np.finfo(float).tiny)
                    ),
                )
                evaluated = _exact_result(
                    network, field, points, semigroup_points, False, monitor
                )
        optimize_layer(layer, False)

    # Restart consistency uses the deepest active layer as a bounded corrector.
    # The second segment always starts from the first segment's network output,
    # so restart states are reachable states rather than arbitrary box corners.
    if evaluated.physics_max <= config.residual_tolerance:
        semigroup_active = True
        if monitor is not None:
            monitor.phase("restart_consistency")
        evaluated = _exact_result(network, field, points, semigroup_points, True, monitor)
        history.append(evaluated.objective)
        if monitor is not None:
            monitor.record(network, evaluated.objective, evaluated.maximum, len(points))
        if evaluated.maximum > config.residual_tolerance:
            optimize_layer(_effective_layer(network), True)

    # Independent validation is paid only after both training criteria pass.
    if semigroup_active and evaluated.maximum <= config.residual_tolerance:
        for epoch in range(int(config.max_validation_epochs) + 1):
            if monitor is not None:
                monitor.phase("validation")
            validation = _evaluate_all(
                network, field, checks, semigroup_checks, monitor=monitor
            )
            validation_performed = True
            if monitor is not None:
                monitor.validation(validation.maximum, len(checks))
            if validation.maximum <= config.residual_tolerance:
                status = "numerically_converged"
                break
            if epoch >= config.max_validation_epochs:
                break

            physics_records = _evaluate_network(network, field, checks, monitor=monitor)
            _, _, physics_norms = _metrics(physics_records)
            sg_records = _evaluate_semigroup(network, semigroup_checks, monitor=monitor)
            _, _, sg_norms = _metrics(sg_records)
            violating = checks[physics_norms > config.residual_tolerance]
            violating_sg = semigroup_checks[sg_norms > config.residual_tolerance]
            points = _unique_rows(points, violating, width=points.shape[1])
            semigroup_points = _unique_rows(
                semigroup_points, violating_sg, width=semigroup_points.shape[1]
            )
            checks = np.asarray(
                config.points(validation=True, seed=2 + epoch), dtype=float
            )
            semigroup_checks = np.asarray(
                config.semigroup_points(validation=True, seed=102 + epoch), dtype=float
            )
            evaluated = _exact_result(
                network, field, points, semigroup_points, True, monitor
            )
            optimize_layer(_effective_layer(network), True)
            if evaluated.maximum > config.residual_tolerance:
                break
    else:
        validation = _validation_proxy(evaluated)
        validation_performed = False
        trial_event(
            monitor,
            event="validation",
            status="skipped",
            reason="training_residual_above_tolerance",
            actual_rms=float(np.sqrt(evaluated.objective)),
            actual_max=float(evaluated.maximum),
        )

    if (
        status == "numerically_converged"
        and validation_performed
        and config.prune_rounds > 0
    ):
        if monitor is not None:
            monitor.phase("structure_pruning")
        network, evaluated, validation = _verified_channel_prune(
            network,
            field,
            points,
            semigroup_points,
            checks,
            semigroup_checks,
            config.residual_tolerance,
            config.prune_rounds,
            monitor,
        )
        if monitor is not None:
            monitor.record(network, evaluated.objective, evaluated.maximum, len(points))
            monitor.validation(validation.maximum, len(checks))

    if validation is None:
        validation = _validation_proxy(evaluated)
        validation_performed = False

    structure = dict(network.structure_summary(0.0))
    structure.update(
        validation_performed=bool(validation_performed),
        initial_training_rank=int(min(config.initial_training_rank, network.n_modes)),
        initial_training_modes=config.initial_active_indices().tolist(),
        gates_frozen_during_training=True,
    )
    horizon = float(config.max_response_time)
    report = ResearchTrainingReport(
        status=status,
        active_response_channels=int(structure["active_response_channels"]),
        max_response_time=horizon,
        initial_rms_residual=initial_rms,
        final_rms_residual=float(np.sqrt(evaluated.objective)),
        maximum_training_residual=float(evaluated.maximum),
        maximum_validation_residual=float(validation.maximum),
        maximum_training_physics_residual=float(evaluated.physics_max),
        maximum_validation_physics_residual=float(validation.physics_max),
        maximum_training_semigroup_rate_defect=float(evaluated.semigroup_max),
        maximum_validation_semigroup_rate_defect=float(validation.semigroup_max),
        maximum_training_semigroup_defect=float(evaluated.semigroup_max * horizon),
        maximum_validation_semigroup_defect=float(validation.semigroup_max * horizon),
        objective_history=tuple(float(v) for v in history),
        numerical_tolerance_met=(status == "numerically_converged"),
        structure=structure,
        validation_performed=bool(validation_performed),
        source_prefit_rms_residual=(
            None if source_report is None else float(source_report.rms)
        ),
        source_prefit_max_residual=(
            None if source_report is None else float(source_report.maximum)
        ),
        trained_response_depth=int(max(trained_depth, structure["effective_depth"])),
    )
    return network, report


__all__ = [
    "train_research_network", "_initialize_center_source",
    "_accept_progressive", "_accept_progressive_reason",
]
