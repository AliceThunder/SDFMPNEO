"""Training loop for finite-horizon fixed analytic response networks."""
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
    _prune,
    _solve_direction,
    _unique_rows,
)
from .research_linearization import linearization as _linearization
from .research_telemetry import trial_event

_MAX_EXACT_TRIALS_PER_ITERATION = 3
_TRIAL_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
_DAMPING_MULTIPLIERS = (1.0, 10.0, 100.0)


def _predicted_trial_score(records, weights, delta, factor):
    """Rank trial steps using the already-built representative linear model."""
    values = []
    for record in records:
        predicted = np.asarray(record.residual, dtype=float) + float(factor) * (
            np.asarray(record.parameter_jacobian, dtype=float) @ delta
        )
        values.append(float(np.linalg.norm(predicted)))
    norms = np.asarray(values, dtype=float)
    if norms.size == 0 or np.any(~np.isfinite(norms)):
        return (float("inf"), float("inf"))
    w = np.asarray(weights, dtype=float)
    return float(np.max(norms)), float(np.dot(w, norms * norms))


def _rank_trial_candidates(
    linearized,
    solve_weights,
    damping,
    network,
    *,
    parameter_ids=None,
):
    """Generate many cheap LM candidates but return only a few exact trials."""
    candidates = []
    for multiplier in _DAMPING_MULTIPLIERS:
        local_damping = max(float(damping) * multiplier, 1e-12)
        delta = _solve_direction(
            linearized.records,
            solve_weights,
            local_damping,
            network,
            parameter_indices=parameter_ids,
        )
        if not np.all(np.isfinite(delta)) or not np.any(delta):
            continue
        for factor in _TRIAL_FACTORS:
            score = _predicted_trial_score(
                linearized.records, solve_weights, delta, factor
            )
            if np.all(np.isfinite(score)):
                candidates.append((score, local_damping, factor, delta))
    candidates.sort(key=lambda item: item[0])
    return candidates[:_MAX_EXACT_TRIALS_PER_ITERATION]


def _exact_trial_result(trial, field, points, semigroup_points, semigroup_active, monitor):
    if semigroup_active:
        return _evaluate_all(trial, field, points, semigroup_points, monitor=monitor)
    physics = _evaluate_network(trial, field, points, monitor=monitor)
    return _combined_metrics(physics, [])


def _initialize_center_source(network, field, config):
    """Match the exact t=0 physical source at the center of the training box."""
    if int(network.depth) != 1:
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
    gates = np.asarray(network._array("channel_gate")[0], dtype=float)
    targets = np.asarray(network.targets, dtype=int)
    gate_sum = np.bincount(targets, weights=gates, minlength=network.n_modes)
    if np.any(np.abs(gate_sum) < 1e-14):
        return network
    bias = source[targets] / gate_sum[targets]
    theta = network.parameters.copy()
    theta[network._indices("bias_0")] = bias
    return network.with_parameters(theta)


def _accept_progressive_reason(old_norms, new_norms, tolerance, weights):
    """Return ``(accepted, reason)`` for the progressive global-fit filter."""
    old = np.asarray(old_norms, dtype=float)
    new = np.asarray(new_norms, dtype=float)
    tol = float(tolerance)
    if old.shape != new.shape:
        return False, "shape_mismatch"
    if np.any(~np.isfinite(new)):
        return False, "nonfinite_residual"
    protected = old <= tol
    margin = 64.0 * np.finfo(float).eps * max(tol, 1e-30)
    if np.any(new[protected] > tol + margin):
        return False, "regressed_converged_point"

    old_max = float(np.max(old, initial=0.0))
    new_max = float(np.max(new, initial=0.0))
    w = np.asarray(weights, dtype=float)
    old_merit = float(np.dot(w, old * old))
    new_merit = float(np.dot(w, new * new))
    numerical = 64.0 * np.finfo(float).eps * max(old_merit, 1.0)

    if old_max <= 20.0 * tol:
        if new_max < old_max - margin:
            if new_merit <= 1.02 * old_merit + numerical:
                return True, "near_target_max_reduced"
            return False, "near_target_merit_regressed"
        if new_max <= old_max + margin:
            if new_merit < old_merit - numerical:
                return True, "near_target_merit_reduced"
            return False, "near_target_no_progress"
        return False, "near_target_max_regressed"

    if new_max < old_max * (1.0 - 1e-6):
        if new_merit <= 1.05 * old_merit + numerical:
            return True, "max_reduced"
        return False, "max_reduced_but_merit_regressed"
    if new_max > 1.02 * old_max + margin:
        return False, "max_filter_exceeded"
    if new_merit < old_merit * (1.0 - 1e-4) - numerical:
        return True, "global_merit_reduced"
    return False, "insufficient_global_progress"


def _accept_progressive(old_norms, new_norms, tolerance, weights):
    return _accept_progressive_reason(old_norms, new_norms, tolerance, weights)[0]


def _validation_proxy(evaluated):
    """Typed placeholder for a report when independent validation was skipped."""
    return SimpleNamespace(
        maximum=float(evaluated.maximum),
        physics_max=float(evaluated.physics_max),
        semigroup_max=float(evaluated.semigroup_max),
    )


def train_research_network(
    field,
    config: ResearchTrainingConfig,
    *,
    network=None,
    operating_names=None,
    progress=None,
    monitor=None,
):
    if network is None:
        network = make_fixed_network(field, config, operating_names=operating_names)
        network = _initialize_center_source(network, field, config)
    if not isinstance(network, FixedAnalyticResponseNetwork):
        raise TypeError("only FixedAnalyticResponseNetwork checkpoints are supported")
    if len(config.initial_lower) != network.n_modes or len(config.operating_lower) != len(network.operating_names):
        raise ValueError("training domain does not match the fixed analytic network")
    if network.max_response_time != config.max_response_time:
        raise ValueError("training max_response_time does not match the saved network")

    points = np.asarray(config.points(validation=False), dtype=float)
    checks = np.asarray(config.points(validation=True), dtype=float)
    semigroup_points = np.asarray(config.semigroup_points(validation=False), dtype=float)
    semigroup_checks = np.asarray(config.semigroup_points(validation=True), dtype=float)
    guards = np.empty((0, points.shape[1]), dtype=float)
    sg_width = len(config.initial_lower) + len(config.operating_lower) + 2
    sg_guards = np.empty((0, sg_width), dtype=float)

    if monitor is not None:
        monitor.retain(network)
        monitor.phase("initial_residual")
    physics_records = _evaluate_network(
        network, field, points, monitor=monitor, work_label="initial_residual"
    )
    evaluated = _combined_metrics(physics_records, [])
    initial_rms = float(np.sqrt(evaluated.objective))
    history = [evaluated.objective]
    if monitor is not None:
        monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

    damping, epoch, iteration = 1e-3, 0, 0
    validation = None
    validation_performed = False
    status = "stalled"
    semigroup_active = False
    final_validation = checks
    final_semigroup_validation = semigroup_checks

    while True:
        if not semigroup_active and evaluated.physics_max <= config.residual_tolerance:
            semigroup_active = True
            if monitor is not None:
                monitor.phase("restart_consistency")
            evaluated = _evaluate_all(
                network, field, points, semigroup_points, monitor=monitor
            )
            history.append(evaluated.objective)
            iteration = 0
            damping = max(damping, 1e-3)
            if monitor is not None:
                monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

        if semigroup_active and evaluated.maximum <= config.residual_tolerance:
            if monitor is not None:
                monitor.phase("validation")
            final_validation = _unique_rows(guards, checks, width=points.shape[1])
            final_semigroup_validation = _unique_rows(
                sg_guards, semigroup_checks, width=sg_width
            )
            validation = _evaluate_all(
                network,
                field,
                final_validation,
                final_semigroup_validation,
                monitor=monitor,
            )
            validation_performed = True
            if monitor is not None:
                monitor.validation(validation.maximum, len(final_validation))
            if validation.maximum <= config.residual_tolerance:
                status = "numerically_converged"
                break
            if epoch >= config.max_validation_epochs:
                break

            physics_records = _evaluate_network(
                network, field, final_validation, monitor=monitor
            )
            _, _, physics_norms = _metrics(physics_records)
            semigroup_records = _evaluate_semigroup(
                network, final_semigroup_validation, monitor=monitor
            )
            _, _, sg_norms = _metrics(semigroup_records)
            violating = final_validation[physics_norms > config.residual_tolerance]
            violating_sg = final_semigroup_validation[
                sg_norms > config.residual_tolerance
            ]
            guards = _unique_rows(
                guards,
                final_validation[physics_norms <= config.residual_tolerance],
                width=points.shape[1],
            )
            sg_guards = _unique_rows(
                sg_guards,
                final_semigroup_validation[sg_norms <= config.residual_tolerance],
                width=sg_width,
            )
            points = _unique_rows(points, violating, width=points.shape[1])
            semigroup_points = _unique_rows(
                semigroup_points, violating_sg, width=sg_width
            )
            epoch += 1
            checks = np.asarray(
                config.points(validation=True, seed=2 + epoch), dtype=float
            )
            semigroup_checks = np.asarray(
                config.semigroup_points(validation=True, seed=102 + epoch), dtype=float
            )
            validation = None
            validation_performed = False
            evaluated = _evaluate_all(
                network, field, points, semigroup_points, monitor=monitor
            )
            history.append(evaluated.objective)
            iteration = 0
            damping = max(damping, 1e-3)
            if monitor is not None:
                monitor.record(
                    network,
                    evaluated.objective,
                    evaluated.maximum,
                    len(points),
                    new_points=True,
                )
            continue

        if iteration >= config.max_iterations:
            break
        iteration += 1
        if monitor is not None:
            monitor.phase(
                "restart_consistency" if semigroup_active else "weight_refinement"
            )
        linearized = _linearization(
            network,
            field,
            points,
            semigroup_points if semigroup_active else semigroup_points[:0],
            evaluated,
            config,
            monitor,
        )
        solve_weights = _hard_weights(linearized.norms, config.residual_tolerance)
        acceptance_weights = _hard_weights(
            evaluated.norms, config.residual_tolerance
        )
        candidates = _rank_trial_candidates(
            linearized,
            solve_weights,
            damping,
            network,
            parameter_ids=None,
        )
        if not candidates:
            trial_event(
                monitor,
                event="iteration",
                iteration=int(iteration),
                status="rejected",
                reason="no_finite_candidate",
                jacobian_points=int(len(getattr(linearized, "physics_subset", ()))),
                jacobian_hard_points=int(getattr(linearized, "physics_hard_count", 0)),
            )
            break

        accepted = False
        weight_sum = max(float(np.sum(solve_weights)), np.finfo(float).tiny)
        for candidate_index, (score, local_damping, factor, delta) in enumerate(candidates, start=1):
            predicted_max, predicted_merit = score
            predicted_weighted_rms = float(np.sqrt(predicted_merit / weight_sum))
            trial_event(
                monitor,
                event="candidate",
                iteration=int(iteration),
                candidate=int(candidate_index),
                status="evaluating",
                damping=float(local_damping),
                factor=float(factor),
                predicted_max=float(predicted_max),
                predicted_weighted_rms=predicted_weighted_rms,
                jacobian_points=int(len(getattr(linearized, "physics_subset", ()))),
                jacobian_hard_points=int(getattr(linearized, "physics_hard_count", 0)),
            )
            trial = network.with_parameters(network.parameters + factor * delta)
            if config.gate_shrink:
                trial = trial.soft_threshold_structure(config.gate_shrink * factor)
            try:
                trial_result = _exact_trial_result(
                    trial,
                    field,
                    points,
                    semigroup_points,
                    semigroup_active,
                    monitor,
                )
            except (
                ValueError,
                FloatingPointError,
                OverflowError,
                np.linalg.LinAlgError,
            ) as exc:
                trial_event(
                    monitor,
                    event="candidate",
                    iteration=int(iteration),
                    candidate=int(candidate_index),
                    status="error",
                    reason=type(exc).__name__,
                    damping=float(local_damping),
                    factor=float(factor),
                    predicted_max=float(predicted_max),
                    predicted_weighted_rms=predicted_weighted_rms,
                )
                continue

            accepted_now, reason = _accept_progressive_reason(
                evaluated.norms,
                trial_result.norms,
                config.residual_tolerance,
                acceptance_weights,
            )
            trial_event(
                monitor,
                event="candidate",
                iteration=int(iteration),
                candidate=int(candidate_index),
                status="accepted" if accepted_now else "rejected",
                reason=reason,
                damping=float(local_damping),
                factor=float(factor),
                predicted_max=float(predicted_max),
                predicted_weighted_rms=predicted_weighted_rms,
                actual_rms=float(np.sqrt(trial_result.objective)),
                actual_max=float(trial_result.maximum),
            )
            if accepted_now:
                network = trial
                evaluated = trial_result
                history.append(evaluated.objective)
                damping = max(local_damping / 3.0, 1e-12)
                accepted = True
                if monitor is not None:
                    monitor.record(
                        network,
                        evaluated.objective,
                        evaluated.maximum,
                        len(points),
                    )
                if progress is not None:
                    progress(
                        iteration,
                        float(np.sqrt(evaluated.objective)),
                        evaluated.maximum,
                    )
                break

        if not accepted:
            break

    # ``evaluated`` is already the exact residual of the current network.  Do
    # not spend another full training sweep merely to reproduce the same value.
    # Independent validation is meaningful only after the training/restart
    # criteria have actually reached their target; otherwise mark it skipped.
    if validation is None:
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

    if status == "numerically_converged" and config.prune_rounds > 0:
        if monitor is not None:
            monitor.phase("structure_pruning")
        verify = _unique_rows(points, final_validation, width=points.shape[1])
        verify_sg = _unique_rows(
            semigroup_points, final_semigroup_validation, width=sg_width
        )
        pruned = _prune(
            network,
            field,
            verify,
            verify_sg,
            config.residual_tolerance,
            config.prune_relative_budget,
            config.prune_rounds,
            monitor,
            config,
        )
        if pruned is not network:
            network = pruned
            evaluated = _evaluate_all(
                network, field, points, semigroup_points, monitor=monitor
            )
            validation = _evaluate_all(
                network,
                field,
                final_validation,
                final_semigroup_validation,
                monitor=monitor,
            )
            validation_performed = True
            if (
                evaluated.maximum > config.residual_tolerance
                or validation.maximum > config.residual_tolerance
            ):
                raise RuntimeError(
                    "validated pruning violated residual/restart tolerance"
                )
            if monitor is not None:
                monitor.record(
                    network,
                    evaluated.objective,
                    evaluated.maximum,
                    len(points),
                )
                monitor.validation(
                    validation.maximum, len(final_validation)
                )

    structure = dict(network.structure_summary(0.0))
    structure["validation_performed"] = bool(validation_performed)
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
    )
    return network, report


__all__ = [
    "train_research_network",
    "_initialize_center_source",
    "_accept_progressive",
    "_accept_progressive_reason",
    "_rank_trial_candidates",
]
