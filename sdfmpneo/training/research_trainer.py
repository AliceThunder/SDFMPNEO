"""Training loop for finite-horizon fixed analytic response networks."""
from __future__ import annotations

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

_MAX_EXACT_TRIALS_PER_ITERATION = 3
_TRIAL_FACTORS = (1.0, 0.5, 0.25, 0.125, 0.0625)
_DAMPING_MULTIPLIERS = (1.0, 10.0, 100.0)


def _predicted_trial_score(records, weights, delta, factor):
    """Rank trial steps using the already-built hard-point linear model."""
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
    parameter_ids,
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
    """Match the exact t=0 physical source at the center of the training box.

    A fresh high-rank network previously started with source amplitudes close to
    zero.  For a forced electrothermal system this leaves the optimizer several
    orders of magnitude away from the actual Joule-heating scale.  At the
    normalized box center every non-bias depth-one feature vanishes, so the
    first-layer bias can be set directly from the physical vector field without
    any transient labels or nonlinear solve.
    """
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


def _accept_progressive(old_norms, new_norms, tolerance, weights):
    """Filter LM trials without forcing strict max monotonicity far from target.

    When the maximum residual is thousands of times above tolerance, requiring
    every step to reduce the single worst point can reject directions that make
    the global physics fit substantially better.  Far from convergence we use a
    small trust-region filter: weighted RMS may improve while max residual moves
    by at most 2%.  Once within 20x tolerance, strict max-residual monotonicity is
    restored.  Points already below tolerance remain protected at all stages.
    """
    old = np.asarray(old_norms, dtype=float)
    new = np.asarray(new_norms, dtype=float)
    tol = float(tolerance)
    if old.shape != new.shape or np.any(~np.isfinite(new)):
        return False
    protected = old <= tol
    margin = 64.0 * np.finfo(float).eps * max(tol, 1e-30)
    if np.any(new[protected] > tol + margin):
        return False

    old_max = float(np.max(old, initial=0.0))
    new_max = float(np.max(new, initial=0.0))
    w = np.asarray(weights, dtype=float)
    old_merit = float(np.dot(w, old * old))
    new_merit = float(np.dot(w, new * new))
    numerical = 64.0 * np.finfo(float).eps * max(old_merit, 1.0)

    if old_max <= 20.0 * tol:
        if new_max < old_max - margin:
            return new_merit <= 1.02 * old_merit + numerical
        if new_max <= old_max + margin:
            return new_merit < old_merit - numerical
        return False

    # Early/global-fit regime.  Prefer reducing the worst residual, but also
    # allow a meaningful weighted-RMS improvement if the worst point remains
    # inside a narrow trust filter.
    if new_max < old_max * (1.0 - 1e-6) and new_merit <= 1.05 * old_merit + numerical:
        return True
    return (
        new_merit < old_merit * (1.0 - 1e-4) - numerical
        and new_max <= 1.02 * old_max + margin
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
    resumed = network is not None
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
    status = "stalled"
    semigroup_active = False
    final_validation = checks
    final_semigroup_validation = semigroup_checks
    # A supplied network is a resume/fine-tune operation. Its amplitudes have
    # already been trained, so never replay the amplitude-only warm start.
    amplitude_warm_start_done = resumed

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
                config.semigroup_points(validation=True, seed=102 + epoch),
                dtype=float,
            )
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

        if not amplitude_warm_start_done and not semigroup_active:
            parameter_ids = network.amplitude_parameter_indices()
        else:
            parameter_ids = None

        candidates = _rank_trial_candidates(
            linearized,
            solve_weights,
            damping,
            network,
            parameter_ids=parameter_ids,
        )
        accepted = False
        for _, local_damping, factor, delta in candidates:
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
            ):
                trial_result = None
            if trial_result is not None and _accept_progressive(
                evaluated.norms,
                trial_result.norms,
                config.residual_tolerance,
                acceptance_weights,
            ):
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

        if not amplitude_warm_start_done and not semigroup_active:
            amplitude_warm_start_done = True
            if not accepted:
                continue
        if not accepted:
            break

    evaluated = _evaluate_all(
        network, field, points, semigroup_points, monitor=monitor
    )
    if validation is None:
        validation = _evaluate_all(
            network,
            field,
            final_validation,
            final_semigroup_validation,
            monitor=monitor,
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
                monitor.validation(validation.maximum, len(final_validation))

    structure = network.structure_summary(0.0)
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
    )
    return network, report


__all__ = ["train_research_network"]
