"""Training loop for finite-horizon fixed analytic response networks."""
from __future__ import annotations

import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from .research_config import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network
from .research_helpers import (
    _accept, _combined_metrics, _evaluate_all, _evaluate_network, _evaluate_semigroup,
    _hard_weights, _linearization, _metrics, _prune, _solve_direction, _unique_rows,
)

def train_research_network(field, config: ResearchTrainingConfig, *, network=None, operating_names=None, progress=None, monitor=None):
    if network is None:
        network = make_fixed_network(field, config, operating_names=operating_names)
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
        monitor.retain(network); monitor.phase("initial_residual")
    physics_records = _evaluate_network(network, field, points, monitor=monitor, work_label="initial_residual")
    evaluated = _combined_metrics(physics_records, [])
    initial_rms = float(np.sqrt(evaluated.objective)); history = [evaluated.objective]
    if monitor is not None: monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

    damping, epoch, iteration = 1e-3, 0, 0
    validation = None; status = "stalled"; semigroup_active = False
    final_validation = checks; final_semigroup_validation = semigroup_checks

    while True:
        if not semigroup_active and evaluated.physics_max <= config.residual_tolerance:
            semigroup_active = True
            if monitor is not None: monitor.phase("restart_consistency")
            evaluated = _evaluate_all(network, field, points, semigroup_points, monitor=monitor)
            history.append(evaluated.objective); iteration = 0; damping = max(damping, 1e-3)
            if monitor is not None: monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

        if semigroup_active and evaluated.maximum <= config.residual_tolerance:
            if monitor is not None: monitor.phase("validation")
            final_validation = _unique_rows(guards, checks, width=points.shape[1])
            final_semigroup_validation = _unique_rows(sg_guards, semigroup_checks, width=sg_width)
            validation = _evaluate_all(network, field, final_validation, final_semigroup_validation, monitor=monitor)
            if monitor is not None: monitor.validation(validation.maximum, len(final_validation))
            if validation.maximum <= config.residual_tolerance:
                status = "numerically_converged"; break
            if epoch >= config.max_validation_epochs: break

            physics_records = _evaluate_network(network, field, final_validation, monitor=monitor)
            _, _, physics_norms = _metrics(physics_records)
            semigroup_records = _evaluate_semigroup(network, final_semigroup_validation, monitor=monitor)
            _, _, sg_norms = _metrics(semigroup_records)
            violating = final_validation[physics_norms > config.residual_tolerance]
            violating_sg = final_semigroup_validation[sg_norms > config.residual_tolerance]
            guards = _unique_rows(guards, final_validation[physics_norms <= config.residual_tolerance], width=points.shape[1])
            sg_guards = _unique_rows(sg_guards, final_semigroup_validation[sg_norms <= config.residual_tolerance], width=sg_width)
            points = _unique_rows(points, violating, width=points.shape[1])
            semigroup_points = _unique_rows(semigroup_points, violating_sg, width=sg_width)
            epoch += 1
            checks = np.asarray(config.points(validation=True, seed=2 + epoch), dtype=float)
            semigroup_checks = np.asarray(config.semigroup_points(validation=True, seed=102 + epoch), dtype=float)
            evaluated = _evaluate_all(network, field, points, semigroup_points, monitor=monitor)
            history.append(evaluated.objective); iteration = 0; damping = max(damping, 1e-3)
            if monitor is not None:
                monitor.record(network, evaluated.objective, evaluated.maximum, len(points), new_points=True)
            continue

        if iteration >= config.max_iterations: break
        iteration += 1
        if monitor is not None:
            monitor.phase("restart_consistency" if semigroup_active else "weight_refinement")
        linearized = _linearization(
            network, field, points, semigroup_points if semigroup_active else semigroup_points[:0],
            evaluated, config, monitor,
        )
        solve_weights = _hard_weights(linearized.norms, config.residual_tolerance)
        acceptance_weights = _hard_weights(evaluated.norms, config.residual_tolerance)
        accepted = False; local_damping = damping
        amplitude_ids = network.amplitude_parameter_indices()
        parameter_blocks = (amplitude_ids, None)
        for _ in range(3):
            for parameter_ids in parameter_blocks:
                delta = _solve_direction(
                    linearized.records, solve_weights, local_damping, network,
                    parameter_indices=parameter_ids,
                )
                if not np.all(np.isfinite(delta)) or not np.any(delta):
                    continue
                factor = 1.0
                for _ in range(5):
                    trial = network.with_parameters(network.parameters + factor * delta)
                    if config.gate_shrink:
                        trial = trial.soft_threshold_structure(config.gate_shrink * factor)
                    try:
                        if semigroup_active:
                            trial_result = _evaluate_all(trial, field, points, semigroup_points, monitor=monitor)
                        else:
                            trial_physics = _evaluate_network(trial, field, points, monitor=monitor)
                            trial_result = _combined_metrics(trial_physics, [])
                    except (ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError):
                        trial_result = None
                    if trial_result is not None and _accept(
                        evaluated.norms, trial_result.norms, config.residual_tolerance, acceptance_weights
                    ):
                        network = trial; evaluated = trial_result; history.append(evaluated.objective)
                        damping = max(local_damping / 3.0, 1e-12); accepted = True
                        if monitor is not None:
                            monitor.record(network, evaluated.objective, evaluated.maximum, len(points))
                        if progress is not None:
                            progress(iteration, float(np.sqrt(evaluated.objective)), evaluated.maximum)
                        break
                    factor *= 0.5
                if accepted:
                    break
            if accepted:
                break
            local_damping *= 10.0
        if not accepted: break

    evaluated = _evaluate_all(network, field, points, semigroup_points, monitor=monitor)
    if validation is None:
        validation = _evaluate_all(network, field, final_validation, final_semigroup_validation, monitor=monitor)

    if status == "numerically_converged" and config.prune_rounds > 0:
        if monitor is not None: monitor.phase("structure_pruning")
        verify = _unique_rows(points, final_validation, width=points.shape[1])
        verify_sg = _unique_rows(semigroup_points, final_semigroup_validation, width=sg_width)
        pruned = _prune(
            network, field, verify, verify_sg, config.residual_tolerance,
            config.prune_relative_budget, config.prune_rounds, monitor, config,
        )
        if pruned is not network:
            network = pruned
            evaluated = _evaluate_all(network, field, points, semigroup_points, monitor=monitor)
            validation = _evaluate_all(network, field, final_validation, final_semigroup_validation, monitor=monitor)
            if evaluated.maximum > config.residual_tolerance or validation.maximum > config.residual_tolerance:
                raise RuntimeError("validated pruning violated residual/restart tolerance")
            if monitor is not None:
                monitor.record(network, evaluated.objective, evaluated.maximum, len(points))
                monitor.validation(validation.maximum, len(final_validation))

    structure = network.structure_summary(0.0); horizon = float(config.max_response_time)
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
