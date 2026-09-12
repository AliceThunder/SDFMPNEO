"""Accepted-step checkpoint/resume trainer for the fixed analytic response network.

The saved state is intentionally small: it stores optimizer/control progress and the
current deterministic collocation sets, but never serializes a Jacobian or trial
candidate.  A cooperative stop therefore resumes from the last safe model state;
at worst the candidate that was in flight is recomputed.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from .research_config import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network
from .research_helpers import (
    _evaluate_all,
    _evaluate_network,
    _evaluate_semigroup,
    _hard_weights,
    _metrics,
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
from . import research_trainer as rt

_STATE_VERSION = 1
_TERMINAL_PHASES = {"completed", "stalled"}


def _config_signature(config: ResearchTrainingConfig) -> str:
    return json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _store_rows(state, key, rows):
    state[key] = np.asarray(rows, dtype=float).tolist()


def _load_rows(state, key, default, width):
    if key not in state:
        _store_rows(state, key, default)
        return np.asarray(default, dtype=float)
    rows = np.asarray(state[key], dtype=float)
    if rows.ndim != 2 or rows.shape[1] != int(width) or np.any(~np.isfinite(rows)):
        raise ValueError(f"saved training state contains invalid {key}")
    return rows


def _source_report_from_state(state):
    saved = state.get("source_prefit_report")
    if not isinstance(saved, dict):
        return None
    return SimpleNamespace(
        rms=float(saved["rms"]),
        maximum=float(saved["maximum"]),
        sample_count=int(saved["sample_count"]),
    )


def _set_optimizer(state, phase, layer, iteration, damping, retry=0, backtrack=0):
    state["phase"] = str(phase)
    state["layer"] = int(layer)
    state["optimizer"] = {
        "phase": str(phase),
        "layer": int(layer),
        "iteration": int(iteration),
        "damping": float(damping),
        "retry": int(retry),
        "backtrack": int(backtrack),
    }


def _optimizer_cursor(state, phase, layer):
    opt = state.get("optimizer")
    if not isinstance(opt, dict):
        return 0, 1e-3, 0, 0
    if opt.get("phase") != phase or int(opt.get("layer", -1)) != int(layer):
        return 0, 1e-3, 0, 0
    return (
        max(0, int(opt.get("iteration", 0))),
        max(1e-12, float(opt.get("damping", 1e-3))),
        max(0, int(opt.get("retry", 0))),
        max(0, int(opt.get("backtrack", 0))),
    )


def _sync_common(state, history, total_accepted, trained_depth):
    state["objective_history"] = [float(v) for v in history]
    state["total_accepted"] = int(total_accepted)
    state["trained_depth"] = int(trained_depth)


def _initial_state(state, config, *, new_network, network):
    signature = _config_signature(config)
    resumable = (
        isinstance(state, dict)
        and int(state.get("version", -1)) == _STATE_VERSION
        and state.get("phase") not in _TERMINAL_PHASES
    )
    if resumable:
        if state.get("config_signature") != signature:
            raise ValueError(
                "unfinished training checkpoint must resume with the exact saved training config"
            )
        return True

    state.clear()
    state.update(
        version=_STATE_VERSION,
        config_signature=signature,
        phase="source_prefit" if new_network else "initial_residual",
        source_prefit_done=not new_network,
        source_prefit_report=None,
        initial_rms_residual=None,
        objective_history=[],
        total_accepted=0,
        trained_depth=int(network.structure_summary(0.0).get("effective_depth", 0)),
        layer=0 if new_network else rt._effective_layer(network),
        optimizer=None,
        semigroup_active=False,
        validation_epoch=0,
        validation_performed=False,
        prune_next_round=0,
    )
    return False


def _resume_prune(
    network,
    field,
    points,
    semigroup_points,
    checks,
    semigroup_checks,
    tolerance,
    rounds,
    monitor,
    state,
):
    current = network
    training = _evaluate_all(current, field, points, semigroup_points, monitor=monitor)
    validation = _evaluate_all(current, field, checks, semigroup_checks, monitor=monitor)
    start_round = max(0, int(state.get("prune_next_round", 0)))
    for round_index in range(start_round, int(rounds)):
        state["phase"] = "pruning"
        state["prune_next_round"] = int(round_index)
        if monitor is not None:
            monitor.phase("structure_pruning")
        gates = current._array("channel_gate")
        candidates = []
        for layer in range(current.depth - 1, -1, -1):
            for channel in range(current.layer_widths[layer]):
                if gates[layer, channel] == 0.0:
                    continue
                strength = current._channel_amplitude_strength(layer, channel)
                candidates.append((strength, layer, channel))
        if not candidates:
            state["prune_next_round"] = int(rounds)
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
                trial, field, checks, semigroup_checks, monitor=monitor
            )
            if trial_validation.maximum > tolerance:
                continue
            current, training, validation = trial, trial_training, trial_validation
            current.training_state = state
            accepted = True
            state["prune_next_round"] = int(round_index + 1)
            if monitor is not None:
                monitor.record(current, training.objective, training.maximum, len(points))
                monitor.validation(validation.maximum, len(checks))
            break
        if not accepted:
            state["prune_next_round"] = int(rounds)
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
    training_state=None,
):
    """Train or resume exactly from the last accepted/safe network boundary."""
    new_network = network is None
    if network is None:
        network = make_fixed_network(field, config, operating_names=operating_names)
    if isinstance(training_state, dict):
        state = training_state
    else:
        saved_state = getattr(network, "training_state", None)
        state = saved_state if isinstance(saved_state, dict) else {}
    network.training_state = state
    if not isinstance(network, FixedAnalyticResponseNetwork):
        raise TypeError("only FixedAnalyticResponseNetwork checkpoints are supported")
    if (
        len(config.initial_lower) != network.n_modes
        or len(config.operating_lower) != len(network.operating_names)
    ):
        raise ValueError("training domain does not match the fixed analytic network")
    if network.max_response_time != config.max_response_time:
        raise ValueError("training max_response_time does not match the saved network")

    _initial_state(state, config, new_network=new_network, network=network)
    point_width = network.n_modes + len(network.operating_names) + 1
    semigroup_width = network.n_modes + len(network.operating_names) + 2
    points = _load_rows(
        state, "points", config.points(validation=False), point_width
    )
    checks = _load_rows(
        state, "validation_points", config.points(validation=True), point_width
    )
    semigroup_points = _load_rows(
        state, "semigroup_points", config.semigroup_points(validation=False), semigroup_width
    )
    semigroup_checks = _load_rows(
        state, "validation_semigroup", config.semigroup_points(validation=True), semigroup_width
    )

    source_report = _source_report_from_state(state)
    if monitor is not None:
        monitor.retain(network)

    if not bool(state.get("source_prefit_done", False)):
        state["phase"] = "source_prefit"
        if monitor is not None:
            monitor.phase("source_prefit")
        network, source_report = source_prefit(network, field, config, monitor)
        network.training_state = state
        state["source_prefit_done"] = True
        state["source_prefit_report"] = {
            "rms": float(source_report.rms),
            "maximum": float(source_report.maximum),
            "sample_count": int(source_report.sample_count),
        }
        state["phase"] = "initial_residual"
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

    phase = str(state.get("phase", "initial_residual"))
    if phase == "initial_residual":
        if monitor is not None:
            monitor.phase("initial_residual")
        evaluated = rt._exact_result(
            network, field, points, semigroup_points, False, monitor
        )
        initial_rms = float(np.sqrt(evaluated.objective))
        history = [float(evaluated.objective)]
        total_accepted = int(state.get("total_accepted", 0))
        trained_depth = max(
            1 if source_report is not None else 0,
            int(state.get("trained_depth", 0)),
            int(network.structure_summary(0.0).get("effective_depth", 0)),
        )
        state["initial_rms_residual"] = initial_rms
        _sync_common(state, history, total_accepted, trained_depth)
        start_layer = 0 if new_network or state.get("source_prefit_report") else rt._effective_layer(network)
        _set_optimizer(state, "physics", start_layer, 0, 1e-3)
        if monitor is not None:
            monitor.record(network, evaluated.objective, evaluated.maximum, len(points))
    else:
        initial_rms = float(
            state.get("initial_rms_residual")
            if state.get("initial_rms_residual") is not None
            else 0.0
        )
        history = [float(v) for v in state.get("objective_history", [])]
        total_accepted = int(state.get("total_accepted", 0))
        trained_depth = max(
            int(state.get("trained_depth", 0)),
            int(network.structure_summary(0.0).get("effective_depth", 0)),
        )
        if phase == "physics":
            evaluated = rt._exact_result(
                network, field, points, semigroup_points, False, monitor
            )
        else:
            evaluated = rt._exact_result(
                network, field, points, semigroup_points,
                phase not in {"restart_pending"}, monitor,
            )

    status = "stalled"
    validation = None
    validation_performed = bool(state.get("validation_performed", False))

    def optimize_layer(layer, include_semigroup, phase_name):
        nonlocal network, evaluated, total_accepted, trained_depth, history
        start_iteration, damping, start_retry, start_backtrack = _optimizer_cursor(
            state, phase_name, layer
        )
        accepted_any = False
        for local_iteration in range(start_iteration, int(config.max_iterations_per_layer)):
            target_met = (
                evaluated.maximum <= config.residual_tolerance
                if include_semigroup
                else evaluated.physics_max <= config.residual_tolerance
            )
            if target_met:
                break
            retry0 = start_retry if local_iteration == start_iteration else 0
            backtrack0 = start_backtrack if local_iteration == start_iteration else 0
            _set_optimizer(
                state, phase_name, layer, local_iteration, damping, retry0, backtrack0
            )
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

            for retry in range(retry0, int(config.max_damping_retries)):
                bt_start = backtrack0 if retry == retry0 else 0
                _set_optimizer(
                    state, phase_name, layer, local_iteration, damping, retry, bt_start
                )
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
                    _set_optimizer(
                        state, phase_name, layer, local_iteration,
                        damping, retry + 1, 0
                    )
                    continue

                for backtrack in range(bt_start, int(config.max_backtracks)):
                    _set_optimizer(
                        state, phase_name, layer, local_iteration,
                        damping, retry, backtrack
                    )
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
                        trial_result = rt._exact_result(
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
                        _set_optimizer(
                            state, phase_name, layer, local_iteration,
                            damping, retry, backtrack + 1
                        )
                        continue

                    accepted_now, reason = rt._accept_progressive_reason(
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
                        _set_optimizer(
                            state, phase_name, layer, local_iteration,
                            damping, retry, backtrack + 1
                        )
                        continue

                    network = trial
                    network.training_state = state
                    evaluated = trial_result
                    history.append(float(evaluated.objective))
                    accepted = accepted_any = True
                    total_accepted += 1
                    trained_depth = max(trained_depth, layer + 1)
                    damping = max(
                        damping / 3.0 if factor >= 0.5 else damping * 2.0,
                        1e-12,
                    )
                    _sync_common(state, history, total_accepted, trained_depth)
                    _set_optimizer(
                        state, phase_name, layer, local_iteration + 1,
                        damping, 0, 0
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
                _set_optimizer(
                    state, phase_name, layer, local_iteration,
                    damping, retry + 1, 0
                )
                trial_event(
                    monitor, event="trust_region", layer=layer + 1,
                    iteration=local_iteration + 1, retry=retry + 1,
                    status="contracted",
                    reason="all_physical_backtracks_rejected",
                    damping=float(damping),
                )
            if not accepted:
                break
            start_retry = 0
            start_backtrack = 0
        return accepted_any

    phase = str(state.get("phase"))
    if phase == "physics":
        start_layer = max(0, min(network.depth - 1, int(state.get("layer", 0))))
        for layer in range(start_layer, network.depth):
            if evaluated.physics_max <= config.residual_tolerance:
                break
            if layer != int(state.get("layer", layer)):
                _set_optimizer(state, "physics", layer, 0, 1e-3)
            if layer > 0 and network._layer_is_zero(layer):
                _set_optimizer(state, "physics", layer, 0, 1e-3)
                targets, energy = residual_target_modes(
                    evaluated.records, network.layer_widths[layer]
                )
                network = network.zero_layer_amplitudes(layer)
                network = network.with_layer_targets(layer, targets)
                network.training_state = state
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
                evaluated = rt._exact_result(
                    network, field, points, semigroup_points, False, monitor
                )
            optimize_layer(layer, False, "physics")

        if evaluated.physics_max <= config.residual_tolerance:
            state["semigroup_active"] = True
            _set_optimizer(
                state, "restart_pending", rt._effective_layer(network), 0, 1e-3
            )
        else:
            state["phase"] = "stalled"

    phase = str(state.get("phase"))
    if phase == "restart_pending":
        if monitor is not None:
            monitor.phase("restart_consistency")
        evaluated = rt._exact_result(
            network, field, points, semigroup_points, True, monitor
        )
        history.append(float(evaluated.objective))
        _sync_common(state, history, total_accepted, trained_depth)
        _set_optimizer(
            state, "restart", rt._effective_layer(network), 0, 1e-3
        )
        if monitor is not None:
            monitor.record(network, evaluated.objective, evaluated.maximum, len(points))

    phase = str(state.get("phase"))
    if phase == "restart":
        if evaluated.maximum > config.residual_tolerance:
            optimize_layer(rt._effective_layer(network), True, "restart")
        if evaluated.maximum <= config.residual_tolerance:
            state["phase"] = "validation"
            state.setdefault("validation_epoch", 0)
            state["validation_performed"] = bool(validation_performed)
            state["optimizer"] = None
        else:
            state["phase"] = "stalled"

    while state.get("phase") in {"validation", "validation_repair"}:
        phase = str(state["phase"])
        epoch = max(0, int(state.get("validation_epoch", 0)))
        if phase == "validation_repair":
            if evaluated.maximum > config.residual_tolerance:
                optimize_layer(
                    rt._effective_layer(network), True, "validation_repair"
                )
            if evaluated.maximum > config.residual_tolerance:
                state["phase"] = "stalled"
                break
            state["phase"] = "validation"
            state["validation_epoch"] = epoch + 1
            state["optimizer"] = None
            continue

        if epoch > int(config.max_validation_epochs):
            state["phase"] = "stalled"
            break
        if monitor is not None:
            monitor.phase("validation")
        validation = _evaluate_all(
            network, field, checks, semigroup_checks, monitor=monitor
        )
        validation_performed = True
        state["validation_performed"] = True
        if monitor is not None:
            monitor.validation(validation.maximum, len(checks))
        if validation.maximum <= config.residual_tolerance:
            status = "numerically_converged"
            state["phase"] = "pruning"
            state.setdefault("prune_next_round", 0)
            break
        if epoch >= int(config.max_validation_epochs):
            state["phase"] = "stalled"
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
        _store_rows(state, "points", points)
        _store_rows(state, "semigroup_points", semigroup_points)
        _store_rows(state, "validation_points", checks)
        _store_rows(state, "validation_semigroup", semigroup_checks)
        state["validation_epoch"] = epoch
        _set_optimizer(
            state, "validation_repair", rt._effective_layer(network), 0, 1e-3
        )
        evaluated = rt._exact_result(
            network, field, points, semigroup_points, True, monitor
        )

    if state.get("phase") == "pruning":
        status = "numerically_converged"
        if config.prune_rounds > 0:
            network, evaluated, validation = _resume_prune(
                network,
                field,
                points,
                semigroup_points,
                checks,
                semigroup_checks,
                config.residual_tolerance,
                config.prune_rounds,
                monitor,
                state,
            )
        elif validation is None:
            validation = _evaluate_all(
                network, field, checks, semigroup_checks, monitor=monitor
            )
            validation_performed = True
        state["phase"] = "completed"
        state["optimizer"] = None

    if state.get("phase") == "stalled":
        status = "stalled"
        if validation is None:
            validation = rt._validation_proxy(evaluated)
            if not validation_performed:
                trial_event(
                    monitor,
                    event="validation",
                    status="skipped",
                    reason="training_residual_above_tolerance",
                    actual_rms=float(np.sqrt(evaluated.objective)),
                    actual_max=float(evaluated.maximum),
                )

    if validation is None:
        validation = rt._validation_proxy(evaluated)

    _sync_common(state, history, total_accepted, trained_depth)
    state["validation_performed"] = bool(validation_performed)

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
        initial_rms_residual=float(initial_rms),
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
    if status == "numerically_converged":
        state["phase"] = "completed"
    else:
        state["phase"] = "stalled"
    return network, report


__all__ = ["train_research_network"]
