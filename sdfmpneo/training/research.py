"""Public fixed-network training API."""
from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from .research_config import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network
from .fixed_physics_runtime import install_physics_acceleration
from .fixed_trial_runtime import accelerated_train_research_network, install_trial_acceleration
from .research_telemetry import trial_event

install_physics_acceleration()
install_trial_acceleration()

from .resume_trainer import train_research_network as _train_research_network


_SOFT_BUDGET_PHASES = {"physics", "restart", "validation_repair"}


def _reopen_soft_iteration_budget(network, config, monitor=None):
    """Continue an accepted iteration block until the configured total budget.

    ``max_iterations_per_layer`` is the block length.  Reaching the end of a block
    immediately after an accepted step is not a convergence/stall condition, so the
    deepest physics/restart/validation-repair stage is reopened with the same damping.
    ``max_iterations`` remains the explicit total safety budget for that stage.
    Genuine trust-region failure exits before the block-end cursor and is never
    reopened.
    """
    state = getattr(network, "training_state", None)
    if not isinstance(state, dict) or state.get("phase") != "stalled":
        return False
    optimizer = state.get("optimizer")
    if not isinstance(optimizer, dict):
        return False
    phase = str(optimizer.get("phase", ""))
    if phase not in _SOFT_BUDGET_PHASES:
        return False
    block = int(config.max_iterations_per_layer)
    if int(optimizer.get("iteration", -1)) < block:
        return False

    layer = int(optimizer.get("layer", state.get("layer", 0)))
    budget = state.get("soft_iteration_budget")
    if not isinstance(budget, dict) or (
        budget.get("phase") != phase or int(budget.get("layer", -1)) != layer
    ):
        used = 0
    else:
        used = max(0, int(budget.get("used", 0)))
    used += block
    total_limit = max(block, int(config.max_iterations))
    state["soft_iteration_budget"] = {
        "phase": phase,
        "layer": layer,
        "used": used,
        "limit": total_limit,
    }
    if used >= total_limit:
        return False

    optimizer["iteration"] = 0
    optimizer["retry"] = 0
    optimizer["backtrack"] = 0
    state["phase"] = phase
    state["layer"] = layer
    network.training_state = state
    trial_event(
        monitor,
        event="iteration_budget",
        status="extended",
        layer=layer + 1,
        damping=float(optimizer.get("damping", 1e-3)),
        block_iterations=block,
        stage_iterations_used=used,
        stage_iterations_limit=total_limit,
        total_accepted=int(state.get("total_accepted", 0)),
    )
    return True


def _train_until_converged_or_true_stall(field, config, **kwargs):
    local = dict(kwargs)
    while True:
        network, report = _train_research_network(field, config, **local)
        if not _reopen_soft_iteration_budget(
            network, config, monitor=local.get("monitor")
        ):
            return network, report
        local["network"] = network
        local["training_state"] = network.training_state


def train_research_network(field, config, **kwargs):
    network = kwargs.get("network")
    if network is not None and not isinstance(network, FixedAnalyticResponseNetwork):
        raise TypeError("only FixedAnalyticResponseNetwork checkpoints are supported")
    if network is not None:
        # Checkpoints written by the previous implementation may already carry
        # ``phase='stalled'`` solely because an accepted block ended exactly at
        # max_iterations_per_layer.  Reopen that cursor before resume_trainer sees
        # the terminal marker, preserving the saved damping/history instead of
        # resetting a perfectly healthy optimization trajectory.
        _reopen_soft_iteration_budget(
            network, config, monitor=kwargs.get("monitor")
        )
    return accelerated_train_research_network(
        _train_until_converged_or_true_stall, field, config, **kwargs
    )


__all__ = ["ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network", "train_research_network"]
