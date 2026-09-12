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
    """Turn a fully-used accepted-step block back into a resumable optimizer phase.

    ``max_iterations_per_layer`` bounds one optimizer block so checkpoint cursors stay
    small and deterministic.  It is not a convergence criterion.  If the final
    iteration in the block was accepted, ``resume_trainer`` leaves the next-iteration
    cursor exactly at the block limit; continue from the same network/damping instead
    of reporting a false ``stalled`` result.  A genuine trust-region failure exits
    before that cursor reaches the limit and therefore remains terminal.
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

    optimizer["iteration"] = 0
    optimizer["retry"] = 0
    optimizer["backtrack"] = 0
    state["phase"] = phase
    state["layer"] = int(optimizer.get("layer", state.get("layer", 0)))
    network.training_state = state
    trial_event(
        monitor,
        event="iteration_budget",
        status="extended",
        layer=state["layer"] + 1,
        damping=float(optimizer.get("damping", 1e-3)),
        block_iterations=block,
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
    return accelerated_train_research_network(
        _train_until_converged_or_true_stall, field, config, **kwargs
    )


__all__ = ["ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network", "train_research_network"]
