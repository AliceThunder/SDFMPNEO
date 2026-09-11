"""Public fixed-network training API."""
from .research_config import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network
from .fixed_physics_runtime import install_physics_acceleration
from .fixed_trial_runtime import accelerated_train_research_network, install_trial_acceleration

install_physics_acceleration()
install_trial_acceleration()

from .resume_trainer import train_research_network as _train_research_network


def train_research_network(field, config, **kwargs):
    return accelerated_train_research_network(
        _train_research_network, field, config, **kwargs
    )


__all__ = ["ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network", "train_research_network"]
