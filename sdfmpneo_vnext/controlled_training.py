from . import _controlled_training_core as _port_core
from . import controlled_spatial_consistent as _spatial_core
from . import spatial_consistent_training as _spatial_objective
from .balanced_objectives import (
    PORT_TRAINING_CONTRACT,
    SPATIAL_TRAINING_CONTRACT,
    balanced_port_batch_loss,
    balanced_spatial_end_to_end_error,
    balanced_spatial_relative_loss,
)


# The controlled trainers resolve these module globals at call time.  Rebinding
# the objective functions keeps checkpointing/resume/device behavior unchanged
# while versioning the numerical training contract explicitly.
_legacy_port_signature = _port_core._port_signature


def _balanced_port_signature(config, cache_key: str) -> str:
    return _port_core._hash(
        {
            "legacy_signature": _legacy_port_signature(config, cache_key),
            "port_training_contract": PORT_TRAINING_CONTRACT,
        }
    )


_port_core._batch_loss = balanced_port_batch_loss
_port_core._port_signature = _balanced_port_signature

_spatial_objective._weighted_relative_loss = balanced_spatial_relative_loss
_spatial_core.SPATIAL_TRAINING_CONTRACT = SPATIAL_TRAINING_CONTRACT
_spatial_core._sample_end_to_end_error = balanced_spatial_end_to_end_error

train_port_controlled = _port_core.train_port_controlled
train_spatial_controlled = _spatial_core.train_spatial_controlled

__all__ = [
    "train_port_controlled",
    "train_spatial_controlled",
]
