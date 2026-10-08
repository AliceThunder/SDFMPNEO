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
from .spatial_boundary_features import (
    BoundaryAwareHybridSpatialLossShapeNet,
    boundary_aware_conductor_coordinates,
)
from .tensor_neural import _sample_tensor_ranges


# Tensor spatial training uses the schema-2 conductor representation without
# altering the legacy hybrid spatial classes used elsewhere in the package.
_spatial_core.HybridSpatialLossShapeNet = BoundaryAwareHybridSpatialLossShapeNet
_spatial_objective._coordinate_features = boundary_aware_conductor_coordinates


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
_spatial_objective.SPATIAL_TRAINING_CONTRACT = SPATIAL_TRAINING_CONTRACT
_spatial_core.SPATIAL_TRAINING_CONTRACT = SPATIAL_TRAINING_CONTRACT
_spatial_core._sample_end_to_end_error = balanced_spatial_end_to_end_error


def train_port_controlled(samples, **kwargs):
    """Train/resume the port model with a domain covering the full dataset.

    Model fitting and validation remain strictly split inside the core trainer.
    The material domain, however, is inference metadata rather than a learned
    target.  It must describe every scene admitted by this training dataset so
    that validation/spatial samples are not falsely rejected as out-of-domain.
    Recomputing it here also repairs already-completed checkpoints created by
    older code without retraining their weights.
    """
    samples = tuple(samples)
    artifact, history = _port_core.train_port_controlled(samples, **kwargs)
    artifact.material_domain = _sample_tensor_ranges(samples)
    return artifact, history


train_spatial_controlled = _spatial_core.train_spatial_controlled

__all__ = [
    "train_port_controlled",
    "train_spatial_controlled",
]
