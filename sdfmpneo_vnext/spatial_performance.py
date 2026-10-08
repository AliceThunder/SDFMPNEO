from . import _spatial_performance_core as _core

# Export the core helpers before importing spatial_consistent_training.  That
# module imports these names back from this facade, so doing this first keeps
# the intentional facade cycle well-defined instead of exposing a partially
# initialized module without the required helpers.
resolve_torch_device = _core.resolve_torch_device
_factor_psd = _core._factor_psd
_conductor_raw_batched = _core._conductor_raw_batched
_package_raw_batched = _core._package_raw_batched
_background_raw_batched = _core._background_raw_batched
_sample_signature = _core._sample_signature
_normalized_tensor_batch = _core._normalized_tensor_batch

from ._spatial_performance_core import *

from . import spatial_consistent_training as _consistent_training
from .balanced_objectives import balanced_spatial_relative_loss

# Keep accelerated and controlled spatial trainers on the same robust objective.
_consistent_training._weighted_relative_loss = balanced_spatial_relative_loss
_batched_spatial_shape_loss = _consistent_training.consistent_batched_spatial_shape_loss

from .spatial_consistent_accelerated import (
    train_tensor_hybrid_spatial_loss_surrogate_accelerated,
)
