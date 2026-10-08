from . import _spatial_performance_core as _core

resolve_torch_device = _core.resolve_torch_device
_factor_psd = _core._factor_psd
_conductor_raw_batched = _core._conductor_raw_batched
_package_raw_batched = _core._package_raw_batched
_background_raw_batched = _core._background_raw_batched
_sample_signature = _core._sample_signature
_normalized_tensor_batch = _core._normalized_tensor_batch

from ._spatial_performance_core import *


def _install_balanced_spatial_objective():
    # Import lazily: spatial_consistent_training itself imports the batch helpers
    # above from this facade.  Keeping this out of module initialization removes
    # the circular-import dependency while preserving one objective everywhere.
    from . import spatial_consistent_training as consistent_training
    from .balanced_objectives import balanced_spatial_relative_loss

    consistent_training._weighted_relative_loss = balanced_spatial_relative_loss
    return consistent_training


def _batched_spatial_shape_loss(*args, **kwargs):
    consistent_training = _install_balanced_spatial_objective()
    return consistent_training.consistent_batched_spatial_shape_loss(*args, **kwargs)


def train_tensor_hybrid_spatial_loss_surrogate_accelerated(*args, **kwargs):
    _install_balanced_spatial_objective()
    from .spatial_consistent_accelerated import (
        train_tensor_hybrid_spatial_loss_surrogate_accelerated as train,
    )

    return train(*args, **kwargs)
