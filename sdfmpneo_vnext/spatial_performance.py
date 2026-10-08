from . import _spatial_performance_core as _core

resolve_torch_device = _core.resolve_torch_device
resolve_training_dtype = _core.resolve_training_dtype
_factor_psd = _core._factor_psd
_conductor_raw_batched = _core._conductor_raw_batched
_package_raw_batched = _core._package_raw_batched
_background_raw_batched = _core._background_raw_batched
_sample_signature = _core._sample_signature
_normalized_tensor_batch = _core._normalized_tensor_batch

from ._spatial_performance_core import *
from .spatial_consistent_training import (
    consistent_batched_spatial_shape_loss as _batched_spatial_shape_loss,
)
from .spatial_consistent_accelerated import (
    train_tensor_hybrid_spatial_loss_surrogate_accelerated,
)
