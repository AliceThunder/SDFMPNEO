from . import _tensor_spatial_neural_core as _core
from .balanced_objectives import balanced_spatial_end_to_end_error
from .spatial_boundary_features import (
    BoundaryAwareHybridSpatialLossShapeNet,
    BoundaryAwarePreparedHybridSpatialLossField,
    boundary_aware_conductor_coordinates,
)


# Tensor spatial schema 2 adds explicit superellipse boundary coordinates to the
# conductor decoder.  Keep this scoped to tensor artifacts so legacy hybrid
# spatial artifacts retain their original four-coordinate representation.
_core.HybridSpatialLossShapeNet = BoundaryAwareHybridSpatialLossShapeNet
_core.PreparedHybridSpatialLossField = BoundaryAwarePreparedHybridSpatialLossField
_core._coordinate_features = boundary_aware_conductor_coordinates
_core.TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA = 2

TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA = _core.TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA
TensorHybridSpatialLossArtifact = _core.TensorHybridSpatialLossArtifact
tensor_port_fingerprint = _core.tensor_port_fingerprint
_sample_end_to_end_error = balanced_spatial_end_to_end_error

from ._tensor_spatial_neural_core import *


def train_tensor_hybrid_spatial_loss_surrogate(
    port_artifact,
    samples,
    *,
    validation_samples=(),
    field_hidden_dim: int = 64,
    factor_rank: int = 4,
    depth: int = 2,
    epochs: int = 120,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    patience: int = 20,
    validation_interval: int = 1,
    min_improvement: float = 1e-5,
    seed: int = 47,
    conductor_longitudinal_points: int = 12,
    conductor_radial_order: int = 3,
    conductor_angular_order: int = 16,
    package_axial_order: int = 6,
    package_radial_order: int = 4,
    package_azimuthal_order: int = 16,
    background_segments_per_turn: int = 16,
    background_radial_order: int = 12,
    background_angular_order: int = 48,
    batch_size: int = 1,
    device: str = "cpu",
):
    from .spatial_performance import (
        train_tensor_hybrid_spatial_loss_surrogate_accelerated,
    )

    return train_tensor_hybrid_spatial_loss_surrogate_accelerated(
        port_artifact,
        samples,
        validation_samples=validation_samples,
        field_hidden_dim=field_hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        validation_interval=validation_interval,
        min_improvement=min_improvement,
        seed=seed,
        conductor_longitudinal_points=conductor_longitudinal_points,
        conductor_radial_order=conductor_radial_order,
        conductor_angular_order=conductor_angular_order,
        package_axial_order=package_axial_order,
        package_radial_order=package_radial_order,
        package_azimuthal_order=package_azimuthal_order,
        background_segments_per_turn=background_segments_per_turn,
        background_radial_order=background_radial_order,
        background_angular_order=background_angular_order,
        batch_size=batch_size,
        device=device,
    )
