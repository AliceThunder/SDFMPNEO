from __future__ import annotations

from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "accelerated tensor spatial training requires the 'neural' extra"
    ) from exc

from ._spatial_performance_core import (
    _sample_signature,
    resolve_torch_device,
)
from .hybrid_spatial_neural import (
    HybridSpatialLossShapeNet,
    HybridSpatialTrainingReport,
)
from .spatial_consistent_training import (
    clear_spatial_consistency_caches,
    consistent_batched_spatial_shape_loss,
    resolve_spatial_normalization,
)
from .tensor_spatial_neural import (
    TensorHybridSpatialLossArtifact,
    _sample_end_to_end_error,
)
from .tensor_spatial_training_data import TensorHybridSpatialTeacherSample


def train_tensor_hybrid_spatial_loss_surrogate_accelerated(
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
    batch_size: int = 8,
    device: str = "auto",
):
    samples = tuple(samples)
    validation_samples = tuple(validation_samples)
    if not samples:
        raise ValueError("at least one tensor spatial training sample is required")
    if any(
        not isinstance(sample, TensorHybridSpatialTeacherSample)
        for sample in samples + validation_samples
    ):
        raise TypeError(
            "accelerated spatial trainer requires TensorHybridSpatialTeacherSample instances"
        )
    if (
        epochs < 1
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or patience < 1
        or validation_interval < 1
        or min_improvement < 0.0
        or batch_size < 1
    ):
        raise ValueError("invalid accelerated tensor spatial training configuration")

    normalization = resolve_spatial_normalization(
        {
            "conductor_longitudinal_points": conductor_longitudinal_points,
            "conductor_radial_order": conductor_radial_order,
            "conductor_angular_order": conductor_angular_order,
            "package_axial_order": package_axial_order,
            "package_radial_order": package_radial_order,
            "package_azimuthal_order": package_azimuthal_order,
            "background_segments_per_turn": background_segments_per_turn,
            "background_radial_order": background_radial_order,
            "background_angular_order": background_angular_order,
        }
    )
    clear_spatial_consistency_caches()
    resolved_device = resolve_torch_device(device)
    port_model = port_artifact.model.to(resolved_device)
    port_artifact.device = resolved_device
    port_model.eval()
    for parameter in port_model.parameters():
        parameter.requires_grad_(False)
    for sample in samples + validation_samples:
        port_artifact.predict_structured(sample.scene, sample.frequency_hz)

    torch.manual_seed(int(seed))
    if resolved_device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    model = HybridSpatialLossShapeNet(
        port_model.hidden_dim,
        port_model.coil_pair_dim,
        port_model.cross_dim,
        field_hidden_dim=field_hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
    ).to(device=resolved_device, dtype=next(port_model.parameters()).dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    def buckets_for(values):
        buckets = defaultdict(list)
        for index, sample in enumerate(values):
            buckets[_sample_signature(sample)].append(index)
        return buckets

    training_buckets = buckets_for(samples)
    validation_buckets = buckets_for(validation_samples)

    def iter_batches(values, buckets, *, shuffle):
        keys = list(buckets)
        if shuffle:
            rng.shuffle(keys)
        for key in keys:
            order = np.asarray(buckets[key], dtype=int)
            if shuffle:
                order = rng.permutation(order)
            for start in range(0, len(order), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                yield tuple(values[int(index)] for index in indices)

    best_state = None
    best_epoch = 0
    best_shape_error = None
    best_end_to_end_error = None
    stale = 0
    final_loss = float("inf")
    stopped_early = False
    epochs_run = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch in iter_batches(samples, training_buckets, shuffle=True):
            optimizer.zero_grad(set_to_none=True)
            loss = consistent_batched_spatial_shape_loss(
                model,
                port_artifact,
                batch,
                device=resolved_device,
                normalization=normalization,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch)
            seen += len(batch)
        final_loss = total / max(seen, 1)
        epochs_run = epoch

        if validation_samples:
            if epoch % int(validation_interval) != 0 and epoch != int(epochs):
                continue
            model.eval()
            shape_total = 0.0
            shape_count = 0
            with torch.no_grad():
                for batch in iter_batches(
                    validation_samples,
                    validation_buckets,
                    shuffle=False,
                ):
                    shape = consistent_batched_spatial_shape_loss(
                        model,
                        port_artifact,
                        batch,
                        device=resolved_device,
                        normalization=normalization,
                    )
                    shape_total += float(shape.detach().cpu()) * len(batch)
                    shape_count += len(batch)
            shape_score = shape_total / max(shape_count, 1)
            probe = TensorHybridSpatialLossArtifact(
                port_artifact,
                model,
                **normalization,
                device=resolved_device,
            )
            end_to_end_score = float(
                np.mean(
                    [
                        _sample_end_to_end_error(probe, sample)
                        for sample in validation_samples
                    ]
                )
            )
            if (
                best_end_to_end_error is None
                or end_to_end_score
                < best_end_to_end_error - float(min_improvement)
            ):
                best_shape_error = shape_score
                best_end_to_end_error = end_to_end_score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= int(patience):
                    stopped_early = True
                    break
        else:
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("accelerated tensor spatial training produced no selectable state")
    model.load_state_dict(best_state)
    model.eval()
    artifact = TensorHybridSpatialLossArtifact(
        port_artifact,
        model,
        **normalization,
        device=resolved_device,
    )
    return artifact, HybridSpatialTrainingReport(
        float(final_loss),
        int(epochs_run),
        int(best_epoch),
        None if best_end_to_end_error is None else float(best_end_to_end_error),
        bool(stopped_early),
        None if best_shape_error is None else float(best_shape_error),
    )
