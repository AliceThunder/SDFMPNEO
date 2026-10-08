from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.spatial_performance requires the 'neural' extra"
    ) from exc

from .hybrid_background_spatial import (
    background_coordinate_features,
    background_loss_gate,
)
from .hybrid_spatial_neural import (
    HybridSpatialLossShapeNet,
    HybridSpatialTrainingReport,
    _apply_by_coil,
    _apply_transform,
    _conductor_transforms,
    _coordinate_features,
    _environment_transform,
    _package_coordinate_features,
    _package_loss_gate,
    _weighted_relative_loss,
)
from .performance import (
    _batched_latent,
    resolve_torch_device,
)
from .tensor_spatial_neural import (
    TensorHybridSpatialLossArtifact,
    _sample_end_to_end_error,
)
from .tensor_spatial_training_data import TensorHybridSpatialTeacherSample


def _factor_psd(raw, factor_rank: int):
    real = raw[..., :factor_rank]
    imag = raw[..., factor_rank:]
    complex_dtype = torch.complex64 if raw.dtype == torch.float32 else torch.complex128
    factors = real.to(complex_dtype) + 1j * imag.to(complex_dtype)
    matrices = torch.einsum("qpr,qsr->qps", factors.conj(), factors)
    n = matrices.shape[-1]
    trace_scale = torch.clamp(
        torch.real(torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1)),
        min=1e-12,
    )
    eye = torch.eye(n, dtype=complex_dtype, device=raw.device)
    matrices = matrices + (
        1e-9 * trace_scale[:, None, None] / max(n, 1)
    ) * eye[None]
    return 0.5 * (matrices + matrices.conj().transpose(-1, -2))


def _conductor_raw_batched(
    shape_net,
    coil_latent,
    coil_pair,
    batch_index,
    coil_index,
    coordinates,
):
    batch_index = torch.as_tensor(batch_index, dtype=torch.long, device=coil_latent.device)
    coil_index = torch.as_tensor(coil_index, dtype=torch.long, device=coil_latent.device)
    coordinates = torch.as_tensor(
        coordinates,
        dtype=coil_latent.dtype,
        device=coil_latent.device,
    )
    n_points = int(batch_index.numel())
    n_ports = int(coil_latent.shape[1])
    source = coil_latent[batch_index, coil_index]
    source_rows = source[:, None, :].expand(n_points, n_ports, -1)
    port_rows = coil_latent[batch_index]
    pair_rows = coil_pair[batch_index, coil_index]
    coordinate_rows = coordinates[:, None, :].expand(n_points, n_ports, -1)
    raw = shape_net.head(
        torch.cat((source_rows, port_rows, pair_rows, coordinate_rows), dim=-1)
    )
    return _factor_psd(raw, shape_net.factor_rank)


def _package_raw_batched(
    shape_net,
    coil_latent,
    package_latent,
    coil_package,
    batch_index,
    package_index,
    coordinates,
):
    batch_index = torch.as_tensor(batch_index, dtype=torch.long, device=coil_latent.device)
    package_index = torch.as_tensor(package_index, dtype=torch.long, device=coil_latent.device)
    coordinates = torch.as_tensor(
        coordinates,
        dtype=coil_latent.dtype,
        device=coil_latent.device,
    )
    n_points = int(batch_index.numel())
    n_ports = int(coil_latent.shape[1])
    selected_package = package_latent[batch_index, package_index]
    package_rows = selected_package[:, None, :].expand(n_points, n_ports, -1)
    coil_rows = coil_latent[batch_index]
    selected_cross = coil_package[batch_index, :, package_index, :]
    coordinate_rows = coordinates[:, None, :].expand(n_points, n_ports, -1)
    raw = shape_net.head(
        torch.cat((package_rows, coil_rows, selected_cross, coordinate_rows), dim=-1)
    )
    return _factor_psd(raw, shape_net.factor_rank)


def _background_raw_batched(
    shape_net,
    coil_latent,
    package_latent,
    batch_index,
    coil_coordinates,
    package_coordinates,
):
    batch_index = torch.as_tensor(batch_index, dtype=torch.long, device=coil_latent.device)
    coil_coordinates = torch.as_tensor(
        coil_coordinates,
        dtype=coil_latent.dtype,
        device=coil_latent.device,
    )
    package_coordinates = torch.as_tensor(
        package_coordinates,
        dtype=coil_latent.dtype,
        device=coil_latent.device,
    )
    n_points = int(batch_index.numel())
    n_ports = int(coil_latent.shape[1])
    n_packages = int(package_latent.shape[1])
    selected_coils = coil_latent[batch_index]
    selected_packages = package_latent[batch_index]
    if n_packages:
        package_inputs = torch.cat((selected_packages, package_coordinates), dim=-1)
        package_messages = shape_net.package_message(package_inputs)
        package_context = package_messages.sum(dim=1) / math.sqrt(n_packages)
    else:
        package_context = torch.zeros(
            (n_points, shape_net.hidden_dim),
            dtype=coil_latent.dtype,
            device=coil_latent.device,
        )
    context_rows = package_context[:, None, :].expand(n_points, n_ports, -1)
    raw = shape_net.head(
        torch.cat((selected_coils, context_rows, coil_coordinates), dim=-1)
    )
    real = raw[..., : shape_net.factor_rank]
    imag = raw[..., shape_net.factor_rank :]
    complex_dtype = torch.complex64 if raw.dtype == torch.float32 else torch.complex128
    factors = real.to(complex_dtype) + 1j * imag.to(complex_dtype)
    inverse_distance = torch.max(coil_coordinates[:, :, 3], dim=1).values
    envelope = inverse_distance**2
    factors = factors * envelope[:, None, None]
    matrices = torch.einsum("qpr,qsr->qps", factors.conj(), factors)
    n = matrices.shape[-1]
    trace_scale = torch.clamp(
        torch.real(torch.diagonal(matrices, dim1=-2, dim2=-1).sum(dim=-1)),
        min=0.0,
    ) + 1e-12 * envelope**2
    eye = torch.eye(n, dtype=complex_dtype, device=raw.device)
    matrices = matrices + (
        1e-9 * trace_scale[:, None, None] / max(n, 1)
    ) * eye[None]
    return 0.5 * (matrices + matrices.conj().transpose(-1, -2))


def _sample_signature(sample):
    return len(sample.scene.coils), len(sample.scene.packages)


def _normalized_tensor_batch(port_artifact, samples, device):
    normalized = [
        tuple(
            np.asarray(value, dtype=float)
            for value in port_artifact.normalizer.normalize(sample.encoded)
        )
        for sample in samples
    ]
    dtype = next(port_artifact.model.parameters()).dtype
    tensors = tuple(
        torch.as_tensor(
            np.stack([item[position] for item in normalized], axis=0),
            dtype=dtype,
            device=device,
        )
        for position in range(5)
    )
    coil_latent, package_latent = _batched_latent(
        port_artifact.model,
        *tensors,
    )
    return (
        coil_latent.detach(),
        package_latent.detach(),
        tensors[1].detach(),
        tensors[3].detach(),
    )


def _batched_spatial_shape_loss(model, port_artifact, samples, *, device):
    samples = tuple(samples)
    coil_latent, package_latent, coil_pair, coil_package = _normalized_tensor_batch(
        port_artifact,
        samples,
        device,
    )

    conductor_batch = []
    conductor_ids = []
    conductor_coordinates = []
    conductor_slices = []
    cursor = 0
    for batch_index, sample in enumerate(samples):
        conductor = sample.conductor_spatial_loss
        coordinates = _coordinate_features(
            sample.scene,
            conductor.coil_index,
            conductor.arc_fraction,
            conductor.xy,
        )
        count = len(conductor.coil_index)
        conductor_batch.append(np.full(count, batch_index, dtype=int))
        conductor_ids.append(np.asarray(conductor.coil_index, dtype=int))
        conductor_coordinates.append(coordinates)
        conductor_slices.append(slice(cursor, cursor + count))
        cursor += count
    raw_conductor = _conductor_raw_batched(
        model.conductor,
        coil_latent,
        coil_pair,
        np.concatenate(conductor_batch),
        np.concatenate(conductor_ids),
        np.concatenate(conductor_coordinates, axis=0),
    )

    package_batch = []
    package_ids = []
    package_coordinates = []
    package_gates = []
    package_slices = []
    cursor = 0
    for batch_index, sample in enumerate(samples):
        package = sample.package_spatial_loss
        coordinates = _package_coordinate_features(
            sample.scene,
            package.package_index,
            package.local_position,
        )
        count = len(package.package_index)
        package_batch.append(np.full(count, batch_index, dtype=int))
        package_ids.append(np.asarray(package.package_index, dtype=int))
        package_coordinates.append(coordinates)
        package_gates.append(
            _package_loss_gate(
                sample.scene,
                sample.frequency_hz,
                package.package_index,
            )
        )
        package_slices.append(slice(cursor, cursor + count))
        cursor += count
    raw_package = _package_raw_batched(
        model.package,
        coil_latent,
        package_latent,
        coil_package,
        np.concatenate(package_batch),
        np.concatenate(package_ids),
        np.concatenate(package_coordinates, axis=0),
    )
    raw_package = raw_package * torch.as_tensor(
        np.concatenate(package_gates),
        dtype=raw_package.real.dtype,
        device=raw_package.device,
    )[:, None, None]

    background_batch = []
    background_coil = []
    background_package = []
    background_slices = []
    background_weights = []
    background_gates = []
    cursor = 0
    for batch_index, sample in enumerate(samples):
        background = sample.background_spatial_loss
        if background is None:
            background_slices.append(slice(cursor, cursor))
            background_weights.append(np.empty(0, dtype=float))
            continue
        root_pose = sample.scene.coils[0].geometry.pose
        world = root_pose.apply(background.root_local_position)
        coil_coordinates, package_coordinates = background_coordinate_features(
            sample.scene,
            world,
            length_scale=float(sample.encoded.length_scale),
        )
        count = len(world)
        background_batch.append(np.full(count, batch_index, dtype=int))
        background_coil.append(coil_coordinates)
        background_package.append(package_coordinates)
        background_slices.append(slice(cursor, cursor + count))
        background_weights.append(np.asarray(background.weights, dtype=float))
        background_gates.append(
            np.full(
                count,
                background_loss_gate(sample.scene, sample.frequency_hz),
                dtype=float,
            )
        )
        cursor += count

    if background_batch:
        raw_background_all = _background_raw_batched(
            model.background,
            coil_latent,
            package_latent,
            np.concatenate(background_batch),
            np.concatenate(background_coil, axis=0),
            np.concatenate(background_package, axis=0),
        )
        raw_background_all = raw_background_all * torch.as_tensor(
            np.concatenate(background_gates),
            dtype=raw_background_all.real.dtype,
            device=raw_background_all.device,
        )[:, None, None]
    else:
        n_ports = len(samples[0].scene.coils)
        complex_dtype = torch.complex64 if coil_latent.dtype == torch.float32 else torch.complex128
        raw_background_all = torch.empty(
            (0, n_ports, n_ports),
            dtype=complex_dtype,
            device=device,
        )

    losses = []
    for batch_index, sample in enumerate(samples):
        conductor = sample.conductor_spatial_loss
        conductor_raw = raw_conductor[conductor_slices[batch_index]]
        conductor_transforms = _conductor_transforms(
            conductor_raw,
            conductor.coil_index,
            conductor.weights,
            sample.target_dissipation_channels[: len(sample.scene.coils)],
        )
        predicted_conductor = _apply_by_coil(
            conductor_raw,
            conductor.coil_index,
            conductor_transforms,
        )
        conductor_loss = _weighted_relative_loss(
            predicted_conductor,
            conductor.dissipation_matrix,
            conductor.weights,
        )

        package = sample.package_spatial_loss
        package_raw = raw_package[package_slices[batch_index]]
        background = sample.background_spatial_loss
        background_raw = raw_background_all[background_slices[batch_index]]
        transform = _environment_transform(
            package_raw,
            package.weights,
            background_raw,
            background_weights[batch_index],
            sample.target_dissipation_channels[len(sample.scene.coils)],
        )
        package_loss = _weighted_relative_loss(
            _apply_transform(package_raw, transform),
            package.dissipation_matrix,
            package.weights,
        )
        background_loss = torch.zeros(
            (),
            dtype=package_loss.dtype,
            device=package_loss.device,
        )
        if background is not None:
            background_loss = _weighted_relative_loss(
                _apply_transform(background_raw, transform),
                background.dissipation_matrix,
                background.weights,
            )
        losses.append(conductor_loss + package_loss + background_loss)
    return torch.stack(losses).mean()


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
    """Vectorize tensor spatial-network point evaluation across scene batches."""
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
    best_validation_error = None
    best_validation_shape_error = None
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
            loss = _batched_spatial_shape_loss(
                model,
                port_artifact,
                batch,
                device=resolved_device,
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
                    shape = _batched_spatial_shape_loss(
                        model,
                        port_artifact,
                        batch,
                        device=resolved_device,
                    )
                    shape_total += float(shape.detach().cpu()) * len(batch)
                    shape_count += len(batch)
            shape_score = shape_total / max(shape_count, 1)
            probe = TensorHybridSpatialLossArtifact(
                port_artifact,
                model,
                conductor_longitudinal_points=conductor_longitudinal_points,
                conductor_radial_order=conductor_radial_order,
                conductor_angular_order=conductor_angular_order,
                package_axial_order=package_axial_order,
                package_radial_order=package_radial_order,
                package_azimuthal_order=package_azimuthal_order,
                background_segments_per_turn=background_segments_per_turn,
                background_radial_order=background_radial_order,
                background_angular_order=background_angular_order,
                device=resolved_device,
            )
            score = float(
                np.mean(
                    [_sample_end_to_end_error(probe, sample) for sample in validation_samples]
                )
            )
            if best_validation_error is None or score < best_validation_error - float(min_improvement):
                best_validation_error = score
                best_validation_shape_error = shape_score
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
        conductor_longitudinal_points=conductor_longitudinal_points,
        conductor_radial_order=conductor_radial_order,
        conductor_angular_order=conductor_angular_order,
        package_axial_order=package_axial_order,
        package_radial_order=package_radial_order,
        package_azimuthal_order=package_azimuthal_order,
        background_segments_per_turn=background_segments_per_turn,
        background_radial_order=background_radial_order,
        background_angular_order=background_angular_order,
        device=resolved_device,
    )
    return artifact, HybridSpatialTrainingReport(
        float(final_loss),
        int(epochs_run),
        int(best_epoch),
        None if best_validation_error is None else float(best_validation_error),
        bool(stopped_early),
        None if best_validation_shape_error is None else float(best_validation_shape_error),
    )
