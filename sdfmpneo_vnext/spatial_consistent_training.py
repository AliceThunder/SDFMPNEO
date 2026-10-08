from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "consistent tensor spatial training requires the 'neural' extra"
    ) from exc

from .exterior_quadrature import (
    scene_conductor_geometry,
    unbounded_background_quadrature,
)
from .hybrid_background_spatial import (
    background_coordinate_features,
    background_loss_gate,
)
from .hybrid_spatial_neural import (
    _apply_by_coil,
    _apply_transform,
    _conductor_transforms,
    _coordinate_features,
    _environment_transform,
    _normalization_rule,
    _package_coordinate_features,
    _package_loss_gate,
    _package_normalization_rule,
    _weighted_relative_loss,
)
from .spatial_performance import (
    _background_raw_batched,
    _conductor_raw_batched,
    _normalized_tensor_batch,
    _package_raw_batched,
)


SPATIAL_TRAINING_CONTRACT = 2

_DEFAULT_NORMALIZATION = {
    "conductor_longitudinal_points": 12,
    "conductor_radial_order": 3,
    "conductor_angular_order": 16,
    "package_axial_order": 6,
    "package_radial_order": 4,
    "package_azimuthal_order": 16,
    "background_segments_per_turn": 16,
    "background_radial_order": 12,
    "background_angular_order": 48,
}


@dataclass(frozen=True)
class _NormalizationGeometry:
    conductor_ids: np.ndarray
    conductor_coordinates: np.ndarray
    conductor_weights: np.ndarray
    package_ids: np.ndarray
    package_coordinates: np.ndarray
    package_weights: np.ndarray
    background_coil_coordinates: np.ndarray
    background_package_coordinates: np.ndarray
    background_weights: np.ndarray


_geometry_cache = {}
_port_channel_cache = {}


def clear_spatial_consistency_caches() -> None:
    _geometry_cache.clear()
    _port_channel_cache.clear()


def resolve_spatial_normalization(config=None) -> dict[str, int]:
    values = dict(_DEFAULT_NORMALIZATION)
    if config:
        for key in values:
            if key in config:
                values[key] = int(config[key])
    if (
        values["conductor_longitudinal_points"] < 4
        or values["conductor_radial_order"] < 2
        or values["conductor_angular_order"] < 8
        or values["package_axial_order"] < 2
        or values["package_radial_order"] < 2
        or values["package_azimuthal_order"] < 8
        or values["background_segments_per_turn"] < 4
        or values["background_radial_order"] < 3
        or values["background_angular_order"] < 8
    ):
        raise ValueError("invalid tensor spatial normalization resolution")
    return values


def _normalization_geometry(sample, normalization) -> _NormalizationGeometry:
    normalization = resolve_spatial_normalization(normalization)
    cache_key = (
        id(sample),
        tuple(sorted((key, int(value)) for key, value in normalization.items())),
    )
    cached = _geometry_cache.get(cache_key)
    if cached is not None:
        return cached

    scene = sample.scene
    frequency_hz = float(sample.frequency_hz)
    (
        conductor_ids,
        conductor_arc,
        conductor_xy,
        conductor_weights,
    ) = _normalization_rule(
        scene,
        longitudinal_points=normalization["conductor_longitudinal_points"],
        radial_order=normalization["conductor_radial_order"],
        angular_order=normalization["conductor_angular_order"],
    )
    conductor_coordinates = _coordinate_features(
        scene,
        conductor_ids,
        conductor_arc,
        conductor_xy,
    )

    if scene.packages:
        package_ids, package_local, package_weights = _package_normalization_rule(
            scene,
            axial_order=normalization["package_axial_order"],
            radial_order=normalization["package_radial_order"],
            azimuthal_order=normalization["package_azimuthal_order"],
        )
    else:
        package_ids = np.empty(0, dtype=int)
        package_local = np.empty((0, 3), dtype=float)
        package_weights = np.empty(0, dtype=float)
    package_coordinates = _package_coordinate_features(
        scene,
        package_ids,
        package_local,
    )

    conductivity = float(scene.medium.loss_conductivity(frequency_hz))
    if conductivity > 0.0:
        segments, anchor_positions, anchor_radii = scene_conductor_geometry(
            scene,
            segments_per_turn=normalization["background_segments_per_turn"],
        )
        background_points, background_weights = unbounded_background_quadrature(
            scene,
            segments,
            anchor_positions,
            anchor_radii,
            radial_order=normalization["background_radial_order"],
            angular_order=normalization["background_angular_order"],
        )
        (
            background_coil_coordinates,
            background_package_coordinates,
        ) = background_coordinate_features(
            scene,
            background_points,
            length_scale=float(sample.encoded.length_scale),
        )
    else:
        n_coils = len(scene.coils)
        background_weights = np.empty(0, dtype=float)
        background_coil_coordinates = np.empty((0, n_coils, 5), dtype=float)
        background_package_coordinates = np.empty(
            (0, len(scene.packages), 5),
            dtype=float,
        )

    result = _NormalizationGeometry(
        np.asarray(conductor_ids, dtype=int),
        np.asarray(conductor_coordinates, dtype=float),
        np.asarray(conductor_weights, dtype=float),
        np.asarray(package_ids, dtype=int),
        np.asarray(package_coordinates, dtype=float),
        np.asarray(package_weights, dtype=float),
        np.asarray(background_coil_coordinates, dtype=float),
        np.asarray(background_package_coordinates, dtype=float),
        np.asarray(background_weights, dtype=float),
    )
    _geometry_cache[cache_key] = result
    return result


def _runtime_port_channels(port_artifact, sample) -> np.ndarray:
    key = (id(port_artifact), id(sample))
    cached = _port_channel_cache.get(key)
    if cached is not None:
        return cached
    prediction = port_artifact.predict_structured(sample.scene, sample.frequency_hz)
    channels = np.asarray(prediction.dissipation_channels, dtype=complex)
    n_coils = len(sample.scene.coils)
    if channels.shape != (n_coils + 1, n_coils, n_coils):
        raise ValueError("FAST port prediction has incompatible dissipation channels")
    channels = channels.copy()
    _port_channel_cache[key] = channels
    return channels


def _runtime_normalization_transforms(
    model,
    sample,
    coil_latent,
    package_latent,
    coil_pair,
    coil_package,
    runtime_channels,
    normalization,
):
    geometry = _normalization_geometry(sample, normalization)
    scene = sample.scene
    frequency_hz = float(sample.frequency_hz)
    device = coil_latent.device

    raw_conductor = model.conductor.raw_matrices(
        coil_latent,
        coil_pair,
        geometry.conductor_ids,
        geometry.conductor_coordinates,
    )
    conductor_transforms = _conductor_transforms(
        raw_conductor,
        geometry.conductor_ids,
        geometry.conductor_weights,
        runtime_channels[: len(scene.coils)],
    )

    raw_package = model.package.raw_matrices(
        coil_latent,
        package_latent,
        coil_package,
        geometry.package_ids,
        geometry.package_coordinates,
    )
    if len(geometry.package_ids):
        package_gate = torch.as_tensor(
            _package_loss_gate(scene, frequency_hz, geometry.package_ids),
            dtype=raw_package.real.dtype,
            device=device,
        )
        raw_package = raw_package * package_gate[:, None, None]

    if len(geometry.background_weights):
        raw_background = model.background.raw_matrices(
            coil_latent,
            package_latent,
            geometry.background_coil_coordinates,
            geometry.background_package_coordinates,
        )
        raw_background = raw_background * float(
            background_loss_gate(scene, frequency_hz)
        )
    else:
        n_ports = len(scene.coils)
        raw_background = torch.empty(
            (0, n_ports, n_ports),
            dtype=raw_package.dtype,
            device=device,
        )

    environment_transform = _environment_transform(
        raw_package,
        geometry.package_weights,
        raw_background,
        geometry.background_weights,
        runtime_channels[len(scene.coils)],
    )
    return conductor_transforms, environment_transform


def _teacher_targets_in_runtime_channel_basis(
    sample,
    runtime_channels,
    *,
    dtype,
    device,
):
    n_coils = len(sample.scene.coils)
    conductor = sample.conductor_spatial_loss
    conductor_matrix = torch.as_tensor(
        conductor.dissipation_matrix,
        dtype=dtype,
        device=device,
    )
    conductor_transforms = _conductor_transforms(
        conductor_matrix,
        conductor.coil_index,
        conductor.weights,
        runtime_channels[:n_coils],
    )
    conductor_target = _apply_by_coil(
        conductor_matrix,
        conductor.coil_index,
        conductor_transforms,
    )

    package = sample.package_spatial_loss
    package_matrix = torch.as_tensor(
        package.dissipation_matrix,
        dtype=dtype,
        device=device,
    )
    background = sample.background_spatial_loss
    if background is None:
        background_matrix = torch.empty(
            (0, n_coils, n_coils),
            dtype=dtype,
            device=device,
        )
        background_weights = np.empty(0, dtype=float)
    else:
        background_matrix = torch.as_tensor(
            background.dissipation_matrix,
            dtype=dtype,
            device=device,
        )
        background_weights = background.weights

    environment_transform = _environment_transform(
        package_matrix,
        package.weights,
        background_matrix,
        background_weights,
        runtime_channels[n_coils],
    )
    package_target = _apply_transform(package_matrix, environment_transform)
    background_target = (
        _apply_transform(background_matrix, environment_transform)
        if int(background_matrix.shape[0])
        else background_matrix
    )
    return conductor_target, package_target, background_target


def consistent_batched_spatial_shape_loss(
    model,
    port_artifact,
    samples,
    *,
    device,
    normalization=None,
):
    """Spatial shape loss using the exact FAST inference normalization contract."""
    samples = tuple(samples)
    if not samples:
        raise ValueError("spatial shape loss requires at least one sample")
    normalization = resolve_spatial_normalization(normalization)
    (
        coil_latent,
        package_latent,
        coil_pair,
        coil_package,
    ) = _normalized_tensor_batch(port_artifact, samples, device)

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
    gates = []
    cursor = 0
    for batch_index, sample in enumerate(samples):
        background = sample.background_spatial_loss
        if background is None:
            background_slices.append(slice(cursor, cursor))
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
        gates.append(
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
            np.concatenate(gates),
            dtype=raw_background_all.real.dtype,
            device=raw_background_all.device,
        )[:, None, None]
    else:
        n_ports = len(samples[0].scene.coils)
        complex_dtype = (
            torch.complex64 if coil_latent.dtype == torch.float32 else torch.complex128
        )
        raw_background_all = torch.empty(
            (0, n_ports, n_ports),
            dtype=complex_dtype,
            device=device,
        )

    losses = []
    for batch_index, sample in enumerate(samples):
        runtime_channels = _runtime_port_channels(port_artifact, sample)
        conductor_transforms, environment_transform = _runtime_normalization_transforms(
            model,
            sample,
            coil_latent[batch_index],
            package_latent[batch_index],
            coil_pair[batch_index],
            coil_package[batch_index],
            runtime_channels,
            normalization,
        )

        conductor = sample.conductor_spatial_loss
        predicted_conductor = _apply_by_coil(
            raw_conductor[conductor_slices[batch_index]],
            conductor.coil_index,
            conductor_transforms,
        )
        package = sample.package_spatial_loss
        predicted_package = _apply_transform(
            raw_package[package_slices[batch_index]],
            environment_transform,
        )
        background = sample.background_spatial_loss
        background_raw = raw_background_all[background_slices[batch_index]]
        predicted_background = (
            _apply_transform(background_raw, environment_transform)
            if background is not None
            else background_raw
        )

        (
            target_conductor,
            target_package,
            target_background,
        ) = _teacher_targets_in_runtime_channel_basis(
            sample,
            runtime_channels,
            dtype=predicted_conductor.dtype,
            device=predicted_conductor.device,
        )
        conductor_loss = _weighted_relative_loss(
            predicted_conductor,
            target_conductor,
            conductor.weights,
        )
        package_loss = _weighted_relative_loss(
            predicted_package,
            target_package,
            package.weights,
        )
        background_loss = torch.zeros(
            (),
            dtype=package_loss.dtype,
            device=package_loss.device,
        )
        if background is not None:
            background_loss = _weighted_relative_loss(
                predicted_background,
                target_background,
                background.weights,
            )
        losses.append(conductor_loss + package_loss + background_loss)
    return torch.stack(losses).mean()
