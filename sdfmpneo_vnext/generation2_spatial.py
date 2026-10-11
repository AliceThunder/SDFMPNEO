from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("generation-2 spatial surrogate requires the 'neural' extra") from exc

from .device import resolve_torch_device
from .exterior_quadrature import scene_conductor_geometry, unbounded_background_quadrature
from .generation2_features import (
    GENERATION2_COIL_FEATURE_DIM,
    GENERATION2_CROSS_FEATURE_DIM,
    GENERATION2_PACKAGE_FEATURE_DIM,
    GENERATION2_PAIR_FEATURE_DIM,
    encode_generation2_scene,
)
from .generation2_objectives import (
    numpy_relative_from_error_energy,
    numpy_weighted_spatial_error_energy,
    spatial_relative_from_error_energy,
    weighted_spatial_error_energy,
)
from .generation2_port import (
    Generation2PortArtifact,
    generation2_batched_latent,
    generation2_port_fingerprint,
)
from .hybrid_background_spatial import (
    background_coordinate_features,
    background_loss_gate,
)
from .hybrid_spatial_neural import (
    _apply_by_coil,
    _apply_transform,
    _conductor_transforms,
    _environment_transform,
    _normalization_rule,
    _package_coordinate_features,
    _package_loss_gate,
    _package_normalization_rule,
)
from .spatial_boundary_features import (
    BoundaryAwareHybridSpatialLossShapeNet,
    BoundaryAwarePreparedHybridSpatialLossField,
    boundary_aware_conductor_coordinates,
)
from .spatial_consistent_training import resolve_spatial_normalization
from .spatial_performance import (
    _background_raw_batched,
    _conductor_raw_batched,
    _package_raw_batched,
)


GENERATION2_SPATIAL_ARTIFACT_SCHEMA = 1
GENERATION2_SPATIAL_MODEL_GENERATION = 2


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int):
    layers = []
    width = int(input_dim)
    for _ in range(int(depth)):
        layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = int(hidden_dim)
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def _aggregate(messages, reference):
    if not messages:
        return torch.zeros_like(reference)
    return torch.stack(messages, dim=0).sum(dim=0) / math.sqrt(len(messages))


class Generation2SpatialNet(nn.Module):
    """Spatial-specific scene refiner plus continuous PSD local decoders."""

    def __init__(
        self,
        port_hidden_dim: int,
        *,
        context_hidden_dim: int = 64,
        context_rounds: int = 1,
        context_depth: int = 1,
        field_hidden_dim: int = 128,
        factor_rank: int = 4,
        depth: int = 3,
    ):
        super().__init__()
        if (
            port_hidden_dim < 1
            or context_hidden_dim < 4
            or context_rounds < 1
            or context_depth < 1
        ):
            raise ValueError("invalid generation-2 spatial context dimensions")
        self.port_hidden_dim = int(port_hidden_dim)
        self.hidden_dim = int(context_hidden_dim)
        self.context_rounds = int(context_rounds)
        self.context_depth = int(context_depth)
        self.field_hidden_dim = int(field_hidden_dim)
        self.factor_rank = int(factor_rank)
        self.depth = int(depth)
        h = self.hidden_dim

        self.coil_adapter = _mlp(
            self.port_hidden_dim + GENERATION2_COIL_FEATURE_DIM,
            h,
            h,
            self.context_depth,
        )
        self.package_adapter = _mlp(
            self.port_hidden_dim + GENERATION2_PACKAGE_FEATURE_DIM,
            h,
            h,
            self.context_depth,
        )
        self.coil_pair_encoder = _mlp(
            2 * h + GENERATION2_PAIR_FEATURE_DIM,
            h,
            h,
            self.context_depth,
        )
        self.package_pair_encoder = _mlp(
            2 * h + GENERATION2_PAIR_FEATURE_DIM,
            h,
            h,
            self.context_depth,
        )
        self.coil_to_package_encoder = _mlp(
            2 * h + GENERATION2_CROSS_FEATURE_DIM,
            h,
            h,
            self.context_depth,
        )
        self.package_to_coil_encoder = _mlp(
            2 * h + GENERATION2_CROSS_FEATURE_DIM,
            h,
            h,
            self.context_depth,
        )
        self.coil_update = _mlp(3 * h, h, h, self.context_depth)
        self.package_update = _mlp(3 * h, h, h, self.context_depth)
        self.coil_norm = nn.LayerNorm(h)
        self.package_norm = nn.LayerNorm(h)

        self.decoder = BoundaryAwareHybridSpatialLossShapeNet(
            h,
            GENERATION2_PAIR_FEATURE_DIM,
            GENERATION2_CROSS_FEATURE_DIM,
            field_hidden_dim=self.field_hidden_dim,
            factor_rank=self.factor_rank,
            depth=self.depth,
        )

    @property
    def conductor(self):
        return self.decoder.conductor

    @property
    def package(self):
        return self.decoder.package

    @property
    def background(self):
        return self.decoder.background

    def refine_batch(
        self,
        port_coil_latent,
        port_package_latent,
        coil_features,
        coil_pair_features,
        package_features,
        coil_package_features,
        package_pair_features,
    ):
        coil = self.coil_adapter(
            torch.cat((port_coil_latent, coil_features), dim=-1)
        )
        package = self.package_adapter(
            torch.cat((port_package_latent, package_features), dim=-1)
        )
        _, n_coils, _ = coil.shape
        _, n_packages, _ = package.shape

        for _ in range(self.context_rounds):
            coil_pair_messages = []
            for i in range(n_coils):
                messages = []
                for j in range(n_coils):
                    if i == j:
                        continue
                    messages.append(
                        self.coil_pair_encoder(
                            torch.cat(
                                (coil[:, i], coil[:, j], coil_pair_features[:, i, j]),
                                dim=-1,
                            )
                        )
                    )
                coil_pair_messages.append(_aggregate(messages, coil[:, i]))

            package_pair_messages = []
            coil_to_package_messages = []
            for p in range(n_packages):
                pair_messages = []
                for q in range(n_packages):
                    if p == q:
                        continue
                    pair_messages.append(
                        self.package_pair_encoder(
                            torch.cat(
                                (package[:, p], package[:, q], package_pair_features[:, p, q]),
                                dim=-1,
                            )
                        )
                    )
                package_pair_messages.append(_aggregate(pair_messages, package[:, p]))
                coil_messages = []
                for c in range(n_coils):
                    coil_messages.append(
                        self.coil_to_package_encoder(
                            torch.cat(
                                (package[:, p], coil[:, c], coil_package_features[:, c, p]),
                                dim=-1,
                            )
                        )
                    )
                coil_to_package_messages.append(_aggregate(coil_messages, package[:, p]))

            package_to_coil_messages = []
            for c in range(n_coils):
                messages = []
                for p in range(n_packages):
                    messages.append(
                        self.package_to_coil_encoder(
                            torch.cat(
                                (coil[:, c], package[:, p], coil_package_features[:, c, p]),
                                dim=-1,
                            )
                        )
                    )
                package_to_coil_messages.append(_aggregate(messages, coil[:, c]))

            coil = torch.stack(
                [
                    self.coil_norm(
                        coil[:, c]
                        + self.coil_update(
                            torch.cat(
                                (
                                    coil[:, c],
                                    coil_pair_messages[c],
                                    package_to_coil_messages[c],
                                ),
                                dim=-1,
                            )
                        )
                    )
                    for c in range(n_coils)
                ],
                dim=1,
            )
            if n_packages:
                package = torch.stack(
                    [
                        self.package_norm(
                            package[:, p]
                            + self.package_update(
                                torch.cat(
                                    (
                                        package[:, p],
                                        package_pair_messages[p],
                                        coil_to_package_messages[p],
                                    ),
                                    dim=-1,
                                )
                            )
                        )
                        for p in range(n_packages)
                    ],
                    dim=1,
                )
        return coil, package


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


def _normalization_geometry(sample, normalization) -> _NormalizationGeometry:
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
    conductor_coordinates = boundary_aware_conductor_coordinates(
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
    package_coordinates = _package_coordinate_features(scene, package_ids, package_local)

    if float(scene.medium.loss_conductivity(frequency_hz)) > 0.0:
        segments, anchors, radii = scene_conductor_geometry(
            scene,
            segments_per_turn=normalization["background_segments_per_turn"],
        )
        points, background_weights = unbounded_background_quadrature(
            scene,
            segments,
            anchors,
            radii,
            radial_order=normalization["background_radial_order"],
            angular_order=normalization["background_angular_order"],
        )
        encoded = encode_generation2_scene(scene, frequency_hz)
        background_coil, background_package = background_coordinate_features(
            scene,
            points,
            length_scale=float(encoded.length_scale),
        )
    else:
        background_weights = np.empty(0, dtype=float)
        background_coil = np.empty((0, len(scene.coils), 5), dtype=float)
        background_package = np.empty((0, len(scene.packages), 5), dtype=float)

    return _NormalizationGeometry(
        np.asarray(conductor_ids, dtype=int),
        np.asarray(conductor_coordinates, dtype=float),
        np.asarray(conductor_weights, dtype=float),
        np.asarray(package_ids, dtype=int),
        np.asarray(package_coordinates, dtype=float),
        np.asarray(package_weights, dtype=float),
        np.asarray(background_coil, dtype=float),
        np.asarray(background_package, dtype=float),
        np.asarray(background_weights, dtype=float),
    )


def _context_batch(port_artifact, spatial_model, samples, device):
    normalized = []
    for sample in samples:
        encoded = encode_generation2_scene(sample.scene, sample.frequency_hz)
        normalized.append(
            tuple(
                np.asarray(value, dtype=float)
                for value in port_artifact.normalizer.normalize(encoded)
            )
        )
    dtype = next(port_artifact.model.parameters()).dtype
    tensors = tuple(
        torch.as_tensor(
            np.stack([item[position] for item in normalized], axis=0),
            dtype=dtype,
            device=device,
        )
        for position in range(5)
    )
    port_artifact.model.eval()
    with torch.no_grad():
        port_coil, port_package = generation2_batched_latent(
            port_artifact.model,
            *tensors,
        )
    coil, package = spatial_model.refine_batch(
        port_coil.detach(),
        port_package.detach(),
        *tensors,
    )
    return coil, package, tensors[1], tensors[3]


def _normalization_transforms(
    model,
    sample,
    coil_latent,
    package_latent,
    coil_pair,
    coil_package,
    target_channels,
    normalization,
):
    geometry = _normalization_geometry(sample, normalization)
    scene = sample.scene
    frequency_hz = float(sample.frequency_hz)
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
        target_channels[: len(scene.coils)],
    )
    raw_package = model.package.raw_matrices(
        coil_latent,
        package_latent,
        coil_package,
        geometry.package_ids,
        geometry.package_coordinates,
    )
    if len(geometry.package_ids):
        gate = torch.as_tensor(
            _package_loss_gate(scene, frequency_hz, geometry.package_ids),
            dtype=raw_package.real.dtype,
            device=raw_package.device,
        )
        raw_package = raw_package * gate[:, None, None]
    if len(geometry.background_weights):
        raw_background = model.background.raw_matrices(
            coil_latent,
            package_latent,
            geometry.background_coil_coordinates,
            geometry.background_package_coordinates,
        )
        raw_background = raw_background * float(background_loss_gate(scene, frequency_hz))
    else:
        complex_dtype = raw_package.dtype
        raw_background = torch.empty(
            (0, len(scene.coils), len(scene.coils)),
            dtype=complex_dtype,
            device=raw_package.device,
        )
    environment_transform = _environment_transform(
        raw_package,
        geometry.package_weights,
        raw_background,
        geometry.background_weights,
        target_channels[len(scene.coils)],
    )
    return conductor_transforms, environment_transform


def generation2_batched_spatial_shape_loss(
    model: Generation2SpatialNet,
    port_artifact: Generation2PortArtifact,
    samples,
    *,
    device,
    normalization=None,
):
    """Canonical Spatial loss normalized only by teacher channels."""
    samples = tuple(samples)
    if not samples:
        raise ValueError("generation-2 spatial loss requires at least one sample")
    normalization = resolve_spatial_normalization(normalization)
    coil_latent, package_latent, coil_pair, coil_package = _context_batch(
        port_artifact,
        model,
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
        coordinates = boundary_aware_conductor_coordinates(
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
            _package_loss_gate(sample.scene, sample.frequency_hz, package.package_index)
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
    if package_gates:
        raw_package = raw_package * torch.as_tensor(
            np.concatenate(package_gates),
            dtype=raw_package.real.dtype,
            device=raw_package.device,
        )[:, None, None]

    background_batch = []
    background_coil = []
    background_package = []
    background_slices = []
    background_gates = []
    cursor = 0
    for batch_index, sample in enumerate(samples):
        background = sample.background_spatial_loss
        if background is None:
            background_slices.append(slice(cursor, cursor))
            continue
        root_pose = sample.scene.coils[0].geometry.pose
        world = root_pose.apply(background.root_local_position)
        encoded = encode_generation2_scene(sample.scene, sample.frequency_hz)
        coil_coordinates, package_coordinates = background_coordinate_features(
            sample.scene,
            world,
            length_scale=float(encoded.length_scale),
        )
        count = len(world)
        background_batch.append(np.full(count, batch_index, dtype=int))
        background_coil.append(coil_coordinates)
        background_package.append(package_coordinates)
        background_slices.append(slice(cursor, cursor + count))
        background_gates.append(
            np.full(
                count,
                background_loss_gate(sample.scene, sample.frequency_hz),
                dtype=float,
            )
        )
        cursor += count

    if background_batch:
        raw_background = _background_raw_batched(
            model.background,
            coil_latent,
            package_latent,
            np.concatenate(background_batch),
            np.concatenate(background_coil, axis=0),
            np.concatenate(background_package, axis=0),
        )
        raw_background = raw_background * torch.as_tensor(
            np.concatenate(background_gates),
            dtype=raw_background.real.dtype,
            device=raw_background.device,
        )[:, None, None]
    else:
        complex_dtype = (
            torch.complex64 if coil_latent.dtype == torch.float32 else torch.complex128
        )
        raw_background = torch.empty(
            (0, len(samples[0].scene.coils), len(samples[0].scene.coils)),
            dtype=complex_dtype,
            device=device,
        )

    losses = []
    for batch_index, sample in enumerate(samples):
        teacher_channels = np.asarray(sample.target_dissipation_channels, dtype=complex)
        conductor_transforms, environment_transform = _normalization_transforms(
            model,
            sample,
            coil_latent[batch_index],
            package_latent[batch_index],
            coil_pair[batch_index],
            coil_package[batch_index],
            teacher_channels,
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
        background_raw = raw_background[background_slices[batch_index]]
        predicted_background = (
            _apply_transform(background_raw, environment_transform)
            if background is not None
            else background_raw
        )

        error, energy = weighted_spatial_error_energy(
            predicted_conductor,
            conductor.dissipation_matrix,
            conductor.weights,
        )
        region_error, region_energy = weighted_spatial_error_energy(
            predicted_package,
            package.dissipation_matrix,
            package.weights,
        )
        error = error + region_error
        energy = energy + region_energy
        if background is not None:
            region_error, region_energy = weighted_spatial_error_energy(
                predicted_background,
                background.dissipation_matrix,
                background.weights,
            )
            error = error + region_error
            energy = energy + region_energy
        losses.append(spatial_relative_from_error_energy(error, energy))
    return torch.stack(losses).mean()


def generation2_spatial_end_to_end_error(artifact, sample) -> float:
    prepared = artifact.prepare(sample.scene, sample.frequency_hz)
    conductor = sample.conductor_spatial_loss
    total_error, total_energy = numpy_weighted_spatial_error_energy(
        prepared.local_dissipation_matrices(
            conductor.coil_index,
            conductor.arc_fraction,
            conductor.xy,
        ),
        conductor.dissipation_matrix,
        conductor.weights,
    )
    package = sample.package_spatial_loss
    error, energy = numpy_weighted_spatial_error_energy(
        prepared.package_local_dissipation_matrices(
            package.package_index,
            package.local_position,
        ),
        package.dissipation_matrix,
        package.weights,
    )
    total_error += error
    total_energy += energy
    if sample.background_spatial_loss is not None:
        background = sample.background_spatial_loss
        root_pose = sample.scene.coils[0].geometry.pose
        world = root_pose.apply(background.root_local_position)
        error, energy = numpy_weighted_spatial_error_energy(
            prepared.background_dissipation_matrices(world),
            background.dissipation_matrix,
            background.weights,
        )
        total_error += error
        total_energy += energy
    return numpy_relative_from_error_energy(total_error, total_energy)


def _background_sigma_upper(port_artifact) -> float:
    bounds = port_artifact.material_domain.get("background_sigma")
    return 0.0 if bounds is None else float(bounds[1])


class Generation2SpatialArtifact:
    artifact_schema = GENERATION2_SPATIAL_ARTIFACT_SCHEMA
    model_generation = GENERATION2_SPATIAL_MODEL_GENERATION
    supports_packages = True
    supports_tensor_electric = True

    def __init__(
        self,
        port_artifact: Generation2PortArtifact,
        model: Generation2SpatialNet,
        *,
        normalization=None,
        device: str = "cpu",
    ):
        if not isinstance(port_artifact, Generation2PortArtifact):
            raise TypeError("generation-2 Spatial requires a generation-2 Port artifact")
        self.normalization = resolve_spatial_normalization(normalization)
        self.port_artifact = port_artifact
        self.port_fingerprint = generation2_port_fingerprint(port_artifact)
        self.device = resolve_torch_device(device)
        self.port_artifact.model.to(self.device)
        self.port_artifact.device = self.device
        dtype = next(self.port_artifact.model.parameters()).dtype
        if self.device == "mps" and dtype == torch.float64:
            raise ValueError("float64 generation-2 Spatial artifacts cannot run on MPS")
        self.model = model.to(device=self.device, dtype=dtype)
        self.model.eval()
        self.supports_lossy_background = bool(
            _background_sigma_upper(port_artifact) > 0.0
        )

    def _context(self, scene, frequency_hz):
        encoded = encode_generation2_scene(scene, frequency_hz)
        normalized_np = tuple(
            np.asarray(value, dtype=float)
            for value in self.port_artifact.normalizer.normalize(encoded)
        )
        dtype = next(self.port_artifact.model.parameters()).dtype
        tensors = tuple(
            torch.as_tensor(value, dtype=dtype, device=self.device).unsqueeze(0)
            for value in normalized_np
        )
        self.port_artifact.model.eval()
        with torch.no_grad():
            port_coil, port_package = generation2_batched_latent(
                self.port_artifact.model,
                *tensors,
            )
            coil, package = self.model.refine_batch(
                port_coil,
                port_package,
                *tensors,
            )
        return coil[0], package[0], tensors[1][0], tensors[3][0], float(encoded.length_scale)

    def prepare(self, scene, frequency_hz):
        prediction = self.port_artifact.predict_structured(scene, frequency_hz)
        n_coils = len(scene.coils)
        if prediction.dissipation_channels.shape != (n_coils + 1, n_coils, n_coils):
            raise ValueError("generation-2 Spatial expects conductor channels plus one environment channel")
        conductivity = float(scene.medium.loss_conductivity(frequency_hz))
        if conductivity > 0.0 and not self.supports_lossy_background:
            raise NotImplementedError("generation-2 Spatial artifact was not trained for lossy backgrounds")
        coil_latent, package_latent, coil_pair, coil_package, length_scale = self._context(
            scene,
            frequency_hz,
        )
        normalization = self.normalization
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
        conductor_coordinates = boundary_aware_conductor_coordinates(
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
        package_coordinates = _package_coordinate_features(scene, package_ids, package_local)
        segments, anchors, radii = scene_conductor_geometry(
            scene,
            segments_per_turn=normalization["background_segments_per_turn"],
        )
        if conductivity > 0.0:
            background_points, background_weights = unbounded_background_quadrature(
                scene,
                segments,
                anchors,
                radii,
                radial_order=normalization["background_radial_order"],
                angular_order=normalization["background_angular_order"],
            )
            background_coil, background_package = background_coordinate_features(
                scene,
                background_points,
                length_scale=length_scale,
            )
        else:
            background_points = np.empty((0, 3), dtype=float)
            background_weights = np.empty(0, dtype=float)
            background_coil = np.empty((0, n_coils, 5), dtype=float)
            background_package = np.empty((0, len(scene.packages), 5), dtype=float)

        self.model.eval()
        with torch.no_grad():
            raw_conductor = self.model.conductor.raw_matrices(
                coil_latent,
                coil_pair,
                conductor_ids,
                conductor_coordinates,
            )
            conductor_transforms = _conductor_transforms(
                raw_conductor,
                conductor_ids,
                conductor_weights,
                prediction.dissipation_channels[:n_coils],
            )
            raw_package = self.model.package.raw_matrices(
                coil_latent,
                package_latent,
                coil_package,
                package_ids,
                package_coordinates,
            )
            if len(package_ids):
                gate = torch.as_tensor(
                    _package_loss_gate(scene, frequency_hz, package_ids),
                    dtype=raw_package.real.dtype,
                    device=raw_package.device,
                )
                raw_package = raw_package * gate[:, None, None]
            if len(background_points):
                raw_background = self.model.background.raw_matrices(
                    coil_latent,
                    package_latent,
                    background_coil,
                    background_package,
                )
                raw_background = raw_background * float(background_loss_gate(scene, frequency_hz))
            else:
                raw_background = torch.empty(
                    (0, n_coils, n_coils),
                    dtype=raw_package.dtype,
                    device=raw_package.device,
                )
            environment_transform = _environment_transform(
                raw_package,
                package_weights,
                raw_background,
                background_weights,
                prediction.dissipation_channels[n_coils],
            )
            conductor_values = _apply_by_coil(
                raw_conductor,
                conductor_ids,
                conductor_transforms,
            )
            package_values = _apply_transform(raw_package, environment_transform)
            background_values = (
                _apply_transform(raw_background, environment_transform)
                if int(raw_background.shape[0])
                else raw_background
            )

        integrated = np.zeros_like(prediction.dissipation_channels, dtype=complex)
        conductor_values_np = conductor_values.detach().cpu().numpy()
        for coil in range(n_coils):
            mask = conductor_ids == coil
            integrated[coil] = np.sum(
                conductor_weights[mask, None, None] * conductor_values_np[mask],
                axis=0,
            )
        if len(package_weights):
            integrated[n_coils] += np.sum(
                package_weights[:, None, None] * package_values.detach().cpu().numpy(),
                axis=0,
            )
        if len(background_weights):
            integrated[n_coils] += np.sum(
                background_weights[:, None, None] * background_values.detach().cpu().numpy(),
                axis=0,
            )
        closure = float(
            np.linalg.norm(integrated - prediction.dissipation_channels)
            / max(float(np.linalg.norm(prediction.dissipation_channels)), 1e-30)
        )
        return BoundaryAwarePreparedHybridSpatialLossField(
            scene,
            float(frequency_hz),
            prediction,
            self.model,
            coil_latent,
            package_latent,
            coil_pair,
            coil_package,
            conductor_transforms,
            environment_transform,
            segments,
            anchors,
            radii,
            float(length_scale),
            closure,
            self.device,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": GENERATION2_SPATIAL_ARTIFACT_SCHEMA,
                "model_generation": GENERATION2_SPATIAL_MODEL_GENERATION,
                "port_fingerprint": self.port_fingerprint,
                "model_config": {
                    "port_hidden_dim": self.model.port_hidden_dim,
                    "context_hidden_dim": self.model.hidden_dim,
                    "context_rounds": self.model.context_rounds,
                    "context_depth": self.model.context_depth,
                    "field_hidden_dim": self.model.field_hidden_dim,
                    "factor_rank": self.model.factor_rank,
                    "depth": self.model.depth,
                },
                "model_state": self.model.state_dict(),
                "normalization": dict(self.normalization),
            },
            path,
        )

    @staticmethod
    def load(path, port_artifact, *, device: str = "cpu"):
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(payload.get("schema", -1)) != GENERATION2_SPATIAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported generation-2 Spatial artifact schema")
        if int(payload.get("model_generation", -1)) != GENERATION2_SPATIAL_MODEL_GENERATION:
            raise ValueError("artifact is not a generation-2 Spatial model")
        if payload.get("port_fingerprint") != generation2_port_fingerprint(port_artifact):
            raise ValueError("generation-2 Spatial Port fingerprint mismatch")
        dtype = next(port_artifact.model.parameters()).dtype
        model = Generation2SpatialNet(**payload["model_config"]).to(dtype=dtype)
        model.load_state_dict(payload["model_state"])
        return Generation2SpatialArtifact(
            port_artifact,
            model,
            normalization=payload["normalization"],
            device=device,
        )


__all__ = [
    "GENERATION2_SPATIAL_ARTIFACT_SCHEMA",
    "GENERATION2_SPATIAL_MODEL_GENERATION",
    "Generation2SpatialNet",
    "Generation2SpatialArtifact",
    "generation2_batched_spatial_shape_loss",
    "generation2_spatial_end_to_end_error",
]
