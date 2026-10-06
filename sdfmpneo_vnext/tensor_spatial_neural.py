from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.tensor_spatial_neural requires the 'neural' extra"
    ) from exc

from .device import resolve_torch_device
from .exterior_quadrature import (
    scene_conductor_geometry,
    unbounded_background_quadrature,
)
from .hybrid_background_spatial import (
    background_coordinate_features,
    background_loss_gate,
)
from .hybrid_spatial_neural import (
    HybridSpatialLossShapeNet,
    HybridSpatialTrainingReport,
    PreparedHybridSpatialLossField,
    _apply_by_coil,
    _apply_transform,
    _conductor_transforms,
    _coordinate_features,
    _environment_transform,
    _normalization_rule,
    _package_coordinate_features,
    _package_loss_gate,
    _package_normalization_rule,
    _weighted_relative_error_numpy,
    _weighted_relative_loss,
)
from .scene import Scene
from .tensor_features import encode_tensor_hybrid_scene_invariant
from .tensor_spatial_training_data import TensorHybridSpatialTeacherSample


TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA = 1


def tensor_port_fingerprint(port_artifact) -> str:
    """Stable fingerprint for tensor FAST port weights and preprocessing."""
    digest = sha256()
    model = port_artifact.model
    config = {
        "schema": int(getattr(port_artifact, "artifact_schema", 1)),
        "model": {
            "coil_dim": int(model.coil_dim),
            "coil_pair_dim": int(model.coil_pair_dim),
            "package_dim": int(model.package_dim),
            "cross_dim": int(model.cross_dim),
            "package_pair_dim": int(model.package_pair_dim),
            "hidden_dim": int(model.hidden_dim),
            "factor_rank": int(model.factor_rank),
            "depth": int(model.depth),
        },
        "baseline_segments": int(port_artifact.baseline_segments),
        "material_domain": port_artifact.material_domain,
        "geometry_domain": getattr(port_artifact, "geometry_domain", None),
    }
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    def update_array(name, value):
        array = np.asarray(value)
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())

    for name, tensor in sorted(model.state_dict().items()):
        update_array(
            "state:" + name,
            tensor.detach().cpu().contiguous().numpy(),
        )
    for name, value in sorted(port_artifact.normalizer.to_dict().items()):
        update_array("normalizer:" + name, value)
    return digest.hexdigest()


def _tensor_latent(port_artifact, scene: Scene, frequency_hz: float):
    encoded = encode_tensor_hybrid_scene_invariant(scene, frequency_hz)
    normalized = port_artifact.normalizer.normalize(encoded)
    model = port_artifact.model
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    tensors = tuple(
        torch.as_tensor(value, dtype=dtype, device=device)
        for value in normalized
    )
    model.eval()
    with torch.no_grad():
        coil_latent, package_latent = model._latent(*tensors)
    return (
        coil_latent,
        package_latent,
        tensors[1],
        tensors[3],
        float(encoded.length_scale),
    )


def _background_sigma_upper(port_artifact) -> float:
    bounds = port_artifact.material_domain.get("background_sigma")
    return 0.0 if bounds is None else float(bounds[1])


class TensorHybridSpatialLossArtifact:
    """Tensor-aware continuous FAST loss artifact."""

    supports_packages = True
    supports_tensor_electric = True

    def __init__(
        self,
        port_artifact,
        model: HybridSpatialLossShapeNet,
        *,
        conductor_longitudinal_points: int = 12,
        conductor_radial_order: int = 3,
        conductor_angular_order: int = 16,
        package_axial_order: int = 6,
        package_radial_order: int = 4,
        package_azimuthal_order: int = 16,
        background_segments_per_turn: int = 16,
        background_radial_order: int = 12,
        background_angular_order: int = 48,
        device: str = "cpu",
    ):
        if not bool(getattr(port_artifact, "supports_tensor_electric", False)):
            raise TypeError(
                "tensor spatial artifact requires a tensor-aware FAST port artifact"
            )
        if not bool(getattr(port_artifact, "supports_packages", False)):
            raise TypeError(
                "tensor spatial artifact requires a package-aware FAST port artifact"
            )
        resolved_device = resolve_torch_device(device)
        self.port_artifact = port_artifact
        self.port_fingerprint = tensor_port_fingerprint(port_artifact)
        self.port_artifact.model.to(resolved_device)
        self.port_artifact.device = resolved_device
        port_dtype = next(self.port_artifact.model.parameters()).dtype
        if resolved_device == "mps" and port_dtype == torch.float64:
            raise ValueError(
                "float64 tensor spatial artifacts cannot run on MPS"
            )
        self.model = model.to(device=resolved_device, dtype=port_dtype)
        self.model.eval()
        self.conductor_longitudinal_points = int(conductor_longitudinal_points)
        self.conductor_radial_order = int(conductor_radial_order)
        self.conductor_angular_order = int(conductor_angular_order)
        self.package_axial_order = int(package_axial_order)
        self.package_radial_order = int(package_radial_order)
        self.package_azimuthal_order = int(package_azimuthal_order)
        self.background_segments_per_turn = int(background_segments_per_turn)
        self.background_radial_order = int(background_radial_order)
        self.background_angular_order = int(background_angular_order)
        if (
            self.conductor_longitudinal_points < 4
            or self.conductor_radial_order < 2
            or self.conductor_angular_order < 8
            or self.package_axial_order < 2
            or self.package_radial_order < 2
            or self.package_azimuthal_order < 8
            or self.background_segments_per_turn < 4
            or self.background_radial_order < 3
            or self.background_angular_order < 8
        ):
            raise ValueError("invalid tensor spatial normalization resolution")
        self.supports_lossy_background = bool(
            getattr(port_artifact, "supports_lossy_background", False)
            and _background_sigma_upper(port_artifact) > 0.0
        )
        self.device = resolved_device

    def prepare(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedHybridSpatialLossField:
        prediction = self.port_artifact.predict_structured(scene, frequency_hz)
        n_coils = len(scene.coils)
        if prediction.dissipation_channels.shape[0] != n_coils + 1:
            raise ValueError(
                "tensor spatial artifact expects one conductor channel per coil "
                "plus one aggregate electric-environment channel"
            )
        conductivity = float(scene.medium.loss_conductivity(frequency_hz))
        if conductivity > 0.0 and not self.supports_lossy_background:
            raise NotImplementedError(
                "this tensor spatial artifact was not trained for lossy backgrounds"
            )

        (
            coil_latent,
            package_latent,
            coil_pair,
            coil_package,
            length_scale,
        ) = _tensor_latent(self.port_artifact, scene, frequency_hz)
        (
            conductor_ids,
            conductor_arc,
            conductor_xy,
            conductor_weights,
        ) = _normalization_rule(
            scene,
            longitudinal_points=self.conductor_longitudinal_points,
            radial_order=self.conductor_radial_order,
            angular_order=self.conductor_angular_order,
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
                axial_order=self.package_axial_order,
                radial_order=self.package_radial_order,
                azimuthal_order=self.package_azimuthal_order,
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

        (
            background_segments,
            background_anchor_positions,
            background_anchor_radii,
        ) = scene_conductor_geometry(
            scene,
            segments_per_turn=self.background_segments_per_turn,
        )
        if conductivity > 0.0:
            background_points, background_weights = unbounded_background_quadrature(
                scene,
                background_segments,
                background_anchor_positions,
                background_anchor_radii,
                radial_order=self.background_radial_order,
                angular_order=self.background_angular_order,
            )
            (
                background_coil_coordinates,
                background_package_coordinates,
            ) = background_coordinate_features(
                scene,
                background_points,
                length_scale=length_scale,
            )
        else:
            background_points = np.empty((0, 3), dtype=float)
            background_weights = np.empty(0, dtype=float)
            background_coil_coordinates = np.empty((0, n_coils, 5), dtype=float)
            background_package_coordinates = np.empty(
                (0, len(scene.packages), 5),
                dtype=float,
            )

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
                package_gate = torch.as_tensor(
                    _package_loss_gate(scene, frequency_hz, package_ids),
                    dtype=raw_package.real.dtype,
                    device=raw_package.device,
                )
                raw_package = raw_package * package_gate[:, None, None]

            if len(background_points):
                raw_background = self.model.background.raw_matrices(
                    coil_latent,
                    package_latent,
                    background_coil_coordinates,
                    background_package_coordinates,
                )
                raw_background = raw_background * float(
                    background_loss_gate(scene, frequency_hz)
                )
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

        integrated = np.zeros_like(
            prediction.dissipation_channels,
            dtype=complex,
        )
        conductor_values_np = conductor_values.detach().cpu().numpy()
        for coil in range(n_coils):
            mask = conductor_ids == coil
            integrated[coil] = np.sum(
                conductor_weights[mask, None, None] * conductor_values_np[mask],
                axis=0,
            )
        if len(package_weights):
            integrated[n_coils] += np.sum(
                package_weights[:, None, None]
                * package_values.detach().cpu().numpy(),
                axis=0,
            )
        if len(background_weights):
            integrated[n_coils] += np.sum(
                background_weights[:, None, None]
                * background_values.detach().cpu().numpy(),
                axis=0,
            )
        closure = float(
            np.linalg.norm(integrated - prediction.dissipation_channels)
            / max(np.linalg.norm(prediction.dissipation_channels), 1e-30)
        )
        return PreparedHybridSpatialLossField(
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
            background_segments,
            background_anchor_positions,
            background_anchor_radii,
            float(length_scale),
            closure,
            self.device,
        )

    def save(self, path):
        torch.save(
            {
                "schema": TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA,
                "port_fingerprint": self.port_fingerprint,
                "model_config": {
                    "hidden_dim": self.model.hidden_dim,
                    "coil_pair_dim": self.model.coil_pair_dim,
                    "cross_dim": self.model.cross_dim,
                    "field_hidden_dim": self.model.field_hidden_dim,
                    "factor_rank": self.model.factor_rank,
                    "depth": self.model.depth,
                },
                "model_state": self.model.state_dict(),
                "conductor_longitudinal_points": self.conductor_longitudinal_points,
                "conductor_radial_order": self.conductor_radial_order,
                "conductor_angular_order": self.conductor_angular_order,
                "package_axial_order": self.package_axial_order,
                "package_radial_order": self.package_radial_order,
                "package_azimuthal_order": self.package_azimuthal_order,
                "background_segments_per_turn": self.background_segments_per_turn,
                "background_radial_order": self.background_radial_order,
                "background_angular_order": self.background_angular_order,
            },
            Path(path),
        )

    @staticmethod
    def load(path, port_artifact, *, device: str = "cpu"):
        resolved_device = resolve_torch_device(device)
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )
        if int(payload.get("schema", -1)) != TENSOR_HYBRID_SPATIAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported tensor hybrid spatial artifact schema")
        if payload.get("port_fingerprint") != tensor_port_fingerprint(port_artifact):
            raise ValueError("tensor spatial port fingerprint mismatch")
        dtype = next(port_artifact.model.parameters()).dtype
        if resolved_device == "mps" and dtype == torch.float64:
            raise ValueError(
                "this tensor spatial artifact uses float64 and cannot be loaded on MPS"
            )
        model = HybridSpatialLossShapeNet(**payload["model_config"]).to(dtype=dtype)
        model.load_state_dict(payload["model_state"])
        model.eval()
        return TensorHybridSpatialLossArtifact(
            port_artifact,
            model,
            conductor_longitudinal_points=int(
                payload["conductor_longitudinal_points"]
            ),
            conductor_radial_order=int(payload["conductor_radial_order"]),
            conductor_angular_order=int(payload["conductor_angular_order"]),
            package_axial_order=int(payload["package_axial_order"]),
            package_radial_order=int(payload["package_radial_order"]),
            package_azimuthal_order=int(payload["package_azimuthal_order"]),
            background_segments_per_turn=int(
                payload.get("background_segments_per_turn", 16)
            ),
            background_radial_order=int(payload.get("background_radial_order", 12)),
            background_angular_order=int(payload.get("background_angular_order", 48)),
            device=resolved_device,
        )


def _sample_loss(
    model,
    port_artifact,
    sample: TensorHybridSpatialTeacherSample,
    *,
    device: str,
    latent=None,
):
    if latent is None:
        latent = _tensor_latent(
            port_artifact,
            sample.scene,
            sample.frequency_hz,
        )
    (
        coil_latent,
        package_latent,
        coil_pair,
        coil_package,
        length_scale,
    ) = latent
    spatial_device = next(model.parameters()).device
    coil_latent = coil_latent.detach().to(spatial_device)
    package_latent = package_latent.detach().to(spatial_device)
    coil_pair = coil_pair.detach().to(spatial_device)
    coil_package = coil_package.detach().to(spatial_device)

    conductor = sample.conductor_spatial_loss
    conductor_coordinates = _coordinate_features(
        sample.scene,
        conductor.coil_index,
        conductor.arc_fraction,
        conductor.xy,
    )
    raw_conductor = model.conductor.raw_matrices(
        coil_latent,
        coil_pair,
        conductor.coil_index,
        conductor_coordinates,
    )
    conductor_transforms = _conductor_transforms(
        raw_conductor,
        conductor.coil_index,
        conductor.weights,
        sample.target_dissipation_channels[: len(sample.scene.coils)],
    )
    predicted_conductor = _apply_by_coil(
        raw_conductor,
        conductor.coil_index,
        conductor_transforms,
    )
    conductor_loss = _weighted_relative_loss(
        predicted_conductor,
        conductor.dissipation_matrix,
        conductor.weights,
    )

    package = sample.package_spatial_loss
    package_coordinates = _package_coordinate_features(
        sample.scene,
        package.package_index,
        package.local_position,
    )
    raw_package = model.package.raw_matrices(
        coil_latent,
        package_latent,
        coil_package,
        package.package_index,
        package_coordinates,
    )
    if len(package.package_index):
        package_gate = torch.as_tensor(
            _package_loss_gate(
                sample.scene,
                sample.frequency_hz,
                package.package_index,
            ),
            dtype=raw_package.real.dtype,
            device=raw_package.device,
        )
        raw_package = raw_package * package_gate[:, None, None]

    background = sample.background_spatial_loss
    if background is not None:
        root_pose = sample.scene.coils[0].geometry.pose
        background_world = root_pose.apply(background.root_local_position)
        (
            background_coil_coordinates,
            background_package_coordinates,
        ) = background_coordinate_features(
            sample.scene,
            background_world,
            length_scale=length_scale,
        )
        raw_background = model.background.raw_matrices(
            coil_latent,
            package_latent,
            background_coil_coordinates,
            background_package_coordinates,
        )
        raw_background = raw_background * float(
            background_loss_gate(sample.scene, sample.frequency_hz)
        )
        background_weights = background.weights
    else:
        n_ports = len(sample.scene.coils)
        raw_background = torch.empty(
            (0, n_ports, n_ports),
            dtype=raw_package.dtype,
            device=raw_package.device,
        )
        background_weights = np.empty(0, dtype=float)

    transform = _environment_transform(
        raw_package,
        package.weights,
        raw_background,
        background_weights,
        sample.target_dissipation_channels[len(sample.scene.coils)],
    )
    package_loss = _weighted_relative_loss(
        _apply_transform(raw_package, transform),
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
            _apply_transform(raw_background, transform),
            background.dissipation_matrix,
            background.weights,
        )
    return conductor_loss + package_loss + background_loss


def _sample_end_to_end_error(
    artifact: TensorHybridSpatialLossArtifact,
    sample: TensorHybridSpatialTeacherSample,
) -> float:
    prepared = artifact.prepare(sample.scene, sample.frequency_hz)
    conductor = sample.conductor_spatial_loss
    conductor_error = _weighted_relative_error_numpy(
        prepared.local_dissipation_matrices(
            conductor.coil_index,
            conductor.arc_fraction,
            conductor.xy,
        ),
        conductor.dissipation_matrix,
        conductor.weights,
    )
    package = sample.package_spatial_loss
    package_error = _weighted_relative_error_numpy(
        prepared.package_local_dissipation_matrices(
            package.package_index,
            package.local_position,
        ),
        package.dissipation_matrix,
        package.weights,
    )
    background_error = 0.0
    if sample.background_spatial_loss is not None:
        background = sample.background_spatial_loss
        root_pose = sample.scene.coils[0].geometry.pose
        world = root_pose.apply(background.root_local_position)
        background_error = _weighted_relative_error_numpy(
            prepared.background_dissipation_matrices(world),
            background.dissipation_matrix,
            background.weights,
        )
    return float(conductor_error + package_error + background_error)


def _legacy_train_tensor_hybrid_spatial_loss_surrogate(
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
    samples = tuple(samples)
    validation_samples = tuple(validation_samples)
    if not samples:
        raise ValueError("at least one tensor spatial training sample is required")
    if not bool(getattr(port_artifact, "supports_tensor_electric", False)):
        raise TypeError("tensor spatial training requires a tensor-aware port artifact")
    if any(
        not isinstance(sample, TensorHybridSpatialTeacherSample)
        for sample in samples + validation_samples
    ):
        raise TypeError(
            "tensor spatial trainer requires TensorHybridSpatialTeacherSample instances"
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
        raise ValueError("invalid tensor spatial training configuration")

    resolved_device = resolve_torch_device(device)
    for sample in samples + validation_samples:
        port_artifact.predict_structured(sample.scene, sample.frequency_hz)

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    port_model = port_artifact.model.to(resolved_device)
    port_artifact.device = resolved_device
    port_model.eval()
    for parameter in port_model.parameters():
        parameter.requires_grad_(False)
    model = HybridSpatialLossShapeNet(
        port_model.hidden_dim,
        port_model.coil_pair_dim,
        port_model.cross_dim,
        field_hidden_dim=field_hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
    ).to(
        device=resolved_device,
        dtype=next(port_model.parameters()).dtype,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    def cache_latent(sample):
        latent = _tensor_latent(
            port_artifact,
            sample.scene,
            sample.frequency_hz,
        )
        return (
            latent[0].detach().to(resolved_device),
            latent[1].detach().to(resolved_device),
            latent[2].detach().to(resolved_device),
            latent[3].detach().to(resolved_device),
            latent[4],
        )

    training_latents = tuple(cache_latent(sample) for sample in samples)
    validation_latents = tuple(cache_latent(sample) for sample in validation_samples)
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
        order = rng.permutation(len(samples))
        epoch_loss = 0.0
        for batch_start in range(0, len(order), int(batch_size)):
            batch_indices = order[batch_start : batch_start + int(batch_size)]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for raw_index in batch_indices:
                index = int(raw_index)
                loss = _sample_loss(
                    model,
                    port_artifact,
                    samples[index],
                    device=resolved_device,
                    latent=training_latents[index],
                )
                losses.append(loss)
                epoch_loss += float(loss.detach().cpu())
            batch_loss = torch.stack(losses).mean()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        final_loss = epoch_loss / len(samples)
        epochs_run = epoch

        if validation_samples:
            if epoch % int(validation_interval) != 0 and epoch != int(epochs):
                continue
            model.eval()
            with torch.no_grad():
                shape_score = float(
                    np.mean(
                        [
                            float(
                                _sample_loss(
                                    model,
                                    port_artifact,
                                    sample,
                                    device=resolved_device,
                                    latent=validation_latents[index],
                                ).detach().cpu()
                            )
                            for index, sample in enumerate(validation_samples)
                        ]
                    )
                )
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
                    [
                        _sample_end_to_end_error(probe, sample)
                        for sample in validation_samples
                    ]
                )
            )
            if (
                best_validation_error is None
                or score < best_validation_error - float(min_improvement)
            ):
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
        raise RuntimeError("tensor spatial training produced no selectable model state")
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
        None
        if best_validation_shape_error is None
        else float(best_validation_shape_error),
    )


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
    """Train tensor spatial FAST loss fields with optional true scene batching."""
    if int(batch_size) > 1 or str(device).strip().lower() == "auto":
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
    return _legacy_train_tensor_hybrid_spatial_loss_surrogate(
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
