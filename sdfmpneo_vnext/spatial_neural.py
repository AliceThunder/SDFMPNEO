from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.spatial_neural requires the 'neural' extra"
    ) from exc

from .basis import superellipse_section_quadrature
from .features import encode_scene_invariant
from .scene import Scene


SPATIAL_ARTIFACT_SCHEMA = 1


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int):
    layers = []
    width = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(width, hidden_dim), nn.SiLU()])
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def _psd_sqrt(matrix, *, inverse=False):
    matrix = 0.5 * (matrix + matrix.conj().transpose(-1, -2))
    values, vectors = torch.linalg.eigh(matrix)
    scale = torch.clamp(torch.max(torch.abs(values)), min=1e-20)
    if inverse:
        values = torch.clamp(values.real, min=1e-10 * scale + 1e-20)
        diagonal = torch.rsqrt(values)
    else:
        diagonal = torch.sqrt(torch.clamp(values.real, min=0.0))
    return (
        vectors
        @ torch.diag(diagonal.to(vectors.dtype))
        @ vectors.conj().transpose(-1, -2)
    )


class SpatialLossShapeNet(nn.Module):
    """Continuous local PSD field with shared per-port factor rows."""

    def __init__(
        self,
        *,
        node_dim: int = 17,
        pair_dim: int = 15,
        hidden_dim: int = 96,
        factor_rank: int = 3,
        depth: int = 3,
    ):
        super().__init__()
        if factor_rank < 1 or hidden_dim < 4 or depth < 1:
            raise ValueError("invalid spatial neural dimensions")
        self.node_dim = int(node_dim)
        self.pair_dim = int(pair_dim)
        self.hidden_dim = int(hidden_dim)
        self.factor_rank = int(factor_rank)
        self.depth = int(depth)
        self.input_dim = 2 * self.node_dim + self.pair_dim + 4
        self.row_net = _mlp(
            self.input_dim,
            self.hidden_dim,
            2 * self.factor_rank,
            self.depth,
        )

    def forward(self, row_features):
        """row_features shape (..., n_ports, input_dim)."""
        raw = self.row_net(row_features)
        real, imag = torch.chunk(raw, 2, dim=-1)
        factor = real + 1j * imag
        matrix = factor @ factor.conj().transpose(-1, -2)
        n_ports = row_features.shape[-2]
        eye = torch.eye(
            n_ports,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        scale = torch.clamp(
            torch.mean(torch.abs(matrix), dim=(-2, -1), keepdim=True),
            min=1.0,
        )
        matrix = matrix + 1e-8 * scale * eye
        return 0.5 * (matrix + matrix.conj().transpose(-1, -2))


def _query_row_features(
    encoded,
    normalizer,
    coil_index,
    arc_fraction,
    xy,
    scene,
):
    node, pair = normalizer.normalize(encoded)
    n_ports = len(scene.coils)
    if not 0 <= coil_index < n_ports:
        raise IndexError("coil_index out of range")
    geometry = scene.coils[coil_index].geometry
    local = np.asarray(xy, dtype=float)
    if local.shape != (2,):
        raise ValueError("xy must have shape (2,)")
    xn = local[0] / (0.5 * geometry.conductor_width)
    yn = local[1] / (0.5 * geometry.conductor_thickness)
    theta = 2.0 * np.pi * float(arc_fraction)
    local_features = np.asarray(
        [xn, yn, np.sin(theta), np.cos(theta)],
        dtype=float,
    )
    rows = []
    for port in range(n_ports):
        rows.append(
            np.concatenate(
                (
                    node[coil_index],
                    node[port],
                    pair[coil_index, port],
                    local_features,
                )
            )
        )
    return np.stack(rows, axis=0)


def _normalize_fields(raw, weights, coil_index, channels):
    # The neural field may train in float32, but passivity/closure normalization
    # is physical linear algebra and is deliberately evaluated in complex128.
    raw = raw.to(torch.complex128)
    weights = weights.to(torch.float64)
    channels = channels.to(torch.complex128)
    out = torch.empty_like(raw)
    n_coils = channels.shape[0]
    for coil in range(n_coils):
        mask = coil_index == coil
        if not torch.any(mask):
            raise ValueError("spatial samples do not cover every coil")
        local_raw = raw[mask]
        local_weights = weights[mask]
        integrated = torch.sum(
            local_weights[:, None, None] * local_raw,
            dim=0,
        )
        transform = (
            _psd_sqrt(channels[coil], inverse=False)
            @ _psd_sqrt(integrated, inverse=True)
        )
        corrected = (
            transform[None, :, :]
            @ local_raw
            @ transform.conj().transpose(-1, -2)[None, :, :]
        )
        out[mask] = 0.5 * (
            corrected + corrected.conj().transpose(-1, -2)
        )
    return out


@dataclass(frozen=True)
class SpatialTrainingReport:
    final_loss: float
    epochs: int
    samples: int
    best_epoch: int
    best_validation_error: float | None
    stopped_early: bool


@dataclass(frozen=True)
class PreparedNeuralLossField:
    scene: Scene
    frequency_hz: float
    artifact: object
    prediction: object
    encoded: object
    transforms: tuple
    integrated_channels: np.ndarray
    normalization_closure_error: float

    @property
    def port_prediction(self):
        return self.prediction

    def local_dissipation_matrices(
        self,
        coil_index,
        arc_fraction,
        xy,
    ) -> np.ndarray:
        coil_index = np.asarray(coil_index, dtype=int)
        arc_fraction = np.asarray(arc_fraction, dtype=float)
        xy = np.asarray(xy, dtype=float)
        if coil_index.ndim != 1:
            raise ValueError("coil_index must be one-dimensional")
        n_query = len(coil_index)
        if arc_fraction.shape != (n_query,) or xy.shape != (n_query, 2):
            raise ValueError("spatial query arrays have incompatible shapes")
        if np.any((arc_fraction < 0.0) | (arc_fraction > 1.0)):
            raise ValueError("arc_fraction must lie in [0,1]")
        if np.any((coil_index < 0) | (coil_index >= len(self.scene.coils))):
            raise IndexError("coil_index out of range")

        n_ports = len(self.scene.coils)
        out = np.zeros((n_query, n_ports, n_ports), dtype=complex)
        inside = np.zeros(n_query, dtype=bool)
        features = []
        feature_indices = []
        for index in range(n_query):
            coil = int(coil_index[index])
            geometry = self.scene.coils[coil].geometry
            local = xy[index]
            a = 0.5 * geometry.conductor_width
            b = 0.5 * geometry.conductor_thickness
            m = geometry.cross_section_exponent
            if (
                (abs(local[0]) / a) ** m
                + (abs(local[1]) / b) ** m
                <= 1.0 + 1e-12
            ):
                inside[index] = True
                feature_indices.append(index)
                features.append(
                    _query_row_features(
                        self.encoded,
                        self.artifact.port_artifact.normalizer,
                        coil,
                        float(arc_fraction[index]),
                        local,
                        self.scene,
                    )
                )

        if not features:
            return out

        dtype = next(self.artifact.model.parameters()).dtype
        device = next(self.artifact.model.parameters()).device
        with torch.no_grad():
            raw = self.artifact.model(
                torch.as_tensor(
                    np.asarray(features, dtype=float),
                    dtype=dtype,
                    device=device,
                )
            ).to(torch.complex128)
            corrected = []
            for local_index, query_index in enumerate(feature_indices):
                transform = self.transforms[int(coil_index[query_index])]
                matrix = (
                    transform
                    @ raw[local_index]
                    @ transform.conj().T
                )
                corrected.append(
                    0.5 * (matrix + matrix.conj().T)
                )
            corrected = torch.stack(corrected).detach().cpu().numpy()

        out[np.asarray(feature_indices, dtype=int)] = corrected
        return out

    def local_dissipation_matrix(
        self,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self.local_dissipation_matrices(
            np.asarray([coil_index], dtype=int),
            np.asarray([arc_fraction], dtype=float),
            np.asarray([xy], dtype=float),
        )[0]

    def local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        matrix = self.local_dissipation_matrix(
            coil_index,
            arc_fraction,
            xy,
        )
        currents = np.asarray(currents, dtype=complex)
        return float(
            0.5
            * np.real(
                np.vdot(
                    currents,
                    matrix @ currents,
                )
            )
        )


class SpatialLossArtifact:
    """Continuous FAST Joule field with exact channel-integral closure."""

    def __init__(
        self,
        model: SpatialLossShapeNet,
        port_artifact,
        *,
        normalization_segments: int = 24,
        radial_order: int = 4,
        angular_order: int = 24,
        device: str = "cpu",
    ):
        if not hasattr(port_artifact, "predict_structured"):
            raise TypeError("port_artifact must expose predict_structured")
        if not hasattr(port_artifact, "normalizer"):
            raise TypeError("spatial neural artifact requires a normalized neural port artifact")
        self.model = model.to(device)
        self.port_artifact = port_artifact
        self.normalization_segments = int(normalization_segments)
        self.radial_order = int(radial_order)
        self.angular_order = int(angular_order)
        self.device = str(device)

    def _normalization_grid(self, scene: Scene, frequency_hz: float):
        encoded = encode_scene_invariant(scene, frequency_hz)
        features = []
        weights = []
        coils = []
        for coil_index, coil in enumerate(scene.coils):
            geometry = coil.geometry
            poly = geometry.polyline(self.normalization_segments)
            section = superellipse_section_quadrature(
                geometry.conductor_width,
                geometry.conductor_thickness,
                geometry.cross_section_exponent,
                self.radial_order,
                self.angular_order,
            )
            for segment_index, length in enumerate(poly.lengths):
                arc = (segment_index + 0.5) / len(poly.lengths)
                for xy, weight in zip(section.xy, section.weights):
                    features.append(
                        _query_row_features(
                            encoded,
                            self.port_artifact.normalizer,
                            coil_index,
                            arc,
                            xy,
                            scene,
                        )
                    )
                    weights.append(float(length * weight))
                    coils.append(coil_index)
        return (
            encoded,
            np.asarray(features, dtype=float),
            np.asarray(weights, dtype=float),
            np.asarray(coils, dtype=int),
        )

    def _transforms(self, scene: Scene, frequency_hz: float):
        encoded, features, weights, coils = self._normalization_grid(
            scene, frequency_hz
        )
        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device
        with torch.no_grad():
            raw = self.model(
                torch.as_tensor(features, dtype=dtype, device=device)
            ).to(torch.complex128)
        prediction = self.port_artifact.predict_structured(scene, frequency_hz)
        channels = torch.as_tensor(
            prediction.dissipation_channels,
            dtype=torch.complex128,
            device=device,
        )
        weight_t = torch.as_tensor(weights, dtype=torch.float64, device=device)
        coil_t = torch.as_tensor(coils, dtype=torch.long, device=device)
        transforms = []
        for coil in range(len(scene.coils)):
            mask = coil_t == coil
            integrated = torch.sum(
                weight_t[mask, None, None] * raw[mask],
                dim=0,
            )
            transforms.append(
                _psd_sqrt(channels[coil], inverse=False)
                @ _psd_sqrt(integrated, inverse=True)
            )
        return encoded, prediction, tuple(transforms), raw, weights, coils

    def prepare(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> PreparedNeuralLossField:
        (
            encoded,
            prediction,
            transforms,
            raw,
            weights,
            coils,
        ) = self._transforms(
            scene,
            frequency_hz,
        )
        dtype = next(self.model.parameters()).dtype
        device = next(self.model.parameters()).device
        integrated = []
        for coil in range(len(scene.coils)):
            mask_np = coils == coil
            mask = torch.as_tensor(
                mask_np,
                dtype=torch.bool,
                device=device,
            )
            local = raw[mask]
            transform = transforms[coil]
            corrected = (
                transform[None, :, :]
                @ local
                @ transform.conj().T[None, :, :]
            )
            w = torch.as_tensor(
                weights[mask_np],
                dtype=torch.float64,
                device=device,
            )
            integrated.append(
                torch.sum(
                    w[:, None, None]
                    * corrected,
                    dim=0,
                )
            )
        integrated = (
            torch.stack(
                integrated
            )
            .detach()
            .cpu()
            .numpy()
        )
        expected = np.asarray(
            prediction.dissipation_channels,
            dtype=complex,
        )
        closure = float(
            np.linalg.norm(
                integrated
                - expected
            )
            / max(
                np.linalg.norm(
                    expected
                ),
                1e-30,
            )
        )
        return PreparedNeuralLossField(
            scene,
            float(
                frequency_hz
            ),
            self,
            prediction,
            encoded,
            transforms,
            integrated,
            closure,
        )

    def local_dissipation_matrix(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy=(0.0, 0.0),
    ) -> np.ndarray:
        return self.prepare(
            scene,
            frequency_hz,
        ).local_dissipation_matrix(
            coil_index,
            arc_fraction,
            xy,
        )

    def local_joule_density(
        self,
        scene: Scene,
        frequency_hz: float,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        return self.prepare(
            scene,
            frequency_hz,
        ).local_joule_density(
            coil_index,
            arc_fraction,
            xy,
            currents,
        )

    def integrated_channels(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> np.ndarray:
        return np.asarray(
            self.prepare(
                scene,
                frequency_hz,
            ).integrated_channels,
            dtype=complex,
        )

    def save(self, path):
        payload = {
            "schema": SPATIAL_ARTIFACT_SCHEMA,
            "model_config": {
                "node_dim": self.model.node_dim,
                "pair_dim": self.model.pair_dim,
                "hidden_dim": self.model.hidden_dim,
                "factor_rank": self.model.factor_rank,
                "depth": self.model.depth,
            },
            "model_state": self.model.state_dict(),
            "normalization_segments": self.normalization_segments,
            "radial_order": self.radial_order,
            "angular_order": self.angular_order,
            "port_fingerprint": self.port_artifact.fingerprint(),
        }
        torch.save(payload, Path(path))

    @staticmethod
    def load(path, port_artifact, *, device="cpu"):
        try:
            payload = torch.load(Path(path), map_location=device, weights_only=False)
        except TypeError:
            payload = torch.load(Path(path), map_location=device)
        if payload.get("schema") != SPATIAL_ARTIFACT_SCHEMA:
            raise ValueError("unsupported spatial artifact schema")
        if payload.get("port_fingerprint") != port_artifact.fingerprint():
            raise ValueError("spatial artifact was trained against a different port artifact")
        model = SpatialLossShapeNet(**payload["model_config"])
        model.load_state_dict(payload["model_state"])
        model.eval()
        return SpatialLossArtifact(
            model,
            port_artifact,
            normalization_segments=int(payload["normalization_segments"]),
            radial_order=int(payload["radial_order"]),
            angular_order=int(payload["angular_order"]),
            device=device,
        )


def _sample_loss(artifact: SpatialLossArtifact, sample):
    spatial = sample.spatial_loss
    if spatial is None or sample.target_dissipation_channels is None:
        raise ValueError("spatial training requires spatial and channel truth")
    features = np.stack(
        [
            _query_row_features(
                sample.encoded,
                artifact.port_artifact.normalizer,
                int(coil),
                float(arc),
                xy,
                sample.scene,
            )
            for coil, arc, xy in zip(
                spatial.coil_index,
                spatial.arc_fraction,
                spatial.xy,
            )
        ],
        axis=0,
    )
    dtype = next(artifact.model.parameters()).dtype
    device = next(artifact.model.parameters()).device
    raw = artifact.model(
        torch.as_tensor(features, dtype=dtype, device=device)
    )
    channels = torch.as_tensor(
        sample.target_dissipation_channels,
        dtype=torch.complex128,
        device=device,
    )
    weights = torch.as_tensor(
        spatial.weights,
        dtype=torch.float64,
        device=device,
    )
    coil_index = torch.as_tensor(
        spatial.coil_index, dtype=torch.long, device=device
    )
    predicted = _normalize_fields(raw, weights, coil_index, channels)
    target = torch.as_tensor(
        spatial.dissipation_matrix,
        dtype=torch.complex128,
        device=device,
    )
    error = torch.sum(weights[:, None, None] * torch.abs(predicted - target) ** 2)
    denom = (
        torch.sum(weights[:, None, None] * torch.abs(target) ** 2) + 1e-30
    )
    return error / denom


def _inference_sample_error(
    artifact: SpatialLossArtifact,
    sample,
) -> float:
    """Validation error under the exact online normalization semantics."""
    spatial = sample.spatial_loss
    if spatial is None:
        raise ValueError(
            "spatial validation requires spatial_loss truth"
        )
    prepared = artifact.prepare(
        sample.scene,
        sample.frequency_hz,
    )
    predicted = np.asarray(
        prepared.local_dissipation_matrices(
            spatial.coil_index,
            spatial.arc_fraction,
            spatial.xy,
        ),
        dtype=complex,
    )
    target = np.asarray(
        spatial.dissipation_matrix,
        dtype=complex,
    )
    weights = np.asarray(
        spatial.weights,
        dtype=float,
    )
    numerator = float(
        np.sum(
            weights[:, None, None]
            * np.abs(
                predicted - target
            ) ** 2
        )
    )
    denominator = max(
        float(
            np.sum(
                weights[:, None, None]
                * np.abs(target) ** 2
            )
        ),
        1e-30,
    )
    return float(
        np.sqrt(
            numerator / denominator
        )
    )


def train_spatial_loss_surrogate(
    port_artifact,
    samples,
    *,
    validation_samples=(),
    hidden_dim=96,
    factor_rank=3,
    depth=3,
    epochs=120,
    learning_rate=1e-3,
    weight_decay=1e-6,
    patience=20,
    seed=23,
    device="cpu",
):
    samples = tuple(samples)
    validation_samples = tuple(validation_samples)
    if not samples:
        raise ValueError("at least one spatial teacher sample is required")
    if any(sample.spatial_loss is None for sample in samples + validation_samples):
        raise ValueError("spatial samples are missing spatial_loss truth")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SpatialLossShapeNet(
        node_dim=samples[0].encoded.node_features.shape[-1],
        pair_dim=samples[0].encoded.pair_features.shape[-1],
        hidden_dim=hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
    ).to(device)
    artifact = SpatialLossArtifact(model, port_artifact, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    best_state = None
    best_epoch = 0
    best_validation = None
    stale = 0
    stopped = False
    final_loss = np.inf

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for index in np.random.permutation(len(samples)):
            optimizer.zero_grad(set_to_none=True)
            loss = _sample_loss(artifact, samples[int(index)])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach().cpu())
        final_loss = total / len(samples)

        if validation_samples:
            model.eval()
            score = float(
                np.mean(
                    [
                        _inference_sample_error(
                            artifact,
                            sample,
                        )
                        for sample
                        in validation_samples
                    ]
                )
            )
            if best_validation is None or score < best_validation - 1e-6:
                best_validation = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    stopped = True
                    break
        else:
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("spatial training produced no selectable model")
    model.load_state_dict(best_state)
    model.eval()
    return (
        artifact,
        SpatialTrainingReport(
            float(final_loss),
            int(epoch),
            len(samples),
            int(best_epoch),
            None if best_validation is None else float(best_validation),
            bool(stopped),
        ),
    )


# Backward-compatible public names used by the existing vNext audit CLI.
NeuralSpatialLossArtifact = SpatialLossArtifact
PreparedNeuralSpatialLossField = PreparedNeuralLossField
