from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("generation-2 port surrogate requires the 'neural' extra") from exc

from .analytic_baseline import analytic_port_baseline
from .device import resolve_torch_device
from .generation2_features import (
    GENERATION2_COIL_FEATURE_DIM,
    GENERATION2_CROSS_FEATURE_DIM,
    GENERATION2_FEATURE_SCHEMA,
    GENERATION2_PACKAGE_FEATURE_DIM,
    GENERATION2_PAIR_FEATURE_DIM,
    EncodedGeneration2Scene,
    encode_generation2_scene,
)
from .hybrid_domain import (
    validate_hybrid_geometry_domain,
    validate_package_conductor_topology,
)
from .hybrid_neural import _dielectric_loss_gate, _reactance_gate
from .prediction import StructuredPortPrediction
from .scene import Scene
from .tensor_neural import _range_contains, _sample_tensor_ranges


GENERATION2_PORT_ARTIFACT_SCHEMA = 1
GENERATION2_PORT_MODEL_GENERATION = 2


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int):
    layers = []
    width = int(input_dim)
    for _ in range(int(depth)):
        layers.extend((nn.Linear(width, hidden_dim), nn.SiLU()))
        width = int(hidden_dim)
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


def _flatten_feature_rows(encodings, getter, width: int) -> np.ndarray:
    rows = []
    for encoded in encodings:
        value = np.asarray(getter(encoded), dtype=float)
        if value.shape[-1] != int(width):
            raise ValueError("generation-2 feature width changed inside one dataset")
        if value.size:
            rows.append(value.reshape(-1, int(width)))
    if not rows:
        return np.empty((0, int(width)), dtype=float)
    return np.concatenate(rows, axis=0)


def _stats(values: np.ndarray, width: int, floor: float):
    values = np.asarray(values, dtype=float).reshape(-1, int(width))
    if not len(values):
        return np.zeros(width, dtype=float), np.ones(width, dtype=float)
    return values.mean(axis=0), np.maximum(values.std(axis=0), float(floor))


@dataclass(frozen=True)
class Generation2Normalizer:
    coil_mean: np.ndarray
    coil_scale: np.ndarray
    coil_pair_mean: np.ndarray
    coil_pair_scale: np.ndarray
    package_mean: np.ndarray
    package_scale: np.ndarray
    cross_mean: np.ndarray
    cross_scale: np.ndarray
    package_pair_mean: np.ndarray
    package_pair_scale: np.ndarray
    reactance_scale: float

    @staticmethod
    def fit(samples, *, floor: float = 1e-8) -> "Generation2Normalizer":
        samples = tuple(samples)
        if not samples:
            raise ValueError("generation-2 normalizer requires training samples")
        if not np.isfinite(floor) or float(floor) <= 0.0:
            raise ValueError("normalization floor must be positive and finite")
        encoded = tuple(
            encode_generation2_scene(sample.scene, sample.frequency_hz)
            for sample in samples
        )
        coil_mean, coil_scale = _stats(
            _flatten_feature_rows(
                encoded,
                lambda value: value.coil.node_features,
                GENERATION2_COIL_FEATURE_DIM,
            ),
            GENERATION2_COIL_FEATURE_DIM,
            floor,
        )
        coil_pair_mean, coil_pair_scale = _stats(
            _flatten_feature_rows(
                encoded,
                lambda value: value.coil.pair_features,
                GENERATION2_PAIR_FEATURE_DIM,
            ),
            GENERATION2_PAIR_FEATURE_DIM,
            floor,
        )
        package_mean, package_scale = _stats(
            _flatten_feature_rows(
                encoded,
                lambda value: value.package_features,
                GENERATION2_PACKAGE_FEATURE_DIM,
            ),
            GENERATION2_PACKAGE_FEATURE_DIM,
            floor,
        )
        cross_mean, cross_scale = _stats(
            _flatten_feature_rows(
                encoded,
                lambda value: value.coil_package_features,
                GENERATION2_CROSS_FEATURE_DIM,
            ),
            GENERATION2_CROSS_FEATURE_DIM,
            floor,
        )
        package_pair_mean, package_pair_scale = _stats(
            _flatten_feature_rows(
                encoded,
                lambda value: value.package_pair_features,
                GENERATION2_PAIR_FEATURE_DIM,
            ),
            GENERATION2_PAIR_FEATURE_DIM,
            floor,
        )
        reactance_residual = np.concatenate(
            [
                (
                    np.asarray(sample.target_impedance, dtype=complex).imag
                    - np.asarray(sample.baseline_reactance, dtype=float)
                ).ravel()
                for sample in samples
            ]
        )
        reactance_scale = max(
            float(np.sqrt(np.mean(reactance_residual**2))),
            float(floor),
        )
        return Generation2Normalizer(
            coil_mean,
            coil_scale,
            coil_pair_mean,
            coil_pair_scale,
            package_mean,
            package_scale,
            cross_mean,
            cross_scale,
            package_pair_mean,
            package_pair_scale,
            reactance_scale,
        )

    def normalize(self, encoded: EncodedGeneration2Scene):
        return (
            (encoded.coil.node_features - self.coil_mean) / self.coil_scale,
            (encoded.coil.pair_features - self.coil_pair_mean) / self.coil_pair_scale,
            (encoded.package_features - self.package_mean) / self.package_scale,
            (encoded.coil_package_features - self.cross_mean) / self.cross_scale,
            (encoded.package_pair_features - self.package_pair_mean)
            / self.package_pair_scale,
        )

    def to_dict(self):
        return {
            "coil_mean": self.coil_mean,
            "coil_scale": self.coil_scale,
            "coil_pair_mean": self.coil_pair_mean,
            "coil_pair_scale": self.coil_pair_scale,
            "package_mean": self.package_mean,
            "package_scale": self.package_scale,
            "cross_mean": self.cross_mean,
            "cross_scale": self.cross_scale,
            "package_pair_mean": self.package_pair_mean,
            "package_pair_scale": self.package_pair_scale,
            "reactance_scale": self.reactance_scale,
        }

    @staticmethod
    def from_dict(payload):
        return Generation2Normalizer(
            np.asarray(payload["coil_mean"], dtype=float),
            np.asarray(payload["coil_scale"], dtype=float),
            np.asarray(payload["coil_pair_mean"], dtype=float),
            np.asarray(payload["coil_pair_scale"], dtype=float),
            np.asarray(payload["package_mean"], dtype=float),
            np.asarray(payload["package_scale"], dtype=float),
            np.asarray(payload["cross_mean"], dtype=float),
            np.asarray(payload["cross_scale"], dtype=float),
            np.asarray(payload["package_pair_mean"], dtype=float),
            np.asarray(payload["package_pair_scale"], dtype=float),
            float(payload["reactance_scale"]),
        )


class Generation2PortNet(nn.Module):
    """Repeated-interaction tensor graph with structure-preserving Port heads."""

    def __init__(
        self,
        *,
        coil_dim: int = GENERATION2_COIL_FEATURE_DIM,
        coil_pair_dim: int = GENERATION2_PAIR_FEATURE_DIM,
        package_dim: int = GENERATION2_PACKAGE_FEATURE_DIM,
        cross_dim: int = GENERATION2_CROSS_FEATURE_DIM,
        package_pair_dim: int = GENERATION2_PAIR_FEATURE_DIM,
        hidden_dim: int = 64,
        factor_rank: int = 4,
        depth: int = 2,
        interaction_rounds: int = 3,
        resistance_log_limit: float = 4.0,
    ):
        super().__init__()
        if hidden_dim < 4 or factor_rank < 1 or depth < 1 or interaction_rounds < 1:
            raise ValueError("invalid generation-2 Port model dimensions")
        if not np.isfinite(resistance_log_limit) or resistance_log_limit <= 0.0:
            raise ValueError("resistance_log_limit must be positive and finite")
        self.coil_dim = int(coil_dim)
        self.coil_pair_dim = int(coil_pair_dim)
        self.package_dim = int(package_dim)
        self.cross_dim = int(cross_dim)
        self.package_pair_dim = int(package_pair_dim)
        self.hidden_dim = int(hidden_dim)
        self.factor_rank = int(factor_rank)
        self.depth = int(depth)
        self.interaction_rounds = int(interaction_rounds)
        self.resistance_log_limit = float(resistance_log_limit)

        h = self.hidden_dim
        self.coil_encoder = _mlp(self.coil_dim, h, h, self.depth)
        self.package_encoder = _mlp(self.package_dim, h, h, self.depth)
        self.coil_pair_encoder = _mlp(2 * h + self.coil_pair_dim, h, h, self.depth)
        self.package_pair_encoder = _mlp(
            2 * h + self.package_pair_dim,
            h,
            h,
            self.depth,
        )
        self.coil_to_package_encoder = _mlp(
            2 * h + self.cross_dim,
            h,
            h,
            self.depth,
        )
        self.package_to_coil_encoder = _mlp(
            2 * h + self.cross_dim,
            h,
            h,
            self.depth,
        )
        self.coil_update = _mlp(3 * h, h, h, self.depth)
        self.package_update = _mlp(3 * h, h, h, self.depth)
        self.coil_norm = nn.LayerNorm(h)
        self.package_norm = nn.LayerNorm(h)

        self.resistance_log_diag_head = nn.Linear(h, 1)
        self.resistance_log_pair_head = _mlp(
            2 * h + self.coil_pair_dim,
            h,
            1,
            self.depth,
        )
        self.reactance_diag_head = nn.Linear(h, 1)
        self.reactance_pair_head = _mlp(
            2 * h + self.coil_pair_dim,
            h,
            1,
            self.depth,
        )
        self.conductor_channel_head = _mlp(
            2 * h + self.coil_pair_dim,
            h,
            2 * self.factor_rank,
            self.depth,
        )
        self.environment_channel_head = _mlp(
            2 * h + self.cross_dim,
            h,
            2 * self.factor_rank,
            self.depth,
        )


def _aggregate(messages, reference):
    if not messages:
        return torch.zeros_like(reference)
    return torch.stack(messages, dim=0).sum(dim=0) / math.sqrt(len(messages))


def generation2_batched_latent(
    model: Generation2PortNet,
    coil_features,
    coil_pair_features,
    package_features,
    coil_package_features,
    package_pair_features,
):
    coil = model.coil_encoder(coil_features)
    package = model.package_encoder(package_features)
    _, n_coils, _ = coil.shape
    _, n_packages, _ = package.shape

    for _ in range(model.interaction_rounds):
        coil_pair_messages = []
        for i in range(n_coils):
            messages = []
            for j in range(n_coils):
                if i == j:
                    continue
                messages.append(
                    model.coil_pair_encoder(
                        torch.cat(
                            (coil[:, i], coil[:, j], coil_pair_features[:, i, j]),
                            dim=-1,
                        )
                    )
                )
            coil_pair_messages.append(_aggregate(messages, coil[:, i]))

        package_pair_messages = []
        coil_to_package_messages = []
        for package_index in range(n_packages):
            pair_messages = []
            for other in range(n_packages):
                if package_index == other:
                    continue
                pair_messages.append(
                    model.package_pair_encoder(
                        torch.cat(
                            (
                                package[:, package_index],
                                package[:, other],
                                package_pair_features[:, package_index, other],
                            ),
                            dim=-1,
                        )
                    )
                )
            package_pair_messages.append(
                _aggregate(pair_messages, package[:, package_index])
            )
            coil_messages = []
            for coil_index in range(n_coils):
                coil_messages.append(
                    model.coil_to_package_encoder(
                        torch.cat(
                            (
                                package[:, package_index],
                                coil[:, coil_index],
                                coil_package_features[:, coil_index, package_index],
                            ),
                            dim=-1,
                        )
                    )
                )
            coil_to_package_messages.append(
                _aggregate(coil_messages, package[:, package_index])
            )

        package_to_coil_messages = []
        for coil_index in range(n_coils):
            messages = []
            for package_index in range(n_packages):
                messages.append(
                    model.package_to_coil_encoder(
                        torch.cat(
                            (
                                coil[:, coil_index],
                                package[:, package_index],
                                coil_package_features[:, coil_index, package_index],
                            ),
                            dim=-1,
                        )
                    )
                )
            package_to_coil_messages.append(_aggregate(messages, coil[:, coil_index]))

        new_coil = []
        for coil_index in range(n_coils):
            update = model.coil_update(
                torch.cat(
                    (
                        coil[:, coil_index],
                        coil_pair_messages[coil_index],
                        package_to_coil_messages[coil_index],
                    ),
                    dim=-1,
                )
            )
            new_coil.append(model.coil_norm(coil[:, coil_index] + update))
        coil = torch.stack(new_coil, dim=1)

        if n_packages:
            new_package = []
            for package_index in range(n_packages):
                update = model.package_update(
                    torch.cat(
                        (
                            package[:, package_index],
                            package_pair_messages[package_index],
                            coil_to_package_messages[package_index],
                        ),
                        dim=-1,
                    )
                )
                new_package.append(
                    model.package_norm(package[:, package_index] + update)
                )
            package = torch.stack(new_package, dim=1)

    return coil, package


def _batched_hermitian_sqrt(matrix, *, inverse: bool = False):
    matrix = 0.5 * (matrix + matrix.conj().transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    real_dtype = matrix.real.dtype if torch.is_complex(matrix) else matrix.dtype
    scale = torch.clamp(
        torch.amax(torch.abs(eigenvalues), dim=-1, keepdim=True),
        min=torch.finfo(real_dtype).tiny,
    )
    floor = torch.finfo(real_dtype).eps * 32.0 * scale
    clipped = torch.maximum(eigenvalues.real, floor)
    diagonal = torch.rsqrt(clipped) if inverse else torch.sqrt(clipped)
    return (
        eigenvectors
        @ torch.diag_embed(diagonal.to(eigenvectors.dtype))
        @ eigenvectors.conj().transpose(-1, -2)
    )


def _symmetric_matrix_exp(matrix, limit: float):
    matrix = 0.5 * (matrix + matrix.transpose(-1, -2))
    dtype = matrix.dtype
    tiny = torch.finfo(dtype).tiny
    norm = torch.sqrt(
        torch.sum(matrix * matrix, dim=(-2, -1), keepdim=True)
    )
    scale = torch.clamp(
        float(limit) / torch.clamp(norm, min=tiny),
        max=1.0,
    )
    return torch.matrix_exp(matrix * scale)


def _complex_factor(model: Generation2PortNet, raw):
    real = raw[..., : model.factor_rank]
    imag = raw[..., model.factor_rank :]
    dtype = torch.complex64 if raw.dtype == torch.float32 else torch.complex128
    return real.to(dtype) + 1j * imag.to(dtype)


def forward_generation2_port_batch(
    model: Generation2PortNet,
    coil_features,
    coil_pair_features,
    package_features,
    coil_package_features,
    package_pair_features,
    baseline_resistance,
    baseline_reactance,
    *,
    reactance_scale: float,
    dielectric_loss_gate,
    reactance_gate,
    return_latent: bool = False,
):
    coil, package = generation2_batched_latent(
        model,
        coil_features,
        coil_pair_features,
        package_features,
        coil_package_features,
        package_pair_features,
    )
    batch, n_ports, _ = coil.shape

    log_correction = torch.diag_embed(
        model.resistance_log_diag_head(coil).squeeze(-1)
    )
    for i in range(n_ports):
        for j in range(i):
            forward = model.resistance_log_pair_head(
                torch.cat((coil[:, i], coil[:, j], coil_pair_features[:, i, j]), dim=-1)
            ).squeeze(-1)
            reverse = model.resistance_log_pair_head(
                torch.cat((coil[:, j], coil[:, i], coil_pair_features[:, j, i]), dim=-1)
            ).squeeze(-1)
            value = 0.5 * (forward + reverse)
            pair = torch.zeros_like(log_correction)
            pair[:, i, j] = value
            pair[:, j, i] = value
            log_correction = log_correction + pair
    multiplier = _symmetric_matrix_exp(
        log_correction,
        model.resistance_log_limit,
    )
    baseline_factor = _batched_hermitian_sqrt(baseline_resistance)
    resistance = baseline_factor @ multiplier @ baseline_factor.transpose(-1, -2)
    resistance = 0.5 * (resistance + resistance.transpose(-1, -2))

    residual_reactance = torch.diag_embed(
        float(reactance_scale) * model.reactance_diag_head(coil).squeeze(-1)
    )
    for i in range(n_ports):
        for j in range(i):
            forward = model.reactance_pair_head(
                torch.cat((coil[:, i], coil[:, j], coil_pair_features[:, i, j]), dim=-1)
            ).squeeze(-1)
            reverse = model.reactance_pair_head(
                torch.cat((coil[:, j], coil[:, i], coil_pair_features[:, j, i]), dim=-1)
            ).squeeze(-1)
            value = 0.5 * float(reactance_scale) * (forward + reverse)
            pair = torch.zeros_like(residual_reactance)
            pair[:, i, j] = value
            pair[:, j, i] = value
            residual_reactance = residual_reactance + pair
    residual_reactance = residual_reactance * torch.as_tensor(
        reactance_gate,
        dtype=residual_reactance.dtype,
        device=residual_reactance.device,
    ).reshape(batch, 1, 1)
    reactance = baseline_reactance + residual_reactance
    reactance = 0.5 * (reactance + reactance.transpose(-1, -2))

    raw_channels = []
    for channel_index in range(n_ports):
        rows = []
        for port_index in range(n_ports):
            raw = model.conductor_channel_head(
                torch.cat(
                    (
                        coil[:, channel_index],
                        coil[:, port_index],
                        coil_pair_features[:, channel_index, port_index],
                    ),
                    dim=-1,
                )
            )
            rows.append(_complex_factor(model, raw))
        factor = torch.stack(rows, dim=1)
        raw_channels.append(factor @ factor.conj().transpose(-1, -2))

    n_packages = int(package.shape[1])
    package_pool = (
        torch.mean(package, dim=1)
        if n_packages
        else torch.zeros(
            (batch, model.hidden_dim),
            dtype=coil.dtype,
            device=coil.device,
        )
    )
    rows = []
    for port_index in range(n_ports):
        cross_summary = (
            torch.mean(coil_package_features[:, port_index], dim=1)
            if n_packages
            else torch.zeros(
                (batch, model.cross_dim),
                dtype=coil.dtype,
                device=coil.device,
            )
        )
        raw = model.environment_channel_head(
            torch.cat((package_pool, coil[:, port_index], cross_summary), dim=-1)
        )
        rows.append(_complex_factor(model, raw))
    environment_factor = torch.stack(rows, dim=1)
    environment_raw = torch.as_tensor(
        dielectric_loss_gate,
        dtype=resistance.dtype,
        device=resistance.device,
    ).reshape(batch, 1, 1) * (
        environment_factor @ environment_factor.conj().transpose(-1, -2)
    )
    raw_channels.append(environment_raw)
    raw_channels = torch.stack(raw_channels, dim=1)

    complex_dtype = torch.complex64 if resistance.dtype == torch.float32 else torch.complex128
    resistance_complex = resistance.to(complex_dtype)
    scale = torch.clamp(
        torch.diagonal(resistance, dim1=-2, dim2=-1).sum(dim=-1)
        / max(n_ports, 1),
        min=torch.finfo(resistance.dtype).tiny,
    )
    eye = torch.eye(n_ports, dtype=complex_dtype, device=resistance.device)
    jitter = 1e-8 * scale / max(n_ports, 1)
    channel_mask = torch.cat(
        (
            torch.ones(n_ports, dtype=resistance.dtype, device=resistance.device),
            torch.zeros(1, dtype=resistance.dtype, device=resistance.device),
        )
    )
    raw_channels = raw_channels + (
        jitter[:, None, None, None]
        * channel_mask[None, :, None, None]
        * eye[None, None, :, :]
    )
    raw_sum = torch.sum(raw_channels, dim=1)
    resistance_cholesky = torch.linalg.cholesky(resistance_complex)
    raw_cholesky = torch.linalg.cholesky(raw_sum)
    congruence = torch.linalg.solve(
        raw_cholesky.transpose(-1, -2),
        resistance_cholesky.transpose(-1, -2),
    ).transpose(-1, -2)
    channels = (
        congruence[:, None]
        @ raw_channels
        @ congruence.conj().transpose(-1, -2)[:, None]
    )
    channels = 0.5 * (channels + channels.conj().transpose(-1, -2))

    if return_latent:
        return resistance, reactance, channels, coil, package
    return resistance, reactance, channels


def _state_dtype(state_dict):
    for value in state_dict.values():
        if torch.is_floating_point(value):
            return value.dtype
    return torch.get_default_dtype()


class Generation2PortArtifact:
    artifact_schema = GENERATION2_PORT_ARTIFACT_SCHEMA
    model_generation = GENERATION2_PORT_MODEL_GENERATION
    supports_packages = True
    supports_lossy_background = True
    supports_tensor_electric = True

    def __init__(
        self,
        model: Generation2PortNet,
        normalizer: Generation2Normalizer,
        *,
        baseline_segments: int,
        material_domain,
        geometry_domain=None,
        partition_fingerprint: str | None = None,
        device: str = "cpu",
    ):
        if model.coil_dim != GENERATION2_COIL_FEATURE_DIM:
            raise ValueError("generation-2 Port artifact has incompatible coil features")
        if model.package_dim != GENERATION2_PACKAGE_FEATURE_DIM:
            raise ValueError("generation-2 Port artifact has incompatible package features")
        if model.coil_pair_dim != GENERATION2_PAIR_FEATURE_DIM:
            raise ValueError("generation-2 Port artifact has incompatible pair features")
        if model.package_pair_dim != GENERATION2_PAIR_FEATURE_DIM:
            raise ValueError("generation-2 Port artifact has incompatible package pair features")
        if model.cross_dim != GENERATION2_CROSS_FEATURE_DIM:
            raise ValueError("generation-2 Port artifact has incompatible cross features")
        if int(baseline_segments) < 8:
            raise ValueError("baseline_segments must be >= 8")
        self.model = model
        self.normalizer = normalizer
        self.baseline_segments = int(baseline_segments)
        self.material_domain = dict(material_domain)
        self.geometry_domain = geometry_domain
        self.partition_fingerprint = (
            None if partition_fingerprint is None else str(partition_fingerprint)
        )
        self.device = resolve_torch_device(device)
        dtype = next(self.model.parameters()).dtype
        if self.device == "mps" and dtype == torch.float64:
            raise ValueError("float64 generation-2 artifacts cannot run on MPS")
        self.model.to(self.device)
        self.model.eval()

    def _validate_material_domain(self, scene: Scene, frequency_hz: float):
        domain = self.material_domain
        package_count_domain = domain.get("package_count")
        if package_count_domain is not None:
            lower, upper = (int(value) for value in package_count_domain)
            count = len(scene.packages)
            if count < lower or count > upper:
                raise ValueError(
                    f"package count is outside generation-2 training domain [{lower}, {upper}]"
                )
        from .tensor_features import _material_tensors

        epsilon, sigma = _material_tensors(scene.medium, frequency_hz)
        _range_contains(
            np.linalg.eigvalsh(epsilon),
            domain.get("background_epsilon"),
            name="background permittivity principal values",
        )
        _range_contains(
            np.linalg.eigvalsh(sigma),
            domain.get("background_sigma"),
            name="background conductivity principal values",
        )
        for package in scene.packages:
            epsilon, sigma = _material_tensors(package.material, frequency_hz)
            _range_contains(
                np.linalg.eigvalsh(epsilon),
                domain.get("package_epsilon"),
                name="package permittivity principal values",
            )
            _range_contains(
                np.linalg.eigvalsh(sigma),
                domain.get("package_sigma"),
                name="package conductivity principal values",
            )
            _range_contains(
                np.asarray([package.material.relative_permeability], dtype=float),
                domain.get("package_mu"),
                name="package relative permeability",
            )

    def _normalized(self, scene: Scene, frequency_hz: float):
        encoded = encode_generation2_scene(scene, frequency_hz)
        return encoded, self.normalizer.normalize(encoded)

    def latent(self, scene: Scene, frequency_hz: float):
        self._validate_material_domain(scene, frequency_hz)
        _, normalized = self._normalized(scene, frequency_hz)
        dtype = next(self.model.parameters()).dtype
        values = [
            torch.as_tensor(value, dtype=dtype, device=self.device).unsqueeze(0)
            for value in normalized
        ]
        with torch.no_grad():
            coil, package = generation2_batched_latent(self.model, *values)
        return coil[0], package[0], normalized

    def predict_structured(self, scene: Scene, frequency_hz: float) -> StructuredPortPrediction:
        validate_package_conductor_topology(scene)
        validate_hybrid_geometry_domain(scene, frequency_hz, self.geometry_domain)
        self._validate_material_domain(scene, frequency_hz)
        _, normalized = self._normalized(scene, frequency_hz)
        baseline = analytic_port_baseline(
            Scene(scene.coils, scene.medium, ()),
            frequency_hz,
            segments_per_coil=self.baseline_segments,
        )
        dtype = next(self.model.parameters()).dtype
        values = [
            torch.as_tensor(value, dtype=dtype, device=self.device).unsqueeze(0)
            for value in normalized
        ]
        with torch.no_grad():
            resistance, reactance, channels = forward_generation2_port_batch(
                self.model,
                *values,
                torch.as_tensor(
                    baseline.resistance,
                    dtype=dtype,
                    device=self.device,
                ).unsqueeze(0),
                torch.as_tensor(
                    2.0 * np.pi * float(frequency_hz) * baseline.inductance,
                    dtype=dtype,
                    device=self.device,
                ).unsqueeze(0),
                reactance_scale=self.normalizer.reactance_scale,
                dielectric_loss_gate=torch.as_tensor(
                    [_dielectric_loss_gate(scene, frequency_hz)],
                    dtype=dtype,
                    device=self.device,
                ),
                reactance_gate=torch.as_tensor(
                    [_reactance_gate(frequency_hz)],
                    dtype=dtype,
                    device=self.device,
                ),
            )
        impedance = (
            resistance[0].detach().cpu().numpy()
            + 1j * reactance[0].detach().cpu().numpy()
        )
        channel_values = channels[0].detach().cpu().numpy()
        return StructuredPortPrediction(
            impedance,
            channel_values,
            tuple(f"coil:{index}" for index in range(len(scene.coils)))
            + ("electric_environment:aggregate",),
        )

    def predict(self, scene: Scene, frequency_hz: float) -> np.ndarray:
        return self.predict_structured(scene, frequency_hz).impedance

    def _payload(self):
        return {
            "schema": GENERATION2_PORT_ARTIFACT_SCHEMA,
            "model_generation": GENERATION2_PORT_MODEL_GENERATION,
            "feature_schema": GENERATION2_FEATURE_SCHEMA,
            "model": {
                "coil_dim": self.model.coil_dim,
                "coil_pair_dim": self.model.coil_pair_dim,
                "package_dim": self.model.package_dim,
                "cross_dim": self.model.cross_dim,
                "package_pair_dim": self.model.package_pair_dim,
                "hidden_dim": self.model.hidden_dim,
                "factor_rank": self.model.factor_rank,
                "depth": self.model.depth,
                "interaction_rounds": self.model.interaction_rounds,
                "resistance_log_limit": self.model.resistance_log_limit,
                "state_dict": self.model.state_dict(),
            },
            "normalizer": self.normalizer.to_dict(),
            "baseline_segments": self.baseline_segments,
            "material_domain": self.material_domain,
            "geometry_domain": self.geometry_domain,
            "partition_fingerprint": self.partition_fingerprint,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._payload(), path)

    @staticmethod
    def load(path, *, device: str = "cpu") -> "Generation2PortArtifact":
        resolved = resolve_torch_device(device)
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if int(payload.get("schema", -1)) != GENERATION2_PORT_ARTIFACT_SCHEMA:
            raise ValueError("unsupported generation-2 Port artifact schema")
        if int(payload.get("model_generation", -1)) != GENERATION2_PORT_MODEL_GENERATION:
            raise ValueError("artifact is not a generation-2 Port model")
        if int(payload.get("feature_schema", -1)) != GENERATION2_FEATURE_SCHEMA:
            raise ValueError("generation-2 Port feature schema is incompatible")
        metadata = payload["model"]
        dtype = _state_dtype(metadata["state_dict"])
        model = Generation2PortNet(
            coil_dim=int(metadata["coil_dim"]),
            coil_pair_dim=int(metadata["coil_pair_dim"]),
            package_dim=int(metadata["package_dim"]),
            cross_dim=int(metadata["cross_dim"]),
            package_pair_dim=int(metadata["package_pair_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
            factor_rank=int(metadata["factor_rank"]),
            depth=int(metadata["depth"]),
            interaction_rounds=int(metadata["interaction_rounds"]),
            resistance_log_limit=float(metadata["resistance_log_limit"]),
        ).to(dtype=dtype)
        model.load_state_dict(metadata["state_dict"])
        return Generation2PortArtifact(
            model,
            Generation2Normalizer.from_dict(payload["normalizer"]),
            baseline_segments=int(payload["baseline_segments"]),
            material_domain=payload["material_domain"],
            geometry_domain=payload.get("geometry_domain"),
            partition_fingerprint=payload.get("partition_fingerprint"),
            device=resolved,
        )


def _fingerprint_array(digest, label: str, value) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(label).encode("utf-8"))
    digest.update(b"\0array\0")
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))


def _fingerprint_value(digest, label: str, value) -> None:
    if isinstance(value, torch.Tensor):
        _fingerprint_array(digest, label, value.detach().cpu().numpy())
        return
    if isinstance(value, np.ndarray):
        _fingerprint_array(digest, label, value)
        return
    digest.update(str(label).encode("utf-8"))
    if value is None:
        digest.update(b"\0none\0")
    elif isinstance(value, dict):
        digest.update(b"\0dict\0")
        for key in sorted(value, key=lambda item: str(item)):
            _fingerprint_value(digest, f"{label}.{key}", value[key])
    elif isinstance(value, (tuple, list)):
        digest.update(b"\0sequence\0")
        digest.update(str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _fingerprint_value(digest, f"{label}[{index}]", item)
    elif isinstance(value, (bool, int, float, str, np.generic)):
        digest.update(b"\0scalar\0")
        digest.update(type(value).__name__.encode("ascii", "backslashreplace"))
        digest.update(b"\0")
        digest.update(
            repr(value.item() if isinstance(value, np.generic) else value).encode("utf-8")
        )
    else:
        digest.update(b"\0repr\0")
        digest.update(repr(value).encode("utf-8"))


def generation2_port_fingerprint(artifact: Generation2PortArtifact) -> str:
    """Device-independent semantic fingerprint of one Gen2 Port artifact."""
    digest = sha256()
    model_metadata = {
        "schema": GENERATION2_PORT_ARTIFACT_SCHEMA,
        "model_generation": GENERATION2_PORT_MODEL_GENERATION,
        "feature_schema": GENERATION2_FEATURE_SCHEMA,
        "coil_dim": artifact.model.coil_dim,
        "coil_pair_dim": artifact.model.coil_pair_dim,
        "package_dim": artifact.model.package_dim,
        "cross_dim": artifact.model.cross_dim,
        "package_pair_dim": artifact.model.package_pair_dim,
        "hidden_dim": artifact.model.hidden_dim,
        "factor_rank": artifact.model.factor_rank,
        "depth": artifact.model.depth,
        "interaction_rounds": artifact.model.interaction_rounds,
        "resistance_log_limit": artifact.model.resistance_log_limit,
        "baseline_segments": artifact.baseline_segments,
        "material_domain": artifact.material_domain,
        "geometry_domain": artifact.geometry_domain,
        "partition_fingerprint": artifact.partition_fingerprint,
        "normalizer": artifact.normalizer.to_dict(),
    }
    _fingerprint_value(digest, "metadata", model_metadata)
    for name, tensor in sorted(artifact.model.state_dict().items()):
        _fingerprint_value(digest, f"state.{name}", tensor)
    return digest.hexdigest()


def generation2_material_domain(samples):
    return _sample_tensor_ranges(tuple(samples))


__all__ = [
    "GENERATION2_PORT_ARTIFACT_SCHEMA",
    "GENERATION2_PORT_MODEL_GENERATION",
    "Generation2Normalizer",
    "Generation2PortNet",
    "Generation2PortArtifact",
    "generation2_batched_latent",
    "forward_generation2_port_batch",
    "generation2_port_fingerprint",
    "generation2_material_domain",
]
