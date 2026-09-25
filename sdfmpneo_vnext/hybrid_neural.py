from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import json
import math
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.hybrid_neural requires the 'neural' extra: "
        "pip install 'sdfmpneo[neural]'"
    ) from exc

from .analytic_baseline import analytic_port_baseline
from .hybrid_features import (
    EncodedHybridScene,
    encode_hybrid_scene_invariant,
)
from .hybrid_training_data import HybridTeacherSample
from .prediction import StructuredPortPrediction
from .scene import Scene


HYBRID_ARTIFACT_SCHEMA = 1


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
):
    layers = []
    width = int(
        input_dim
    )
    for _ in range(
        depth
    ):
        layers.extend(
            [
                nn.Linear(
                    width,
                    hidden_dim,
                ),
                nn.SiLU(),
            ]
        )
        width = (
            hidden_dim
        )
    layers.append(
        nn.Linear(
            width,
            output_dim,
        )
    )
    return nn.Sequential(
        *layers
    )


def _matrix_sqrt(
    matrix,
    *,
    inverse: bool,
):
    matrix = 0.5 * (
        matrix
        + matrix.conj().transpose(
            -1,
            -2,
        )
    )
    eigenvalues, eigenvectors = (
        torch.linalg.eigh(
            matrix
        )
    )
    scale = torch.clamp(
        torch.max(
            torch.abs(
                eigenvalues
            )
        ),
        min=1e-12,
    )
    floor = (
        1e-10
        * scale
        + 1e-14
    )
    clipped = torch.clamp(
        eigenvalues.real,
        min=floor,
    )
    diagonal = (
        torch.rsqrt(
            clipped
        )
        if inverse
        else torch.sqrt(
            clipped
        )
    )
    return (
        eigenvectors
        @ torch.diag(
            diagonal.to(
                eigenvectors.dtype
            )
        )
        @ eigenvectors.conj().transpose(
            -1,
            -2,
        )
    )


@dataclass(frozen=True)
class HybridNormalizer:
    coil_node_mean: np.ndarray
    coil_node_scale: np.ndarray
    coil_pair_mean: np.ndarray
    coil_pair_scale: np.ndarray
    package_mean: np.ndarray
    package_scale: np.ndarray
    coil_package_mean: np.ndarray
    coil_package_scale: np.ndarray
    package_pair_mean: np.ndarray
    package_pair_scale: np.ndarray
    resistance_scale: float
    reactance_scale: float

    @staticmethod
    def fit(
        samples,
        *,
        floor: float = 1e-8,
    ) -> "HybridNormalizer":
        samples = tuple(
            samples
        )
        if not samples:
            raise ValueError(
                "at least one hybrid sample is required"
            )

        def stack(
            getter,
        ):
            arrays = [
                np.asarray(
                    getter(
                        sample
                    ),
                    dtype=float,
                ).reshape(
                    -1,
                    np.asarray(
                        getter(
                            sample
                        )
                    ).shape[
                        -1
                    ],
                )
                for sample
                in samples
            ]
            return np.concatenate(
                arrays,
                axis=0,
            )

        coil_node = stack(
            lambda sample: (
                sample.encoded.coil.node_features
            )
        )
        coil_pair = stack(
            lambda sample: (
                sample.encoded.coil.pair_features
            )
        )
        package = stack(
            lambda sample: (
                sample.encoded.package_features
            )
        )
        coil_package = stack(
            lambda sample: (
                sample.encoded.coil_package_features
            )
        )
        package_pair = stack(
            lambda sample: (
                sample.encoded.package_pair_features
            )
        )

        def stats(
            values,
        ):
            return (
                values.mean(
                    axis=0
                ),
                np.maximum(
                    values.std(
                        axis=0
                    ),
                    floor,
                ),
            )

        (
            coil_node_mean,
            coil_node_scale,
        ) = stats(
            coil_node
        )
        (
            coil_pair_mean,
            coil_pair_scale,
        ) = stats(
            coil_pair
        )
        (
            package_mean,
            package_scale,
        ) = stats(
            package
        )
        (
            coil_package_mean,
            coil_package_scale,
        ) = stats(
            coil_package
        )
        (
            package_pair_mean,
            package_pair_scale,
        ) = stats(
            package_pair
        )

        r_values = []
        x_values = []
        for sample in samples:
            target = np.asarray(
                sample.target_impedance,
                dtype=complex,
            )
            r_values.append(
                (
                    target.real
                    - sample.baseline_resistance
                ).ravel()
            )
            x_values.append(
                (
                    target.imag
                    - sample.baseline_reactance
                ).ravel()
            )
        resistance_values = np.concatenate(
            r_values
        )
        reactance_values = np.concatenate(
            x_values
        )

        return HybridNormalizer(
            coil_node_mean,
            coil_node_scale,
            coil_pair_mean,
            coil_pair_scale,
            package_mean,
            package_scale,
            coil_package_mean,
            coil_package_scale,
            package_pair_mean,
            package_pair_scale,
            max(
                float(
                    np.sqrt(
                        np.mean(
                            resistance_values**2
                        )
                    )
                ),
                floor,
            ),
            max(
                float(
                    np.sqrt(
                        np.mean(
                            reactance_values**2
                        )
                    )
                ),
                floor,
            ),
        )

    def normalize(
        self,
        encoded: EncodedHybridScene,
    ):
        return (
            (
                encoded.coil.node_features
                - self.coil_node_mean
            )
            / self.coil_node_scale,
            (
                encoded.coil.pair_features
                - self.coil_pair_mean
            )
            / self.coil_pair_scale,
            (
                encoded.package_features
                - self.package_mean
            )
            / self.package_scale,
            (
                encoded.coil_package_features
                - self.coil_package_mean
            )
            / self.coil_package_scale,
            (
                encoded.package_pair_features
                - self.package_pair_mean
            )
            / self.package_pair_scale,
        )

    def to_dict(
        self,
    ):
        return {
            "coil_node_mean": self.coil_node_mean,
            "coil_node_scale": self.coil_node_scale,
            "coil_pair_mean": self.coil_pair_mean,
            "coil_pair_scale": self.coil_pair_scale,
            "package_mean": self.package_mean,
            "package_scale": self.package_scale,
            "coil_package_mean": self.coil_package_mean,
            "coil_package_scale": self.coil_package_scale,
            "package_pair_mean": self.package_pair_mean,
            "package_pair_scale": self.package_pair_scale,
            "resistance_scale": self.resistance_scale,
            "reactance_scale": self.reactance_scale,
        }

    @staticmethod
    def from_dict(
        data,
    ):
        return HybridNormalizer(
            np.asarray(
                data[
                    "coil_node_mean"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "coil_node_scale"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "coil_pair_mean"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "coil_pair_scale"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "package_mean"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "package_scale"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "coil_package_mean"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "coil_package_scale"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "package_pair_mean"
                ],
                dtype=float,
            ),
            np.asarray(
                data[
                    "package_pair_scale"
                ],
                dtype=float,
            ),
            float(
                data[
                    "resistance_scale"
                ]
            ),
            float(
                data[
                    "reactance_scale"
                ]
            ),
        )


class HybridPhysicsFactoredResidualNet(
    nn.Module
):
    """Heterogeneous coil/package graph with hard passive port decoding."""

    def __init__(
        self,
        coil_dim: int = 17,
        coil_pair_dim: int = 15,
        package_dim: int = 13,
        cross_dim: int = 15,
        package_pair_dim: int = 15,
        hidden_dim: int = 64,
        factor_rank: int = 4,
        depth: int = 2,
    ):
        super().__init__()
        if (
            hidden_dim < 4
            or factor_rank < 1
            or depth < 1
        ):
            raise ValueError(
                "invalid hybrid neural dimensions"
            )
        self.coil_dim = int(
            coil_dim
        )
        self.coil_pair_dim = int(
            coil_pair_dim
        )
        self.package_dim = int(
            package_dim
        )
        self.cross_dim = int(
            cross_dim
        )
        self.package_pair_dim = int(
            package_pair_dim
        )
        self.hidden_dim = int(
            hidden_dim
        )
        self.factor_rank = int(
            factor_rank
        )
        self.depth = int(
            depth
        )

        h = self.hidden_dim
        self.coil_encoder = _mlp(
            self.coil_dim,
            h,
            h,
            self.depth,
        )
        self.package_encoder = _mlp(
            self.package_dim,
            h,
            h,
            self.depth,
        )
        self.coil_pair_encoder = _mlp(
            2 * h
            + self.coil_pair_dim,
            h,
            h,
            self.depth,
        )
        self.package_pair_encoder = _mlp(
            2 * h
            + self.package_pair_dim,
            h,
            h,
            self.depth,
        )
        self.coil_to_package_encoder = _mlp(
            2 * h
            + self.cross_dim,
            h,
            h,
            self.depth,
        )
        self.package_to_coil_encoder = _mlp(
            2 * h
            + self.cross_dim,
            h,
            h,
            self.depth,
        )
        self.package_update = _mlp(
            3 * h,
            h,
            h,
            self.depth,
        )
        self.coil_update = _mlp(
            3 * h,
            h,
            h,
            self.depth,
        )

        self.loss_factor_head = (
            nn.Linear(
                h,
                self.factor_rank,
            )
        )
        self.reactance_diag_head = (
            nn.Linear(
                h,
                1,
            )
        )
        self.reactance_pair_head = _mlp(
            2 * h
            + self.coil_pair_dim,
            h,
            1,
            self.depth,
        )
        self.conductor_channel_head = _mlp(
            2 * h
            + self.coil_pair_dim,
            h,
            2 * self.factor_rank,
            self.depth,
        )
        self.dielectric_channel_head = _mlp(
            2 * h
            + self.cross_dim,
            h,
            2 * self.factor_rank,
            self.depth,
        )

    @staticmethod
    def _aggregate(
        messages,
        reference,
    ):
        if not messages:
            return torch.zeros_like(
                reference
            )
        return torch.stack(
            messages,
            dim=0,
        ).sum(
            dim=0
        ) / math.sqrt(
            len(
                messages
            )
        )

    def _latent(
        self,
        coil_features,
        coil_pair_features,
        package_features,
        coil_package_features,
        package_pair_features,
    ):
        coil0 = self.coil_encoder(
            coil_features
        )
        package0 = (
            self.package_encoder(
                package_features
            )
        )
        n_coils = coil0.shape[
            0
        ]
        n_packages = package0.shape[
            0
        ]

        coil_pair_messages = []
        for i in range(
            n_coils
        ):
            messages = []
            for j in range(
                n_coils
            ):
                if i == j:
                    continue
                messages.append(
                    self.coil_pair_encoder(
                        torch.cat(
                            (
                                coil0[
                                    i
                                ],
                                coil0[
                                    j
                                ],
                                coil_pair_features[
                                    i,
                                    j,
                                ],
                            ),
                            dim=-1,
                        )
                    )
                )
            coil_pair_messages.append(
                self._aggregate(
                    messages,
                    coil0[
                        i
                    ],
                )
            )

        package_pair_messages = []
        coil_to_package_messages = []
        for package_index in range(
            n_packages
        ):
            pair_messages = []
            for other in range(
                n_packages
            ):
                if (
                    package_index
                    == other
                ):
                    continue
                pair_messages.append(
                    self.package_pair_encoder(
                        torch.cat(
                            (
                                package0[
                                    package_index
                                ],
                                package0[
                                    other
                                ],
                                package_pair_features[
                                    package_index,
                                    other,
                                ],
                            ),
                            dim=-1,
                        )
                    )
                )
            package_pair_messages.append(
                self._aggregate(
                    pair_messages,
                    package0[
                        package_index
                    ],
                )
            )

            coil_messages = []
            for coil_index in range(
                n_coils
            ):
                coil_messages.append(
                    self.coil_to_package_encoder(
                        torch.cat(
                            (
                                package0[
                                    package_index
                                ],
                                coil0[
                                    coil_index
                                ],
                                coil_package_features[
                                    coil_index,
                                    package_index,
                                ],
                            ),
                            dim=-1,
                        )
                    )
                )
            coil_to_package_messages.append(
                self._aggregate(
                    coil_messages,
                    package0[
                        package_index
                    ],
                )
            )

        package = torch.stack(
            [
                self.package_update(
                    torch.cat(
                        (
                            package0[
                                index
                            ],
                            package_pair_messages[
                                index
                            ],
                            coil_to_package_messages[
                                index
                            ],
                        ),
                        dim=-1,
                    )
                )
                for index in range(
                    n_packages
                )
            ],
            dim=0,
        )

        package_to_coil_messages = []
        for coil_index in range(
            n_coils
        ):
            messages = []
            for package_index in range(
                n_packages
            ):
                messages.append(
                    self.package_to_coil_encoder(
                        torch.cat(
                            (
                                coil0[
                                    coil_index
                                ],
                                package[
                                    package_index
                                ],
                                coil_package_features[
                                    coil_index,
                                    package_index,
                                ],
                            ),
                            dim=-1,
                        )
                    )
                )
            package_to_coil_messages.append(
                self._aggregate(
                    messages,
                    coil0[
                        coil_index
                    ],
                )
            )

        coil = torch.stack(
            [
                self.coil_update(
                    torch.cat(
                        (
                            coil0[
                                index
                            ],
                            coil_pair_messages[
                                index
                            ],
                            package_to_coil_messages[
                                index
                            ],
                        ),
                        dim=-1,
                    )
                )
                for index in range(
                    n_coils
                )
            ],
            dim=0,
        )
        return (
            coil,
            package,
        )

    def _decode_impedance(
        self,
        coil,
        coil_pair_features,
        baseline_resistance,
        baseline_reactance,
        *,
        resistance_scale: float,
        reactance_scale: float,
    ):
        n = coil.shape[
            0
        ]
        factors = (
            self.loss_factor_head(
                coil
            )
        )
        resistance = (
            baseline_resistance
            + float(
                resistance_scale
            )
            * (
                factors
                @ factors.transpose(
                    0,
                    1,
                )
            )
        )
        resistance = 0.5 * (
            resistance
            + resistance.transpose(
                0,
                1,
            )
        )

        residual = torch.zeros_like(
            baseline_reactance
        )
        diagonal = (
            float(
                reactance_scale
            )
            * self.reactance_diag_head(
                coil
            ).squeeze(
                -1
            )
        )
        residual = (
            residual
            + torch.diag(
                diagonal
            )
        )
        for i in range(
            n
        ):
            for j in range(
                i
            ):
                forward = (
                    self.reactance_pair_head(
                        torch.cat(
                            (
                                coil[
                                    i
                                ],
                                coil[
                                    j
                                ],
                                coil_pair_features[
                                    i,
                                    j,
                                ],
                            ),
                            dim=-1,
                        )
                    ).squeeze()
                )
                reverse = (
                    self.reactance_pair_head(
                        torch.cat(
                            (
                                coil[
                                    j
                                ],
                                coil[
                                    i
                                ],
                                coil_pair_features[
                                    j,
                                    i,
                                ],
                            ),
                            dim=-1,
                        )
                    ).squeeze()
                )
                value = (
                    0.5
                    * float(
                        reactance_scale
                    )
                    * (
                        forward
                        + reverse
                    )
                )
                residual[
                    i,
                    j,
                ] = value
                residual[
                    j,
                    i,
                ] = value
        reactance = (
            baseline_reactance
            + residual
        )
        reactance = 0.5 * (
            reactance
            + reactance.transpose(
                0,
                1,
            )
        )
        return (
            resistance,
            reactance,
        )

    def _complex_factor(
        self,
        raw,
        dtype,
    ):
        real = raw[
            : self.factor_rank
        ]
        imag = raw[
            self.factor_rank :
        ]
        return (
            real.to(
                dtype
            )
            + 1j
            * imag.to(
                dtype
            )
        )

    def _decode_channels(
        self,
        coil,
        package,
        coil_pair_features,
        coil_package_features,
        resistance,
        *,
        dielectric_loss_gate: float,
    ):
        n_ports = coil.shape[
            0
        ]
        complex_dtype = (
            torch.complex64
            if resistance.dtype
            == torch.float32
            else torch.complex128
        )

        raw_channels = []
        for channel_index in range(
            n_ports
        ):
            rows = []
            for port_index in range(
                n_ports
            ):
                raw = (
                    self.conductor_channel_head(
                        torch.cat(
                            (
                                coil[
                                    channel_index
                                ],
                                coil[
                                    port_index
                                ],
                                coil_pair_features[
                                    channel_index,
                                    port_index,
                                ],
                            ),
                            dim=-1,
                        )
                    )
                )
                rows.append(
                    self._complex_factor(
                        raw,
                        complex_dtype,
                    )
                )
            factor = torch.stack(
                rows,
                dim=0,
            )
            raw_channels.append(
                factor
                @ factor.conj().transpose(
                    0,
                    1,
                )
            )

        package_pool = torch.mean(
            package,
            dim=0,
        )
        rows = []
        for port_index in range(
            n_ports
        ):
            cross_summary = torch.mean(
                coil_package_features[
                    port_index
                ],
                dim=0,
            )
            raw = (
                self.dielectric_channel_head(
                    torch.cat(
                        (
                            package_pool,
                            coil[
                                port_index
                            ],
                            cross_summary,
                        ),
                        dim=-1,
                    )
                )
            )
            rows.append(
                self._complex_factor(
                    raw,
                    complex_dtype,
                )
            )
        dielectric_factor = (
            torch.stack(
                rows,
                dim=0,
            )
        )
        dielectric_raw = (
            float(
                dielectric_loss_gate
            )
            * (
                dielectric_factor
                @ dielectric_factor.conj().transpose(
                    0,
                    1,
                )
            )
        )
        raw_channels.append(
            dielectric_raw
        )

        raw_channels = torch.stack(
            raw_channels,
            dim=0,
        )
        resistance_complex = resistance.to(
            complex_dtype
        )
        scale = torch.clamp(
            torch.trace(
                resistance_complex
            ).real
            / max(
                n_ports,
                1,
            ),
            min=1e-12,
        )
        eye = torch.eye(
            n_ports,
            dtype=complex_dtype,
            device=(
                resistance.device
            ),
        )
        jitter = (
            1e-8
            * scale
            / max(
                n_ports,
                1,
            )
        )
        conductor_raw = (
            raw_channels[
                :n_ports
            ]
            + jitter
            * eye.unsqueeze(
                0
            )
        )
        raw_channels = torch.cat(
            (
                conductor_raw,
                raw_channels[
                    n_ports:
                ],
            ),
            dim=0,
        )

        raw_sum = torch.sum(
            raw_channels,
            dim=0,
        )
        congruence = (
            _matrix_sqrt(
                resistance_complex,
                inverse=False,
            )
            @ _matrix_sqrt(
                raw_sum,
                inverse=True,
            )
        )
        channels = torch.stack(
            [
                congruence
                @ channel
                @ congruence.conj().transpose(
                    0,
                    1,
                )
                for channel
                in raw_channels
            ],
            dim=0,
        )
        return 0.5 * (
            channels
            + channels.conj().transpose(
                -1,
                -2,
            )
        )

    def forward_structured(
        self,
        coil_features,
        coil_pair_features,
        package_features,
        coil_package_features,
        package_pair_features,
        baseline_resistance,
        baseline_reactance,
        *,
        resistance_scale: float,
        reactance_scale: float,
        dielectric_loss_gate: float,
    ):
        coil, package = (
            self._latent(
                coil_features,
                coil_pair_features,
                package_features,
                coil_package_features,
                package_pair_features,
            )
        )
        (
            resistance,
            reactance,
        ) = self._decode_impedance(
            coil,
            coil_pair_features,
            baseline_resistance,
            baseline_reactance,
            resistance_scale=(
                resistance_scale
            ),
            reactance_scale=(
                reactance_scale
            ),
        )
        channels = self._decode_channels(
            coil,
            package,
            coil_pair_features,
            coil_package_features,
            resistance,
            dielectric_loss_gate=(
                dielectric_loss_gate
            ),
        )
        return (
            resistance,
            reactance,
            channels,
        )


def _dielectric_loss_gate(
    scene: Scene,
) -> float:
    return float(
        any(
            package.material.conductivity
            > 0.0
            for package
            in scene.packages
        )
    )


@dataclass(frozen=True)
class HybridTrainingReport:
    final_loss: float
    epochs: int
    samples: int
    best_epoch: int
    best_validation_score: float | None
    best_validation_z_error: float | None
    best_validation_channel_error: float | None
    stopped_early: bool


class HybridNeuralResidualArtifact:
    supports_packages = True
    def __init__(
        self,
        model: HybridPhysicsFactoredResidualNet,
        normalizer: HybridNormalizer,
        *,
        baseline_segments: int,
        device: str = "cpu",
    ):
        self.model = model
        self.normalizer = (
            normalizer
        )
        self.baseline_segments = int(
            baseline_segments
        )
        self.device = str(
            device
        )
        self.model.to(
            self.device
        )
        self.model.eval()

    def fingerprint(
        self,
    ) -> str:
        """Stable semantic fingerprint for hybrid weights and preprocessing."""
        digest = sha256()
        config = {
            "schema": HYBRID_ARTIFACT_SCHEMA,
            "model_config": {
                "coil_dim": self.model.coil_dim,
                "coil_pair_dim": self.model.coil_pair_dim,
                "package_dim": self.model.package_dim,
                "cross_dim": self.model.cross_dim,
                "package_pair_dim": self.model.package_pair_dim,
                "hidden_dim": self.model.hidden_dim,
                "factor_rank": self.model.factor_rank,
                "depth": self.model.depth,
            },
            "baseline_segments": self.baseline_segments,
        }
        digest.update(
            json.dumps(
                config,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

        def update_array(name, value):
            array = np.asarray(value)
            digest.update(
                str(name).encode("utf-8")
            )
            digest.update(
                str(array.dtype).encode("ascii")
            )
            digest.update(
                np.asarray(
                    array.shape,
                    dtype=np.int64,
                ).tobytes()
            )
            digest.update(
                np.ascontiguousarray(
                    array
                ).tobytes()
            )

        for name, tensor in sorted(
            self.model.state_dict().items()
        ):
            update_array(
                "state:" + name,
                tensor.detach()
                .cpu()
                .contiguous()
                .numpy(),
            )
        for name, value in sorted(
            self.normalizer.to_dict().items()
        ):
            update_array(
                "normalizer:" + name,
                value,
            )
        return digest.hexdigest()

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> StructuredPortPrediction:
        if not scene.packages:
            raise ValueError(
                "hybrid neural artifact requires at least one package"
            )
        encoded = (
            encode_hybrid_scene_invariant(
                scene,
                frequency_hz,
            )
        )
        (
            coil_node,
            coil_pair,
            package,
            coil_package,
            package_pair,
        ) = self.normalizer.normalize(
            encoded
        )
        conductor_scene = Scene(
            scene.coils,
            scene.medium,
            (),
        )
        baseline = (
            analytic_port_baseline(
                conductor_scene,
                frequency_hz,
                segments_per_coil=(
                    self.baseline_segments
                ),
            )
        )
        dtype = next(
            self.model.parameters()
        ).dtype
        device = next(
            self.model.parameters()
        ).device
        with torch.no_grad():
            (
                resistance,
                reactance,
                channels,
            ) = self.model.forward_structured(
                torch.as_tensor(
                    coil_node,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    coil_pair,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    package,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    coil_package,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    package_pair,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    baseline.resistance,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    2.0
                    * np.pi
                    * float(
                        frequency_hz
                    )
                    * baseline.inductance,
                    dtype=dtype,
                    device=device,
                ),
                resistance_scale=(
                    self.normalizer.resistance_scale
                ),
                reactance_scale=(
                    self.normalizer.reactance_scale
                ),
                dielectric_loss_gate=(
                    _dielectric_loss_gate(
                        scene
                    )
                ),
            )
        return StructuredPortPrediction(
            (
                resistance.detach().cpu().numpy()
                + 1j
                * reactance.detach().cpu().numpy()
            ),
            channels.detach().cpu().numpy(),
        )

    def predict(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> np.ndarray:
        return (
            self.predict_structured(
                scene,
                frequency_hz,
            ).impedance
        )

    def save(
        self,
        path,
    ):
        torch.save(
            {
                "schema": (
                    HYBRID_ARTIFACT_SCHEMA
                ),
                "model_config": {
                    "coil_dim": (
                        self.model.coil_dim
                    ),
                    "coil_pair_dim": (
                        self.model.coil_pair_dim
                    ),
                    "package_dim": (
                        self.model.package_dim
                    ),
                    "cross_dim": (
                        self.model.cross_dim
                    ),
                    "package_pair_dim": (
                        self.model.package_pair_dim
                    ),
                    "hidden_dim": (
                        self.model.hidden_dim
                    ),
                    "factor_rank": (
                        self.model.factor_rank
                    ),
                    "depth": (
                        self.model.depth
                    ),
                },
                "model_state": (
                    self.model.state_dict()
                ),
                "normalizer": (
                    self.normalizer.to_dict()
                ),
                "baseline_segments": (
                    self.baseline_segments
                ),
            },
            Path(
                path
            ),
        )

    @staticmethod
    def load(
        path,
        *,
        device: str = "cpu",
    ):
        try:
            payload = torch.load(
                Path(
                    path
                ),
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(
                Path(
                    path
                ),
                map_location=device,
            )
        if (
            payload.get(
                "schema"
            )
            != HYBRID_ARTIFACT_SCHEMA
        ):
            raise ValueError(
                "unsupported hybrid neural artifact schema"
            )
        model = (
            HybridPhysicsFactoredResidualNet(
                **payload[
                    "model_config"
                ]
            )
        )
        model.load_state_dict(
            payload[
                "model_state"
            ]
        )
        return HybridNeuralResidualArtifact(
            model,
            HybridNormalizer.from_dict(
                payload[
                    "normalizer"
                ]
            ),
            baseline_segments=int(
                payload[
                    "baseline_segments"
                ]
            ),
            device=device,
        )


def _relative_error(
    predicted,
    target,
) -> float:
    return float(
        np.linalg.norm(
            np.asarray(
                predicted
            )
            - np.asarray(
                target
            )
        )
        / max(
            np.linalg.norm(
                np.asarray(
                    target
                )
            ),
            1e-30,
        )
    )


def _predict_sample(
    model,
    normalizer,
    sample,
    *,
    device,
):
    (
        coil_node,
        coil_pair,
        package,
        coil_package,
        package_pair,
    ) = normalizer.normalize(
        sample.encoded
    )
    dtype = next(
        model.parameters()
    ).dtype
    with torch.no_grad():
        (
            resistance,
            reactance,
            channels,
        ) = model.forward_structured(
            torch.as_tensor(
                coil_node,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                coil_pair,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                package,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                coil_package,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                package_pair,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                sample.baseline_resistance,
                dtype=dtype,
                device=device,
            ),
            torch.as_tensor(
                sample.baseline_reactance,
                dtype=dtype,
                device=device,
            ),
            resistance_scale=(
                normalizer.resistance_scale
            ),
            reactance_scale=(
                normalizer.reactance_scale
            ),
            dielectric_loss_gate=(
                _dielectric_loss_gate(
                    sample.scene
                )
            ),
        )
    return (
        (
            resistance.detach().cpu().numpy()
            + 1j
            * reactance.detach().cpu().numpy()
        ),
        channels.detach().cpu().numpy(),
    )


def train_hybrid_residual_surrogate(
    samples,
    *,
    validation_samples=(),
    hidden_dim: int = 64,
    factor_rank: int = 4,
    depth: int = 2,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-6,
    channel_loss_weight: float = 1.0,
    validation_channel_weight: float = 0.5,
    patience: int = 30,
    validation_interval: int = 1,
    min_improvement: float = 1e-5,
    seed: int = 17,
    device: str = "cpu",
):
    samples = tuple(
        samples
    )
    validation_samples = tuple(
        validation_samples
    )
    if not samples:
        raise ValueError(
            "at least one hybrid training sample is required"
        )
    if (
        epochs < 1
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or channel_loss_weight < 0.0
        or validation_channel_weight < 0.0
        or patience < 1
        or validation_interval < 1
        or min_improvement < 0.0
    ):
        raise ValueError(
            "invalid hybrid training configuration"
        )
    baseline_segments = {
        int(
            sample.baseline_segments
        )
        for sample
        in samples
    }
    if len(
        baseline_segments
    ) != 1:
        raise ValueError(
            "all hybrid samples must use the same analytic baseline resolution"
        )
    baseline_segments = (
        baseline_segments.pop()
    )

    torch.manual_seed(
        seed
    )
    np.random.seed(
        seed
    )
    normalizer = (
        HybridNormalizer.fit(
            samples
        )
    )
    first = samples[
        0
    ].encoded
    model = (
        HybridPhysicsFactoredResidualNet(
            coil_dim=(
                first.coil.node_features.shape[
                    -1
                ]
            ),
            coil_pair_dim=(
                first.coil.pair_features.shape[
                    -1
                ]
            ),
            package_dim=(
                first.package_features.shape[
                    -1
                ]
            ),
            cross_dim=(
                first.coil_package_features.shape[
                    -1
                ]
            ),
            package_pair_dim=(
                first.package_pair_features.shape[
                    -1
                ]
            ),
            hidden_dim=(
                hidden_dim
            ),
            factor_rank=(
                factor_rank
            ),
            depth=depth,
        ).to(
            device
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=(
            weight_decay
        ),
    )
    dtype = next(
        model.parameters()
    ).dtype

    best_state = None
    best_score = None
    best_z = None
    best_channel = None
    best_epoch = 0
    stale = 0
    stopped_early = False
    final_loss = np.inf
    epochs_run = 0

    def validation_metrics():
        z_errors = []
        channel_errors = []
        for sample in (
            validation_samples
        ):
            predicted_z, predicted_channels = (
                _predict_sample(
                    model,
                    normalizer,
                    sample,
                    device=device,
                )
            )
            z_errors.append(
                _relative_error(
                    predicted_z,
                    sample.target_impedance,
                )
            )
            channel_errors.append(
                _relative_error(
                    predicted_channels,
                    sample.target_dissipation_channels,
                )
            )
        z_error = float(
            np.mean(
                z_errors
            )
        )
        channel_error = float(
            np.mean(
                channel_errors
            )
        )
        return (
            z_error
            + validation_channel_weight
            * channel_error,
            z_error,
            channel_error,
        )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()
        order = np.random.permutation(
            len(
                samples
            )
        )
        epoch_loss = 0.0

        for sample_index in order:
            sample = samples[
                int(
                    sample_index
                )
            ]
            (
                coil_node,
                coil_pair,
                package,
                coil_package,
                package_pair,
            ) = normalizer.normalize(
                sample.encoded
            )
            optimizer.zero_grad(
                set_to_none=True
            )
            (
                resistance,
                reactance,
                channels,
            ) = model.forward_structured(
                torch.as_tensor(
                    coil_node,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    coil_pair,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    package,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    coil_package,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    package_pair,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    sample.baseline_resistance,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    sample.baseline_reactance,
                    dtype=dtype,
                    device=device,
                ),
                resistance_scale=(
                    normalizer.resistance_scale
                ),
                reactance_scale=(
                    normalizer.reactance_scale
                ),
                dielectric_loss_gate=(
                    _dielectric_loss_gate(
                        sample.scene
                    )
                ),
            )
            target_z = np.asarray(
                sample.target_impedance,
                dtype=complex,
            )
            target_r = torch.as_tensor(
                target_z.real,
                dtype=dtype,
                device=device,
            )
            target_x = torch.as_tensor(
                target_z.imag,
                dtype=dtype,
                device=device,
            )
            r_denom = (
                torch.mean(
                    target_r**2
                )
                + 1e-18
            )
            x_denom = (
                torch.mean(
                    target_x**2
                )
                + 1e-18
            )
            loss = (
                torch.mean(
                    (
                        resistance
                        - target_r
                    ) ** 2
                )
                / r_denom
                + torch.mean(
                    (
                        reactance
                        - target_x
                    ) ** 2
                )
                / x_denom
            )

            target_channels = (
                torch.as_tensor(
                    sample.target_dissipation_channels,
                    dtype=(
                        torch.complex64
                        if dtype
                        == torch.float32
                        else torch.complex128
                    ),
                    device=device,
                )
            )
            channel_denom = (
                torch.mean(
                    torch.abs(
                        target_channels
                    ) ** 2
                )
                + 1e-18
            )
            channel_loss = (
                torch.mean(
                    torch.abs(
                        channels
                        - target_channels
                    ) ** 2
                )
                / channel_denom
            )
            loss = (
                loss
                + channel_loss_weight
                * channel_loss
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                10.0,
            )
            optimizer.step()
            epoch_loss += float(
                loss.detach().cpu()
            )

        final_loss = (
            epoch_loss
            / len(
                samples
            )
        )
        epochs_run = epoch

        if validation_samples:
            if (
                epoch
                % validation_interval
                != 0
                and epoch
                != epochs
            ):
                continue
            model.eval()
            (
                score,
                z_error,
                channel_error,
            ) = validation_metrics()
            if (
                best_score is None
                or score
                < best_score
                - min_improvement
            ):
                best_score = float(
                    score
                )
                best_z = float(
                    z_error
                )
                best_channel = float(
                    channel_error
                )
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    stopped_early = True
                    break
        else:
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError(
            "hybrid training completed without a selectable model state"
        )
    model.load_state_dict(
        best_state
    )
    model.eval()
    artifact = (
        HybridNeuralResidualArtifact(
            model,
            normalizer,
            baseline_segments=(
                baseline_segments
            ),
            device=device,
        )
    )
    return (
        artifact,
        HybridTrainingReport(
            final_loss=float(
                final_loss
            ),
            epochs=int(
                epochs_run
            ),
            samples=len(
                samples
            ),
            best_epoch=int(
                best_epoch
            ),
            best_validation_score=(
                best_score
            ),
            best_validation_z_error=(
                best_z
            ),
            best_validation_channel_error=(
                best_channel
            ),
            stopped_early=bool(
                stopped_early
            ),
        ),
    )
