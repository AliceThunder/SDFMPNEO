from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "sdfmpneo_vnext.neural requires the 'neural' extra: "
        "pip install 'sdfmpneo[neural]'"
    ) from exc

from .analytic_baseline import analytic_port_baseline
from .features import EncodedScene, encode_scene_invariant
from .scene import Scene
from .training_data import TeacherSample


ARTIFACT_SCHEMA = 2


@dataclass(frozen=True)
class ResidualNormalizer:
    node_mean: np.ndarray
    node_scale: np.ndarray
    pair_mean: np.ndarray
    pair_scale: np.ndarray
    resistance_scale: float
    reactance_scale: float

    @staticmethod
    def fit(
        samples,
        *,
        floor: float = 1e-8,
    ) -> "ResidualNormalizer":
        samples = tuple(samples)
        if not samples:
            raise ValueError(
                "at least one teacher sample is required"
            )
        node = np.concatenate(
            [
                s.encoded.node_features
                for s in samples
            ],
            axis=0,
        )
        pair = np.concatenate(
            [
                s.encoded.pair_features.reshape(
                    -1,
                    s.encoded.pair_features.shape[-1],
                )
                for s in samples
            ],
            axis=0,
        )
        node_mean = node.mean(axis=0)
        node_scale = np.maximum(
            node.std(axis=0),
            floor,
        )
        pair_mean = pair.mean(axis=0)
        pair_scale = np.maximum(
            pair.std(axis=0),
            floor,
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
        r = np.concatenate(r_values)
        x = np.concatenate(x_values)
        resistance_scale = max(
            float(
                np.sqrt(
                    np.mean(r * r)
                )
            ),
            floor,
        )
        reactance_scale = max(
            float(
                np.sqrt(
                    np.mean(x * x)
                )
            ),
            floor,
        )
        return ResidualNormalizer(
            node_mean,
            node_scale,
            pair_mean,
            pair_scale,
            resistance_scale,
            reactance_scale,
        )

    def normalize(
        self,
        encoded: EncodedScene,
    ):
        return (
            (
                encoded.node_features
                - self.node_mean
            )
            / self.node_scale,
            (
                encoded.pair_features
                - self.pair_mean
            )
            / self.pair_scale,
        )

    def to_dict(self):
        return {
            "node_mean": self.node_mean,
            "node_scale": self.node_scale,
            "pair_mean": self.pair_mean,
            "pair_scale": self.pair_scale,
            "resistance_scale": self.resistance_scale,
            "reactance_scale": self.reactance_scale,
        }

    @staticmethod
    def from_dict(data):
        def _array(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            return np.asarray(
                value,
                dtype=float,
            )

        return ResidualNormalizer(
            _array(
                data["node_mean"]
            ),
            _array(
                data["node_scale"]
            ),
            _array(
                data["pair_mean"]
            ),
            _array(
                data["pair_scale"]
            ),
            float(
                data["resistance_scale"]
            ),
            float(
                data["reactance_scale"]
            ),
        )


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
):
    layers = []
    width = input_dim
    for _ in range(depth):
        layers.extend(
            [
                nn.Linear(
                    width,
                    hidden_dim,
                ),
                nn.SiLU(),
            ]
        )
        width = hidden_dim
    layers.append(
        nn.Linear(
            width,
            output_dim,
        )
    )
    return nn.Sequential(*layers)


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
        torch.linalg.eigh(matrix)
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
        1e-10 * scale
        + 1e-14
    )
    clipped = torch.clamp(
        eigenvalues.real,
        min=floor,
    )
    if inverse:
        diagonal = torch.rsqrt(
            clipped
        )
    else:
        diagonal = torch.sqrt(
            clipped
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


class PhysicsFactoredResidualNet(nn.Module):
    """Permutation-equivariant graph residual with hard port structure."""

    def __init__(
        self,
        node_dim: int = 17,
        pair_dim: int = 15,
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
                "invalid neural residual dimensions"
            )
        self.node_dim = int(
            node_dim
        )
        self.pair_dim = int(
            pair_dim
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

        self.node_encoder = _mlp(
            self.node_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.depth,
        )
        self.edge_encoder = _mlp(
            2 * self.hidden_dim
            + self.pair_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.depth,
        )
        self.node_update = _mlp(
            2 * self.hidden_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.depth,
        )
        self.loss_factor_head = nn.Linear(
            self.hidden_dim,
            self.factor_rank,
        )
        self.reactance_diag_head = nn.Linear(
            self.hidden_dim,
            1,
        )
        self.reactance_pair_head = _mlp(
            2 * self.hidden_dim
            + self.pair_dim,
            self.hidden_dim,
            1,
            self.depth,
        )
        self.channel_factor_head = _mlp(
            2 * self.hidden_dim
            + self.pair_dim,
            self.hidden_dim,
            2 * self.factor_rank,
            self.depth,
        )

    def _edge_input(
        self,
        h,
        pair,
        i: int,
        j: int,
    ):
        return torch.cat(
            (
                h[i],
                h[j],
                pair[i, j],
            ),
            dim=-1,
        )

    def _updated_features(
        self,
        node_features,
        pair_features,
    ):
        h = self.node_encoder(
            node_features
        )
        n = h.shape[0]
        messages = []
        for i in range(n):
            incoming = []
            for j in range(n):
                if i == j:
                    continue
                incoming.append(
                    self.edge_encoder(
                        self._edge_input(
                            h,
                            pair_features,
                            i,
                            j,
                        )
                    )
                )
            if incoming:
                aggregate = torch.stack(
                    incoming,
                    dim=0,
                ).sum(
                    dim=0
                ) / math.sqrt(
                    len(incoming)
                )
            else:
                aggregate = (
                    torch.zeros_like(
                        h[i]
                    )
                )
            messages.append(
                aggregate
            )
        return self.node_update(
            torch.cat(
                (
                    h,
                    torch.stack(
                        messages,
                        dim=0,
                    ),
                ),
                dim=-1,
            )
        )

    def _decode_impedance(
        self,
        updated,
        pair_features,
        baseline_resistance,
        baseline_reactance,
        *,
        resistance_scale: float,
        reactance_scale: float,
    ):
        n = updated.shape[0]

        factors = (
            self.loss_factor_head(
                updated
            )
        )
        r_residual = (
            float(
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
        resistance = (
            baseline_resistance
            + r_residual
        )
        resistance = 0.5 * (
            resistance
            + resistance.transpose(
                0,
                1,
            )
        )

        x_residual = (
            torch.zeros_like(
                baseline_reactance
            )
        )
        diag = (
            float(
                reactance_scale
            )
            * self.reactance_diag_head(
                updated
            ).squeeze(-1)
        )
        x_residual = (
            x_residual
            + torch.diag(diag)
        )
        for i in range(n):
            for j in range(i):
                forward = (
                    self.reactance_pair_head(
                        self._edge_input(
                            updated,
                            pair_features,
                            i,
                            j,
                        )
                    ).squeeze()
                )
                reverse = (
                    self.reactance_pair_head(
                        self._edge_input(
                            updated,
                            pair_features,
                            j,
                            i,
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
                x_residual[
                    i,
                    j,
                ] = value
                x_residual[
                    j,
                    i,
                ] = value

        reactance = (
            baseline_reactance
            + x_residual
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

    def _decode_channels(
        self,
        updated,
        pair_features,
        resistance,
    ):
        n = updated.shape[0]
        complex_dtype = (
            torch.complex64
            if resistance.dtype
            == torch.float32
            else torch.complex128
        )
        raw_channels = []
        for channel in range(n):
            rows = []
            for port in range(n):
                raw = (
                    self.channel_factor_head(
                        self._edge_input(
                            updated,
                            pair_features,
                            channel,
                            port,
                        )
                    )
                )
                real = raw[
                    : self.factor_rank
                ]
                imag = raw[
                    self.factor_rank :
                ]
                rows.append(
                    real.to(
                        complex_dtype
                    )
                    + 1j
                    * imag.to(
                        complex_dtype
                    )
                )
            factor = torch.stack(
                rows,
                dim=0,
            )
            channel_matrix = (
                factor
                @ factor.conj().transpose(
                    0,
                    1,
                )
            )
            raw_channels.append(
                channel_matrix
            )

        raw_channels = torch.stack(
            raw_channels,
            dim=0,
        )
        resistance_complex = (
            resistance.to(
                complex_dtype
            )
        )
        mean_scale = torch.clamp(
            torch.trace(
                resistance_complex
            ).real
            / max(n, 1),
            min=1e-12,
        )
        jitter = (
            1e-8
            * mean_scale
        )
        eye = torch.eye(
            n,
            dtype=complex_dtype,
            device=resistance.device,
        )
        raw_channels = (
            raw_channels
            + (
                jitter
                / max(n, 1)
            )
            * eye.unsqueeze(0)
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
        channels = 0.5 * (
            channels
            + channels.conj().transpose(
                -1,
                -2,
            )
        )
        return channels

    def forward(
        self,
        node_features,
        pair_features,
        baseline_resistance,
        baseline_reactance,
        *,
        resistance_scale: float,
        reactance_scale: float,
    ):
        updated = (
            self._updated_features(
                node_features,
                pair_features,
            )
        )
        return (
            self._decode_impedance(
                updated,
                pair_features,
                baseline_resistance,
                baseline_reactance,
                resistance_scale=(
                    resistance_scale
                ),
                reactance_scale=(
                    reactance_scale
                ),
            )
        )

    def forward_structured(
        self,
        node_features,
        pair_features,
        baseline_resistance,
        baseline_reactance,
        *,
        resistance_scale: float,
        reactance_scale: float,
    ):
        updated = (
            self._updated_features(
                node_features,
                pair_features,
            )
        )
        resistance, reactance = (
            self._decode_impedance(
                updated,
                pair_features,
                baseline_resistance,
                baseline_reactance,
                resistance_scale=(
                    resistance_scale
                ),
                reactance_scale=(
                    reactance_scale
                ),
            )
        )
        channels = (
            self._decode_channels(
                updated,
                pair_features,
                resistance,
            )
        )
        return (
            resistance,
            reactance,
            channels,
        )


@dataclass(frozen=True)
class NeuralPortPrediction:
    impedance: np.ndarray
    dissipation_channels: np.ndarray

    def coil_power(
        self,
        currents,
    ) -> np.ndarray:
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if currents.shape != (
            self.impedance.shape[0],
        ):
            raise ValueError(
                "currents have wrong shape"
            )
        return np.asarray(
            [
                0.5
                * np.real(
                    np.vdot(
                        currents,
                        channel @ currents,
                    )
                )
                for channel
                in self.dissipation_channels
            ],
            dtype=float,
        )

    def power_closure_error(self) -> float:
        summed = np.sum(
            self.dissipation_channels,
            axis=0,
        )
        target = 0.5 * (
            self.impedance
            + self.impedance.conj().T
        )
        return float(
            np.linalg.norm(
                summed - target
            )
            / max(
                np.linalg.norm(target),
                1e-30,
            )
        )


@dataclass
class NeuralTrainingReport:
    final_loss: float
    epochs: int
    samples: int
    best_epoch: int
    best_validation_score: float | None
    best_validation_z_error: float | None
    best_validation_channel_error: float | None
    stopped_early: bool


def _relative_z_error(
    predicted,
    target,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=complex,
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    return float(
        np.linalg.norm(
            predicted
            - target
        )
        / max(
            np.linalg.norm(
                target
            ),
            1e-30,
        )
    )


def _relative_channel_error(
    predicted,
    target,
) -> float:
    predicted = np.asarray(
        predicted,
        dtype=complex,
    )
    target = np.asarray(
        target,
        dtype=complex,
    )
    return float(
        np.linalg.norm(
            predicted
            - target
        )
        / max(
            np.linalg.norm(
                target
            ),
            1e-30,
        )
    )


def _predict_sample_numpy(
    model,
    normalizer: ResidualNormalizer,
    sample: TeacherSample,
    *,
    device: str,
):
    node, pair = (
        normalizer.normalize(
            sample.encoded
        )
    )
    dtype = next(
        model.parameters()
    ).dtype
    with torch.no_grad():
        resistance, reactance, channels = (
            model.forward_structured(
                torch.as_tensor(
                    node,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    pair,
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
            )
        )
    impedance = (
        resistance.detach().cpu().numpy()
        + 1j
        * reactance.detach().cpu().numpy()
    )
    return (
        impedance,
        channels.detach().cpu().numpy(),
    )


def _validation_metrics(
    model,
    normalizer: ResidualNormalizer,
    samples,
    *,
    device: str,
    channel_weight: float,
):
    samples = tuple(
        samples
    )
    if not samples:
        raise ValueError(
            "validation samples are empty"
        )
    z_errors = []
    channel_errors = []
    for sample in samples:
        predicted_z, predicted_channels = (
            _predict_sample_numpy(
                model,
                normalizer,
                sample,
                device=device,
            )
        )
        z_errors.append(
            _relative_z_error(
                predicted_z,
                sample.target_impedance,
            )
        )
        if (
            sample.target_dissipation_channels
            is not None
        ):
            channel_errors.append(
                _relative_channel_error(
                    predicted_channels,
                    sample.target_dissipation_channels,
                )
            )
    z_mean = float(
        np.mean(
            z_errors
        )
    )
    channel_mean = (
        None
        if not channel_errors
        else float(
            np.mean(
                channel_errors
            )
        )
    )
    score = z_mean
    if (
        channel_mean is not None
        and channel_weight > 0.0
    ):
        score += (
            channel_weight
            * channel_mean
        )
    return (
        float(score),
        z_mean,
        channel_mean,
    )


class NeuralResidualArtifact:
    def __init__(
        self,
        model: PhysicsFactoredResidualNet,
        normalizer: ResidualNormalizer,
        *,
        baseline_segments: int = 96,
        device: str = "cpu",
    ):
        self.model = model
        self.normalizer = normalizer
        self.baseline_segments = int(
            baseline_segments
        )
        self.device = str(
            device
        )
        self.model.to(
            self.device
        )

    def _encoded_baseline(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        encoded = (
            encode_scene_invariant(
                scene,
                frequency_hz,
            )
        )
        node, pair = (
            self.normalizer.normalize(
                encoded
            )
        )
        baseline = (
            analytic_port_baseline(
                scene,
                frequency_hz,
                segments_per_coil=(
                    self.baseline_segments
                ),
            )
        )
        return (
            node,
            pair,
            baseline,
        )

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> NeuralPortPrediction:
        node, pair, baseline = (
            self._encoded_baseline(
                scene,
                frequency_hz,
            )
        )
        dtype = next(
            self.model.parameters()
        ).dtype
        device = next(
            self.model.parameters()
        ).device
        self.model.eval()
        with torch.no_grad():
            (
                resistance,
                reactance,
                channels,
            ) = self.model.forward_structured(
                torch.as_tensor(
                    node,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    pair,
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
                    * frequency_hz
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
            )
        impedance = (
            resistance.detach().cpu().numpy()
            + 1j
            * reactance.detach().cpu().numpy()
        )
        return NeuralPortPrediction(
            impedance,
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
        payload = {
            "schema": ARTIFACT_SCHEMA,
            "model_config": {
                "node_dim": self.model.node_dim,
                "pair_dim": self.model.pair_dim,
                "hidden_dim": self.model.hidden_dim,
                "factor_rank": self.model.factor_rank,
                "depth": self.model.depth,
            },
            "model_state": (
                self.model.state_dict()
            ),
            "normalizer": {
                key: (
                    torch.as_tensor(value)
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value
                in self.normalizer.to_dict().items()
            },
            "baseline_segments": (
                self.baseline_segments
            ),
        }
        torch.save(
            payload,
            Path(path),
        )

    @staticmethod
    def load(
        path,
        *,
        device: str = "cpu",
    ) -> "NeuralResidualArtifact":
        try:
            payload = torch.load(
                Path(path),
                map_location=device,
                weights_only=True,
            )
        except TypeError:
            # torch 2.2 compatibility: weights_only existed in later minor
            # releases. The artifact contains only tensors and primitives.
            payload = torch.load(
                Path(path),
                map_location=device,
            )
        if (
            payload.get(
                "schema"
            )
            != ARTIFACT_SCHEMA
        ):
            raise ValueError(
                "unsupported vNext neural artifact schema"
            )
        model = (
            PhysicsFactoredResidualNet(
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
        model.eval()
        return NeuralResidualArtifact(
            model,
            ResidualNormalizer.from_dict(
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


def train_residual_surrogate(
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
    baseline_segments: int | None = None,
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
            "at least one teacher sample is required"
        )
    if (
        epochs < 1
        or learning_rate <= 0.0
        or channel_loss_weight < 0.0
        or validation_channel_weight < 0.0
        or patience < 1
        or validation_interval < 1
        or min_improvement < 0.0
    ):
        raise ValueError(
            "invalid training configuration"
        )
    if (
        channel_loss_weight > 0.0
        and any(
            sample.target_dissipation_channels
            is None
            for sample in samples
        )
    ):
        raise ValueError(
            "channel training requires physical dissipation-channel labels"
        )

    torch.manual_seed(
        seed
    )
    np.random.seed(
        seed
    )

    sample_baseline_segments = {
        int(
            sample.baseline_segments
        )
        for sample in samples
    }
    if (
        len(
            sample_baseline_segments
        )
        != 1
    ):
        raise ValueError(
            "all teacher samples must use the same analytic baseline resolution"
        )
    sample_baseline_segments = (
        sample_baseline_segments.pop()
    )
    if baseline_segments is None:
        baseline_segments = (
            sample_baseline_segments
        )
    elif (
        int(
            baseline_segments
        )
        != sample_baseline_segments
    ):
        raise ValueError(
            "training baseline_segments disagrees with teacher samples"
        )

    normalizer = (
        ResidualNormalizer.fit(
            samples
        )
    )
    model = (
        PhysicsFactoredResidualNet(
            node_dim=(
                samples[
                    0
                ].encoded.node_features.shape[
                    -1
                ]
            ),
            pair_dim=(
                samples[
                    0
                ].encoded.pair_features.shape[
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

    final_loss = np.inf
    best_epoch = 0
    best_validation_score = None
    best_validation_z_error = None
    best_validation_channel_error = None
    best_state = None
    stale = 0
    stopped_early = False
    epochs_run = 0

    for epoch in range(
        1,
        epochs + 1,
    ):
        order = (
            np.random.permutation(
                len(samples)
            )
        )
        epoch_loss = 0.0
        model.train()

        for index in order:
            sample = samples[
                int(index)
            ]
            node, pair = (
                normalizer.normalize(
                    sample.encoded
                )
            )
            target = np.asarray(
                sample.target_impedance,
                dtype=complex,
            )
            optimizer.zero_grad(
                set_to_none=True
            )
            (
                resistance,
                reactance,
                channels,
            ) = (
                model.forward_structured(
                    torch.as_tensor(
                        node,
                        dtype=dtype,
                        device=device,
                    ),
                    torch.as_tensor(
                        pair,
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
                )
            )

            target_r = (
                torch.as_tensor(
                    target.real,
                    dtype=dtype,
                    device=device,
                )
            )
            target_x = (
                torch.as_tensor(
                    target.imag,
                    dtype=dtype,
                    device=device,
                )
            )
            r_denom = (
                torch.mean(
                    target_r
                    * target_r
                )
                + 1e-18
            )
            x_denom = (
                torch.mean(
                    target_x
                    * target_x
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

            if (
                channel_loss_weight
                > 0.0
            ):
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
            / len(samples)
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
                validation_z_error,
                validation_channel_error,
            ) = _validation_metrics(
                model,
                normalizer,
                validation_samples,
                device=device,
                channel_weight=(
                    validation_channel_weight
                ),
            )
            if (
                best_validation_score
                is None
                or score
                < best_validation_score
                - min_improvement
            ):
                best_validation_score = (
                    score
                )
                best_validation_z_error = (
                    validation_z_error
                )
                best_validation_channel_error = (
                    validation_channel_error
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
            "training completed without a selectable model state"
        )
    model.load_state_dict(
        best_state
    )
    model.eval()

    artifact = (
        NeuralResidualArtifact(
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
        NeuralTrainingReport(
            float(
                final_loss
            ),
            int(
                epochs_run
            ),
            len(
                samples
            ),
            int(
                best_epoch
            ),
            (
                None
                if best_validation_score
                is None
                else float(
                    best_validation_score
                )
            ),
            (
                None
                if best_validation_z_error
                is None
                else float(
                    best_validation_z_error
                )
            ),
            (
                None
                if best_validation_channel_error
                is None
                else float(
                    best_validation_channel_error
                )
            ),
            bool(
                stopped_early
            ),
        ),
    )
