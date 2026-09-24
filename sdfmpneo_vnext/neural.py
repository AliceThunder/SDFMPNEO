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


ARTIFACT_SCHEMA = 1


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
            target = sample.target_impedance
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
        return ResidualNormalizer(
            np.asarray(data["node_mean"], dtype=float),
            np.asarray(data["node_scale"], dtype=float),
            np.asarray(data["pair_mean"], dtype=float),
            np.asarray(data["pair_scale"], dtype=float),
            float(data["resistance_scale"]),
            float(data["reactance_scale"]),
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
        if hidden_dim < 4 or factor_rank < 1 or depth < 1:
            raise ValueError(
                "invalid neural residual dimensions"
            )
        self.node_dim = int(node_dim)
        self.pair_dim = int(pair_dim)
        self.hidden_dim = int(hidden_dim)
        self.factor_rank = int(factor_rank)
        self.depth = int(depth)

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
                ).sum(dim=0) / math.sqrt(
                    len(incoming)
                )
            else:
                aggregate = torch.zeros_like(
                    h[i]
                )
            messages.append(aggregate)
        updated = self.node_update(
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

        factors = self.loss_factor_head(
            updated
        )
        r_residual = (
            float(resistance_scale)
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

        x_residual = torch.zeros_like(
            baseline_reactance
        )
        diag = (
            float(reactance_scale)
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
                    * float(reactance_scale)
                    * (
                        forward + reverse
                    )
                )
                x_residual[i, j] = value
                x_residual[j, i] = value
        reactance = (
            baseline_reactance
            + x_residual
        )
        return resistance, reactance


@dataclass
class NeuralTrainingReport:
    final_loss: float
    epochs: int
    samples: int
    best_epoch: int
    best_validation_error: float | None
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
            predicted - target
        )
        / max(
            np.linalg.norm(target),
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
    node, pair = normalizer.normalize(
        sample.encoded
    )
    dtype = next(
        model.parameters()
    ).dtype
    with torch.no_grad():
        resistance, reactance = model(
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
    return (
        resistance.detach().cpu().numpy()
        + 1j
        * reactance.detach().cpu().numpy()
    )


def _validation_error(
    model,
    normalizer: ResidualNormalizer,
    samples,
    *,
    device: str,
) -> float:
    samples = tuple(samples)
    if not samples:
        raise ValueError(
            "validation samples are empty"
        )
    errors = [
        _relative_z_error(
            _predict_sample_numpy(
                model,
                normalizer,
                sample,
                device=device,
            ),
            sample.target_impedance,
        )
        for sample in samples
    ]
    return float(
        np.mean(errors)
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
        self.device = str(device)
        self.model.to(self.device)

    def predict(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> np.ndarray:
        encoded = encode_scene_invariant(
            scene,
            frequency_hz,
        )
        node, pair = (
            self.normalizer.normalize(
                encoded
            )
        )
        baseline = analytic_port_baseline(
            scene,
            frequency_hz,
            segments_per_coil=self.baseline_segments,
        )
        dtype = next(
            self.model.parameters()
        ).dtype
        device = next(
            self.model.parameters()
        ).device
        with torch.no_grad():
            resistance, reactance = (
                self.model(
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
            )
        return (
            resistance.detach().cpu().numpy()
            + 1j
            * reactance.detach().cpu().numpy()
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
            "model_state": self.model.state_dict(),
            "normalizer": self.normalizer.to_dict(),
            "baseline_segments": self.baseline_segments,
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
        payload = torch.load(
            Path(path),
            map_location=device,
        )
        if payload.get("schema") != ARTIFACT_SCHEMA:
            raise ValueError(
                "unsupported vNext neural artifact schema"
            )
        model = PhysicsFactoredResidualNet(
            **payload["model_config"]
        )
        model.load_state_dict(
            payload["model_state"]
        )
        model.eval()
        return NeuralResidualArtifact(
            model,
            ResidualNormalizer.from_dict(
                payload["normalizer"]
            ),
            baseline_segments=int(
                payload["baseline_segments"]
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
    patience: int = 30,
    validation_interval: int = 1,
    min_improvement: float = 1e-5,
    seed: int = 17,
    baseline_segments: int | None = None,
    device: str = "cpu",
):
    samples = tuple(samples)
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
        or patience < 1
        or validation_interval < 1
        or min_improvement < 0.0
    ):
        raise ValueError(
            "invalid training configuration"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)

    sample_baseline_segments = {
        int(s.baseline_segments)
        for s in samples
    }
    if len(sample_baseline_segments) != 1:
        raise ValueError(
            "all teacher samples must use the same analytic baseline resolution"
        )
    sample_baseline_segments = sample_baseline_segments.pop()
    if baseline_segments is None:
        baseline_segments = sample_baseline_segments
    elif int(baseline_segments) != sample_baseline_segments:
        raise ValueError(
            "training baseline_segments disagrees with teacher samples"
        )

    normalizer = ResidualNormalizer.fit(
        samples
    )
    model = PhysicsFactoredResidualNet(
        node_dim=samples[0].encoded.node_features.shape[-1],
        pair_dim=samples[0].encoded.pair_features.shape[-1],
        hidden_dim=hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    dtype = next(
        model.parameters()
    ).dtype

    final_loss = np.inf
    best_epoch = 0
    best_validation_error = None
    best_state = None
    stale = 0
    stopped_early = False
    epochs_run = 0

    for epoch in range(1, epochs + 1):
        order = np.random.permutation(
            len(samples)
        )
        epoch_loss = 0.0
        model.train()
        for index in order:
            sample = samples[int(index)]
            node, pair = (
                normalizer.normalize(
                    sample.encoded
                )
            )
            target = sample.target_impedance
            optimizer.zero_grad(
                set_to_none=True
            )
            resistance, reactance = model(
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
            target_r = torch.as_tensor(
                target.real,
                dtype=dtype,
                device=device,
            )
            target_x = torch.as_tensor(
                target.imag,
                dtype=dtype,
                device=device,
            )
            r_denom = (
                torch.mean(
                    target_r * target_r
                )
                + 1e-18
            )
            x_denom = (
                torch.mean(
                    target_x * target_x
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
                epoch % validation_interval
                != 0
                and epoch != epochs
            ):
                continue
            model.eval()
            score = _validation_error(
                model,
                normalizer,
                validation_samples,
                device=device,
            )
            if (
                best_validation_error is None
                or score
                < best_validation_error
                - min_improvement
            ):
                best_validation_error = score
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

    artifact = NeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=baseline_segments,
        device=device,
    )
    return (
        artifact,
        NeuralTrainingReport(
            float(final_loss),
            int(epochs_run),
            len(samples),
            int(best_epoch),
            (
                None
                if best_validation_error is None
                else float(best_validation_error)
            ),
            bool(stopped_early),
        ),
    )
