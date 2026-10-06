from __future__ import annotations

from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import math

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.performance requires the 'neural' extra"
    ) from exc

from .analytic_baseline import analytic_port_baseline
from .em import MQSConfig
from .hybrid_domain import (
    validate_hybrid_geometry_domain,
    validate_package_conductor_topology,
)
from .hybrid_neural import (
    HybridNormalizer,
    HybridPhysicsFactoredResidualNet,
    _dielectric_loss_gate,
    _reactance_gate,
)
from .prediction import StructuredPortPrediction
from .scene import Scene
from .serialization import scene_to_dict
from .tensor_features import (
    TENSOR_COIL_FEATURE_DIM,
    TENSOR_PACKAGE_FEATURE_DIM,
    TENSOR_PAIR_FEATURE_DIM,
    TENSOR_CROSS_FEATURE_DIM,
    encode_tensor_hybrid_scene_invariant,
)
from .tensor_neural import (
    TensorHybridNeuralResidualArtifact,
    TensorHybridTrainingReport,
    _sample_tensor_ranges,
)
from .tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from .tensor_spatial_training_data import TensorHybridSpatialTeacherSample
from .tensor_training_data import TensorHybridTeacherSample


def resolve_torch_device(device: str | None = "auto") -> str:
    requested = "auto" if device is None else str(device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and bool(mps.is_available()):
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false"
        )
    if requested == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not bool(mps.is_available()):
            raise RuntimeError("MPS was requested but is not available")
    return str(device)


def resolve_training_dtype(precision: str = "auto", *, device: str):
    value = str(precision).strip().lower()
    if value == "auto":
        return torch.float32 if device.startswith(("cuda", "mps")) else torch.float64
    if value in {"float32", "fp32", "single"}:
        return torch.float32
    if value in {"float64", "fp64", "double"}:
        if device == "mps":
            raise ValueError("MPS does not support float64 tensor FAST training")
        return torch.float64
    raise ValueError("precision must be auto, float32, or float64")


def _batched_matrix_sqrt(matrix, *, inverse: bool):
    matrix = 0.5 * (matrix + matrix.conj().transpose(-1, -2))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    scale = torch.clamp(
        torch.amax(torch.abs(eigenvalues), dim=-1, keepdim=True),
        min=1e-12,
    )
    floor = 1e-10 * scale + 1e-14
    clipped = torch.maximum(eigenvalues.real, floor)
    diagonal = torch.rsqrt(clipped) if inverse else torch.sqrt(clipped)
    return (
        eigenvectors
        @ torch.diag_embed(diagonal.to(eigenvectors.dtype))
        @ eigenvectors.conj().transpose(-1, -2)
    )


def _aggregate(messages, reference):
    if not messages:
        return torch.zeros_like(reference)
    return torch.stack(messages, dim=0).sum(dim=0) / math.sqrt(len(messages))


def _batched_latent(
    model,
    coil_features,
    coil_pair_features,
    package_features,
    coil_package_features,
    package_pair_features,
):
    coil0 = model.coil_encoder(coil_features)
    package0 = model.package_encoder(package_features)
    _, n_coils, _ = coil0.shape
    _, n_packages, _ = package0.shape
    if n_packages < 1:
        raise ValueError("tensor hybrid batches require at least one package")

    coil_pair_messages = []
    for i in range(n_coils):
        messages = []
        for j in range(n_coils):
            if i == j:
                continue
            messages.append(
                model.coil_pair_encoder(
                    torch.cat(
                        (coil0[:, i], coil0[:, j], coil_pair_features[:, i, j]),
                        dim=-1,
                    )
                )
            )
        coil_pair_messages.append(_aggregate(messages, coil0[:, i]))

    package_pair_messages = []
    coil_to_package_messages = []
    for p in range(n_packages):
        pair_messages = []
        for q in range(n_packages):
            if p == q:
                continue
            pair_messages.append(
                model.package_pair_encoder(
                    torch.cat(
                        (package0[:, p], package0[:, q], package_pair_features[:, p, q]),
                        dim=-1,
                    )
                )
            )
        package_pair_messages.append(_aggregate(pair_messages, package0[:, p]))
        coil_messages = []
        for c in range(n_coils):
            coil_messages.append(
                model.coil_to_package_encoder(
                    torch.cat(
                        (package0[:, p], coil0[:, c], coil_package_features[:, c, p]),
                        dim=-1,
                    )
                )
            )
        coil_to_package_messages.append(_aggregate(coil_messages, package0[:, p]))

    package = torch.stack(
        [
            model.package_update(
                torch.cat(
                    (package0[:, p], package_pair_messages[p], coil_to_package_messages[p]),
                    dim=-1,
                )
            )
            for p in range(n_packages)
        ],
        dim=1,
    )

    package_to_coil_messages = []
    for c in range(n_coils):
        messages = []
        for p in range(n_packages):
            messages.append(
                model.package_to_coil_encoder(
                    torch.cat(
                        (coil0[:, c], package[:, p], coil_package_features[:, c, p]),
                        dim=-1,
                    )
                )
            )
        package_to_coil_messages.append(_aggregate(messages, coil0[:, c]))

    coil = torch.stack(
        [
            model.coil_update(
                torch.cat(
                    (coil0[:, c], coil_pair_messages[c], package_to_coil_messages[c]),
                    dim=-1,
                )
            )
            for c in range(n_coils)
        ],
        dim=1,
    )
    return coil, package


def _complex_factor(model, raw):
    real = raw[..., : model.factor_rank]
    imag = raw[..., model.factor_rank :]
    dtype = torch.complex64 if raw.dtype == torch.float32 else torch.complex128
    return real.to(dtype) + 1j * imag.to(dtype)


def forward_structured_batch(
    model: HybridPhysicsFactoredResidualNet,
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
    dielectric_loss_gate,
    reactance_gate,
):
    """Vectorized same-topology forward pass over a batch of scenes."""
    coil, package = _batched_latent(
        model,
        coil_features,
        coil_pair_features,
        package_features,
        coil_package_features,
        package_pair_features,
    )
    batch, n_ports, _ = coil.shape

    factors = model.loss_factor_head(coil)
    resistance = baseline_resistance + float(resistance_scale) * (
        factors @ factors.transpose(-1, -2)
    )
    resistance = 0.5 * (resistance + resistance.transpose(-1, -2))

    diagonal = float(reactance_scale) * model.reactance_diag_head(coil).squeeze(-1)
    residual = torch.diag_embed(diagonal)
    for i in range(n_ports):
        for j in range(i):
            forward = model.reactance_pair_head(
                torch.cat(
                    (coil[:, i], coil[:, j], coil_pair_features[:, i, j]),
                    dim=-1,
                )
            ).squeeze(-1)
            reverse = model.reactance_pair_head(
                torch.cat(
                    (coil[:, j], coil[:, i], coil_pair_features[:, j, i]),
                    dim=-1,
                )
            ).squeeze(-1)
            value = 0.5 * float(reactance_scale) * (forward + reverse)
            pair = torch.zeros_like(residual)
            pair[:, i, j] = value
            pair[:, j, i] = value
            residual = residual + pair
    residual = residual * torch.as_tensor(
        reactance_gate,
        dtype=residual.dtype,
        device=residual.device,
    ).reshape(batch, 1, 1)
    reactance = baseline_reactance + residual
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

    package_pool = torch.mean(package, dim=1)
    rows = []
    for port_index in range(n_ports):
        cross_summary = torch.mean(coil_package_features[:, port_index], dim=1)
        raw = model.dielectric_channel_head(
            torch.cat((package_pool, coil[:, port_index], cross_summary), dim=-1)
        )
        rows.append(_complex_factor(model, raw))
    dielectric_factor = torch.stack(rows, dim=1)
    dielectric_raw = torch.as_tensor(
        dielectric_loss_gate,
        dtype=resistance.dtype,
        device=resistance.device,
    ).reshape(batch, 1, 1) * (
        dielectric_factor @ dielectric_factor.conj().transpose(-1, -2)
    )
    raw_channels.append(dielectric_raw)
    raw_channels = torch.stack(raw_channels, dim=1)

    complex_dtype = torch.complex64 if resistance.dtype == torch.float32 else torch.complex128
    resistance_complex = resistance.to(complex_dtype)
    scale = torch.clamp(
        torch.diagonal(resistance_complex, dim1=-2, dim2=-1).sum(dim=-1).real
        / max(n_ports, 1),
        min=1e-12,
    )
    eye = torch.eye(n_ports, dtype=complex_dtype, device=resistance.device)
    jitter = 1e-8 * scale / max(n_ports, 1)
    conductor_raw = raw_channels[:, :n_ports] + (
        jitter[:, None, None, None] * eye[None, None]
    )
    raw_channels = torch.cat((conductor_raw, raw_channels[:, n_ports:]), dim=1)
    raw_sum = torch.sum(raw_channels, dim=1)
    congruence = _batched_matrix_sqrt(
        resistance_complex,
        inverse=False,
    ) @ _batched_matrix_sqrt(raw_sum, inverse=True)
    channels = (
        congruence[:, None]
        @ raw_channels
        @ congruence.conj().transpose(-1, -2)[:, None]
    )
    channels = 0.5 * (channels + channels.conj().transpose(-1, -2))
    return resistance, reactance, channels


@dataclass(frozen=True)
class _TensorRecord:
    topology: tuple[int, int]
    normalized: tuple[np.ndarray, ...]
    baseline_resistance: np.ndarray
    baseline_reactance: np.ndarray
    target_impedance: np.ndarray
    target_channels: np.ndarray
    dielectric_loss_gate: float
    reactance_gate: float


def _record_from_sample(normalizer, sample):
    return _TensorRecord(
        topology=(len(sample.scene.coils), len(sample.scene.packages)),
        normalized=tuple(
            np.asarray(value, dtype=float)
            for value in normalizer.normalize(sample.encoded)
        ),
        baseline_resistance=np.asarray(sample.baseline_resistance, dtype=float),
        baseline_reactance=np.asarray(sample.baseline_reactance, dtype=float),
        target_impedance=np.asarray(sample.target_impedance, dtype=complex),
        target_channels=np.asarray(sample.target_dissipation_channels, dtype=complex),
        dielectric_loss_gate=_dielectric_loss_gate(sample.scene, sample.frequency_hz),
        reactance_gate=_reactance_gate(sample.frequency_hz),
    )


def _topology_buckets(records):
    buckets = defaultdict(list)
    for index, record in enumerate(records):
        buckets[record.topology].append(index)
    return buckets


def _stack_records(records, indices, *, dtype, device):
    selected = [records[int(index)] for index in indices]
    if len({record.topology for record in selected}) != 1:
        raise ValueError("accelerated tensor batches must use one topology")
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return {
        "normalized": tuple(
            torch.as_tensor(
                np.stack([record.normalized[position] for record in selected], axis=0),
                dtype=dtype,
                device=device,
            )
            for position in range(5)
        ),
        "baseline_resistance": torch.as_tensor(
            np.stack([record.baseline_resistance for record in selected], axis=0),
            dtype=dtype,
            device=device,
        ),
        "baseline_reactance": torch.as_tensor(
            np.stack([record.baseline_reactance for record in selected], axis=0),
            dtype=dtype,
            device=device,
        ),
        "target_impedance": torch.as_tensor(
            np.stack([record.target_impedance for record in selected], axis=0),
            dtype=complex_dtype,
            device=device,
        ),
        "target_channels": torch.as_tensor(
            np.stack([record.target_channels for record in selected], axis=0),
            dtype=complex_dtype,
            device=device,
        ),
        "dielectric_loss_gate": torch.as_tensor(
            [record.dielectric_loss_gate for record in selected],
            dtype=dtype,
            device=device,
        ),
        "reactance_gate": torch.as_tensor(
            [record.reactance_gate for record in selected],
            dtype=dtype,
            device=device,
        ),
    }


def _batch_loss(model, normalizer, batch, *, channel_loss_weight: float):
    resistance, reactance, channels = forward_structured_batch(
        model,
        *batch["normalized"],
        batch["baseline_resistance"],
        batch["baseline_reactance"],
        resistance_scale=normalizer.resistance_scale,
        reactance_scale=normalizer.reactance_scale,
        dielectric_loss_gate=batch["dielectric_loss_gate"],
        reactance_gate=batch["reactance_gate"],
    )
    target = batch["target_impedance"]
    resistance_error = (resistance - target.real) / max(
        normalizer.resistance_scale,
        1e-12,
    )
    reactance_error = (reactance - target.imag) / max(
        normalizer.reactance_scale,
        1e-12,
    )
    impedance_loss = torch.mean(resistance_error**2, dim=(-2, -1)) + torch.mean(
        reactance_error**2,
        dim=(-2, -1),
    )
    target_channels = batch["target_channels"]
    channel_scale = torch.clamp(
        torch.sqrt(
            torch.mean(
                torch.abs(target_channels) ** 2,
                dim=(-3, -2, -1),
            )
        ),
        min=1e-12,
    )
    channel_loss = torch.mean(
        torch.abs(channels - target_channels) ** 2,
        dim=(-3, -2, -1),
    ) / channel_scale**2
    return torch.mean(impedance_loss + float(channel_loss_weight) * channel_loss)


def train_tensor_hybrid_residual_surrogate_accelerated(
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
    patience: int = 30,
    min_improvement: float = 1e-5,
    seed: int = 17,
    geometry_domain=None,
    batch_size: int = 16,
    precision: str = "auto",
    device: str = "auto",
):
    """Tensor FAST port training with true same-topology vectorized batches."""
    samples = tuple(samples)
    validation_samples = tuple(validation_samples)
    if not samples:
        raise ValueError("at least one tensor hybrid training sample is required")
    if (
        epochs < 1
        or learning_rate <= 0.0
        or weight_decay < 0.0
        or channel_loss_weight < 0.0
        or patience < 1
        or min_improvement < 0.0
        or batch_size < 1
    ):
        raise ValueError("invalid accelerated tensor training configuration")

    baseline_segments = {int(sample.baseline_segments) for sample in samples}
    if len(baseline_segments) != 1:
        raise ValueError("all tensor hybrid samples must use one baseline resolution")
    baseline_segments = baseline_segments.pop()

    for sample in samples + validation_samples:
        if not isinstance(sample, TensorHybridTeacherSample):
            raise TypeError(
                "accelerated tensor trainer requires TensorHybridTeacherSample instances"
            )
        validate_package_conductor_topology(sample.scene)
        validate_hybrid_geometry_domain(
            sample.scene,
            sample.frequency_hz,
            geometry_domain,
        )
        if sample.encoded.coil.node_features.shape[1] != TENSOR_COIL_FEATURE_DIM:
            raise ValueError("tensor sample uses an incompatible coil feature schema")
        if sample.encoded.package_features.shape[1] != TENSOR_PACKAGE_FEATURE_DIM:
            raise ValueError("tensor sample uses an incompatible package feature schema")

    resolved_device = resolve_torch_device(device)
    dtype = resolve_training_dtype(precision, device=resolved_device)
    torch.manual_seed(int(seed))
    if resolved_device.startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))

    normalizer = HybridNormalizer.fit(samples)
    model = HybridPhysicsFactoredResidualNet(
        coil_dim=TENSOR_COIL_FEATURE_DIM,
        coil_pair_dim=TENSOR_PAIR_FEATURE_DIM,
        package_dim=TENSOR_PACKAGE_FEATURE_DIM,
        cross_dim=TENSOR_CROSS_FEATURE_DIM,
        package_pair_dim=TENSOR_PAIR_FEATURE_DIM,
        hidden_dim=hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
    ).to(device=resolved_device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )

    training_records = tuple(_record_from_sample(normalizer, sample) for sample in samples)
    validation_records = tuple(
        _record_from_sample(normalizer, sample) for sample in validation_samples
    )
    training_buckets = _topology_buckets(training_records)
    validation_buckets = _topology_buckets(validation_records)

    def iter_batches(records, buckets, *, shuffle: bool):
        keys = list(buckets)
        if shuffle:
            rng.shuffle(keys)
        for key in keys:
            order = np.asarray(buckets[key], dtype=int)
            if shuffle:
                order = rng.permutation(order)
            for start in range(0, len(order), int(batch_size)):
                indices = order[start : start + int(batch_size)]
                yield _stack_records(
                    records,
                    indices,
                    dtype=dtype,
                    device=resolved_device,
                ), len(indices)

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    stopped_early = False
    final_loss = float("inf")
    epochs_run = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        seen = 0
        for batch, count in iter_batches(
            training_records,
            training_buckets,
            shuffle=True,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model,
                normalizer,
                batch,
                channel_loss_weight=channel_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * count
            seen += count
        final_loss = total / max(seen, 1)
        epochs_run = epoch

        model.eval()
        eval_records = validation_records if validation_records else training_records
        eval_buckets = validation_buckets if validation_records else training_buckets
        score_total = 0.0
        score_count = 0
        with torch.no_grad():
            for batch, count in iter_batches(eval_records, eval_buckets, shuffle=False):
                score = _batch_loss(
                    model,
                    normalizer,
                    batch,
                    channel_loss_weight=channel_loss_weight,
                )
                score_total += float(score.detach().cpu()) * count
                score_count += count
        score = score_total / max(score_count, 1)
        if score < best_score - float(min_improvement):
            best_score = score
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

    if best_state is None:
        raise RuntimeError("accelerated tensor training produced no selectable state")
    model.load_state_dict(best_state)
    model.eval()
    artifact = TensorHybridNeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=baseline_segments,
        material_domain=_sample_tensor_ranges(samples),
        geometry_domain=geometry_domain,
        device=resolved_device,
    )
    return artifact, TensorHybridTrainingReport(
        final_loss=float(final_loss),
        epochs=int(epochs_run),
        samples=len(samples),
        best_epoch=int(best_epoch),
        best_validation_score=(
            None if not np.isfinite(best_score) else float(best_score)
        ),
        stopped_early=bool(stopped_early),
    )


@dataclass(frozen=True)
class _PredictionRecord:
    index: int
    topology: tuple[int, int]
    normalized: tuple[np.ndarray, ...]
    baseline_resistance: np.ndarray
    baseline_reactance: np.ndarray
    dielectric_loss_gate: float
    reactance_gate: float


class TensorAcceleratedRuntime:
    """Cached same-topology CUDA batching for tensor FAST port inference."""

    def __init__(self, port_artifact, *, cache_size: int = 256):
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.port_artifact = port_artifact
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()

    def _cache_key(self, scene, frequency_hz):
        return (
            json.dumps(scene_to_dict(scene), sort_keys=True, separators=(",", ":")),
            float(frequency_hz),
            int(self.port_artifact.baseline_segments),
        )

    def _prepare_uncached(self, index, scene, frequency_hz):
        validate_package_conductor_topology(scene)
        validate_hybrid_geometry_domain(
            scene,
            frequency_hz,
            self.port_artifact.geometry_domain,
        )
        self.port_artifact._validate_material_domain(scene, frequency_hz)
        encoded = encode_tensor_hybrid_scene_invariant(scene, frequency_hz)
        normalized = tuple(
            np.asarray(value, dtype=float)
            for value in self.port_artifact.normalizer.normalize(encoded)
        )
        baseline = analytic_port_baseline(
            Scene(scene.coils, scene.medium, ()),
            frequency_hz,
            segments_per_coil=self.port_artifact.baseline_segments,
        )
        return _PredictionRecord(
            index=index,
            topology=(len(scene.coils), len(scene.packages)),
            normalized=normalized,
            baseline_resistance=np.asarray(baseline.resistance, dtype=float),
            baseline_reactance=(
                2.0 * np.pi * float(frequency_hz) * np.asarray(baseline.inductance, dtype=float)
            ),
            dielectric_loss_gate=_dielectric_loss_gate(scene, frequency_hz),
            reactance_gate=_reactance_gate(frequency_hz),
        )

    def _prepare(self, index, scene, frequency_hz):
        if self.cache_size == 0:
            return self._prepare_uncached(index, scene, frequency_hz)
        key = self._cache_key(scene, frequency_hz)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return _PredictionRecord(
                index=index,
                topology=cached.topology,
                normalized=cached.normalized,
                baseline_resistance=cached.baseline_resistance,
                baseline_reactance=cached.baseline_reactance,
                dielectric_loss_gate=cached.dielectric_loss_gate,
                reactance_gate=cached.reactance_gate,
            )
        record = self._prepare_uncached(index, scene, frequency_hz)
        self._cache[key] = record
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return record

    def predict_structured_batch(
        self,
        scenes,
        frequencies_hz,
        *,
        batch_size: int = 64,
    ):
        scenes = tuple(scenes)
        frequencies_hz = tuple(float(value) for value in frequencies_hz)
        if len(scenes) != len(frequencies_hz):
            raise ValueError("scenes and frequencies_hz must have equal length")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not scenes:
            return ()
        records = tuple(
            self._prepare(index, scene, frequency)
            for index, (scene, frequency) in enumerate(zip(scenes, frequencies_hz))
        )
        buckets = defaultdict(list)
        for record in records:
            buckets[record.topology].append(record)
        outputs = [None] * len(records)
        model = self.port_artifact.model
        model.eval()
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        with torch.no_grad():
            for topology, bucket in buckets.items():
                n_coils, _ = topology
                labels = tuple(f"coil:{index}" for index in range(n_coils)) + (
                    "electric_environment:aggregate",
                )
                for start in range(0, len(bucket), int(batch_size)):
                    selected = bucket[start : start + int(batch_size)]
                    normalized = tuple(
                        torch.as_tensor(
                            np.stack(
                                [record.normalized[position] for record in selected],
                                axis=0,
                            ),
                            dtype=dtype,
                            device=device,
                        )
                        for position in range(5)
                    )
                    baseline_r = torch.as_tensor(
                        np.stack([record.baseline_resistance for record in selected], axis=0),
                        dtype=dtype,
                        device=device,
                    )
                    baseline_x = torch.as_tensor(
                        np.stack([record.baseline_reactance for record in selected], axis=0),
                        dtype=dtype,
                        device=device,
                    )
                    gates = torch.as_tensor(
                        [record.dielectric_loss_gate for record in selected],
                        dtype=dtype,
                        device=device,
                    )
                    reactance_gates = torch.as_tensor(
                        [record.reactance_gate for record in selected],
                        dtype=dtype,
                        device=device,
                    )
                    resistance, reactance, channels = forward_structured_batch(
                        model,
                        *normalized,
                        baseline_r,
                        baseline_x,
                        resistance_scale=self.port_artifact.normalizer.resistance_scale,
                        reactance_scale=self.port_artifact.normalizer.reactance_scale,
                        dielectric_loss_gate=gates,
                        reactance_gate=reactance_gates,
                    )
                    impedance = resistance.cpu().numpy() + 1j * reactance.cpu().numpy()
                    channel_values = channels.cpu().numpy()
                    for local_index, record in enumerate(selected):
                        outputs[record.index] = StructuredPortPrediction(
                            impedance[local_index],
                            channel_values[local_index],
                            labels,
                        )
        return tuple(outputs)

    def predict_batch(self, scenes, frequencies_hz, **kwargs):
        return tuple(
            prediction.impedance
            for prediction in self.predict_structured_batch(
                scenes,
                frequencies_hz,
                **kwargs,
            )
        )


def _tensor_teacher_job(payload):
    (
        index,
        seed,
        sampler_config,
        teacher_config,
        teacher_options,
        include_spatial,
        spatial_options,
    ) = payload
    rng = np.random.default_rng([int(seed), int(index)])
    scene, frequency_hz = sample_tensor_hybrid_scene(rng, sampler_config)
    port = TensorHybridTeacherSample.generate(
        scene,
        frequency_hz,
        teacher_config=teacher_config,
        **teacher_options,
    )
    if include_spatial:
        sample = TensorHybridSpatialTeacherSample.generate(
            port,
            teacher_config=teacher_config,
            **spatial_options,
        )
    else:
        sample = port
    return index, sample


def iter_tensor_teacher_samples_parallel(
    count: int,
    *,
    sampler_config: TensorHybridSceneSamplerConfig | None = None,
    teacher_config: MQSConfig | None = None,
    workers: int = 1,
    seed: int = 37,
    include_spatial: bool = False,
    teacher_options=None,
    spatial_options=None,
):
    """Stream deterministic tensor teacher samples using local processes."""
    if count < 1 or workers < 1:
        raise ValueError("count and workers must be positive")
    sampler_config = TensorHybridSceneSamplerConfig() if sampler_config is None else sampler_config
    teacher_config = MQSConfig() if teacher_config is None else teacher_config
    teacher_options = {} if teacher_options is None else dict(teacher_options)
    spatial_options = {} if spatial_options is None else dict(spatial_options)
    jobs = (
        (
            index,
            int(seed),
            sampler_config,
            teacher_config,
            teacher_options,
            bool(include_spatial),
            spatial_options,
        )
        for index in range(int(count))
    )
    if workers == 1:
        for _, sample in map(_tensor_teacher_job, jobs):
            yield sample
        return
    with ProcessPoolExecutor(max_workers=int(workers)) as executor:
        for _, sample in executor.map(_tensor_teacher_job, jobs, chunksize=1):
            yield sample
