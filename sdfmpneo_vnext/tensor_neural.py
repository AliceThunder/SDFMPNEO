from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "sdfmpneo_vnext.tensor_neural requires the 'neural' extra"
    ) from exc

from .analytic_baseline import analytic_port_baseline
from .device import resolve_torch_device
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
from .tensor_features import (
    TENSOR_COIL_FEATURE_DIM,
    TENSOR_PACKAGE_FEATURE_DIM,
    TENSOR_PAIR_FEATURE_DIM,
    TENSOR_CROSS_FEATURE_DIM,
    _material_tensors,
    encode_tensor_hybrid_scene_invariant,
)
from .tensor_training_data import TensorHybridTeacherSample


TENSOR_HYBRID_ARTIFACT_SCHEMA = 1


@dataclass(frozen=True)
class TensorHybridTrainingReport:
    final_loss: float
    epochs: int
    samples: int
    best_epoch: int
    best_validation_score: float | None
    stopped_early: bool


def _eigenvalue_range(matrices, *, nonnegative: bool):
    values = []
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=float)
        eigenvalues = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
        values.extend(float(value) for value in eigenvalues)
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    if nonnegative:
        values = np.maximum(values, 0.0)
    return float(np.min(values)), float(np.max(values))


def _sample_tensor_ranges(samples):
    background_epsilon = []
    background_sigma = []
    package_epsilon = []
    package_sigma = []
    permeability = []
    for sample in samples:
        epsilon, sigma = _material_tensors(
            sample.scene.medium,
            sample.frequency_hz,
        )
        background_epsilon.append(epsilon)
        background_sigma.append(sigma)
        for package in sample.scene.packages:
            epsilon, sigma = _material_tensors(
                package.material,
                sample.frequency_hz,
            )
            package_epsilon.append(epsilon)
            package_sigma.append(sigma)
            permeability.append(float(package.material.relative_permeability))
    if not permeability:
        raise ValueError("tensor hybrid training requires at least one package")
    return {
        "background_epsilon": _eigenvalue_range(
            background_epsilon,
            nonnegative=False,
        ),
        "background_sigma": _eigenvalue_range(
            background_sigma,
            nonnegative=True,
        ),
        "package_epsilon": _eigenvalue_range(
            package_epsilon,
            nonnegative=False,
        ),
        "package_sigma": _eigenvalue_range(
            package_sigma,
            nonnegative=True,
        ),
        "package_mu": (
            float(np.min(permeability)),
            float(np.max(permeability)),
        ),
    }


def _range_contains(values, bounds, *, name: str, tolerance: float = 1e-10):
    if bounds is None:
        return
    lower, upper = bounds
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    scale = max(abs(lower), abs(upper), 1.0)
    slack = tolerance * scale
    if minimum < lower - slack or maximum > upper + slack:
        raise ValueError(
            f"{name} is outside the tensor FAST training domain "
            f"[{lower:.6g}, {upper:.6g}]"
        )


def _state_dtype(state_dict):
    for value in state_dict.values():
        if torch.is_floating_point(value):
            return value.dtype
    return torch.get_default_dtype()


class TensorHybridNeuralResidualArtifact:
    """Tensor-aware FAST port artifact with hard passive channel decoding."""

    supports_packages = True
    supports_lossy_background = True
    supports_tensor_electric = True

    def __init__(
        self,
        model: HybridPhysicsFactoredResidualNet,
        normalizer: HybridNormalizer,
        *,
        baseline_segments: int,
        material_domain,
        geometry_domain=None,
        device: str = "cpu",
    ):
        if model.coil_dim != TENSOR_COIL_FEATURE_DIM:
            raise ValueError("tensor artifact requires the tensor coil feature schema")
        if model.package_dim != TENSOR_PACKAGE_FEATURE_DIM:
            raise ValueError("tensor artifact requires the tensor package feature schema")
        if (
            model.coil_pair_dim != TENSOR_PAIR_FEATURE_DIM
            or model.package_pair_dim != TENSOR_PAIR_FEATURE_DIM
            or model.cross_dim != TENSOR_CROSS_FEATURE_DIM
        ):
            raise ValueError("tensor artifact pair/cross dimensions are incompatible")
        if baseline_segments < 8:
            raise ValueError("baseline_segments must be >= 8")
        self.model = model
        self.normalizer = normalizer
        self.baseline_segments = int(baseline_segments)
        self.material_domain = dict(material_domain)
        self.geometry_domain = geometry_domain
        self.device = resolve_torch_device(device)
        dtype = next(self.model.parameters()).dtype
        if self.device == "mps" and dtype == torch.float64:
            raise ValueError(
                "float64 tensor FAST artifacts cannot run on MPS; use cpu or float32"
            )
        self.model.to(self.device)
        self.model.eval()

    def _validate_material_domain(self, scene: Scene, frequency_hz: float):
        epsilon, sigma = _material_tensors(scene.medium, frequency_hz)
        _range_contains(
            np.linalg.eigvalsh(epsilon),
            self.material_domain.get("background_epsilon"),
            name="background permittivity principal values",
        )
        _range_contains(
            np.linalg.eigvalsh(sigma),
            self.material_domain.get("background_sigma"),
            name="background conductivity principal values",
        )
        for package in scene.packages:
            epsilon, sigma = _material_tensors(package.material, frequency_hz)
            _range_contains(
                np.linalg.eigvalsh(epsilon),
                self.material_domain.get("package_epsilon"),
                name="package permittivity principal values",
            )
            _range_contains(
                np.linalg.eigvalsh(sigma),
                self.material_domain.get("package_sigma"),
                name="package conductivity principal values",
            )
            _range_contains(
                np.asarray([package.material.relative_permeability], dtype=float),
                self.material_domain.get("package_mu"),
                name="package relative permeability",
            )

    def predict_structured(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> StructuredPortPrediction:
        validate_package_conductor_topology(scene)
        validate_hybrid_geometry_domain(
            scene,
            frequency_hz,
            self.geometry_domain,
        )
        self._validate_material_domain(scene, frequency_hz)
        encoded = encode_tensor_hybrid_scene_invariant(scene, frequency_hz)
        normalized = self.normalizer.normalize(encoded)
        baseline = analytic_port_baseline(
            Scene(scene.coils, scene.medium, ()),
            frequency_hz,
            segments_per_coil=self.baseline_segments,
        )
        dtype = next(self.model.parameters()).dtype
        device = self.device
        with torch.no_grad():
            resistance, reactance, channels = self.model.forward_structured(
                *[
                    torch.as_tensor(value, dtype=dtype, device=device)
                    for value in normalized
                ],
                torch.as_tensor(
                    baseline.resistance,
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    2.0
                    * np.pi
                    * float(frequency_hz)
                    * baseline.inductance,
                    dtype=dtype,
                    device=device,
                ),
                resistance_scale=self.normalizer.resistance_scale,
                reactance_scale=self.normalizer.reactance_scale,
                dielectric_loss_gate=_dielectric_loss_gate(scene, frequency_hz),
                reactance_gate=_reactance_gate(frequency_hz),
            )
        impedance = (
            resistance.detach().cpu().numpy()
            + 1j * reactance.detach().cpu().numpy()
        )
        channel_values = channels.detach().cpu().numpy()
        return StructuredPortPrediction(
            impedance,
            channel_values,
            tuple(f"coil:{index}" for index in range(len(scene.coils)))
            + ("electric_environment:aggregate",),
        )

    def predict(self, scene: Scene, frequency_hz: float) -> np.ndarray:
        return self.predict_structured(scene, frequency_hz).impedance

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": TENSOR_HYBRID_ARTIFACT_SCHEMA,
                "model": {
                    "coil_dim": self.model.coil_dim,
                    "coil_pair_dim": self.model.coil_pair_dim,
                    "package_dim": self.model.package_dim,
                    "cross_dim": self.model.cross_dim,
                    "package_pair_dim": self.model.package_pair_dim,
                    "hidden_dim": self.model.hidden_dim,
                    "factor_rank": self.model.factor_rank,
                    "depth": self.model.depth,
                    "state_dict": self.model.state_dict(),
                },
                "normalizer": self.normalizer.to_dict(),
                "baseline_segments": self.baseline_segments,
                "material_domain": self.material_domain,
                "geometry_domain": self.geometry_domain,
            },
            path,
        )

    @staticmethod
    def load(
        path,
        *,
        device: str = "cpu",
    ) -> "TensorHybridNeuralResidualArtifact":
        resolved_device = resolve_torch_device(device)
        payload = torch.load(
            Path(path),
            map_location="cpu",
            weights_only=False,
        )
        if int(payload.get("schema", -1)) != TENSOR_HYBRID_ARTIFACT_SCHEMA:
            raise ValueError("unsupported tensor hybrid artifact schema")
        metadata = payload["model"]
        state_dict = metadata["state_dict"]
        dtype = _state_dtype(state_dict)
        if resolved_device == "mps" and dtype == torch.float64:
            raise ValueError(
                "this tensor artifact uses float64 and cannot be loaded on MPS"
            )
        model = HybridPhysicsFactoredResidualNet(
            coil_dim=int(metadata["coil_dim"]),
            coil_pair_dim=int(metadata["coil_pair_dim"]),
            package_dim=int(metadata["package_dim"]),
            cross_dim=int(metadata["cross_dim"]),
            package_pair_dim=int(metadata["package_pair_dim"]),
            hidden_dim=int(metadata["hidden_dim"]),
            factor_rank=int(metadata["factor_rank"]),
            depth=int(metadata["depth"]),
        ).to(dtype=dtype)
        model.load_state_dict(state_dict)
        return TensorHybridNeuralResidualArtifact(
            model,
            HybridNormalizer.from_dict(payload["normalizer"]),
            baseline_segments=int(payload["baseline_segments"]),
            material_domain=payload["material_domain"],
            geometry_domain=payload.get("geometry_domain"),
            device=resolved_device,
        )


def _tensor_sample_loss(
    model,
    normalizer,
    sample,
    *,
    channel_loss_weight: float,
    device: str,
):
    normalized = normalizer.normalize(sample.encoded)
    dtype = next(model.parameters()).dtype
    resistance, reactance, channels = model.forward_structured(
        *[
            torch.as_tensor(value, dtype=dtype, device=device)
            for value in normalized
        ],
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
        resistance_scale=normalizer.resistance_scale,
        reactance_scale=normalizer.reactance_scale,
        dielectric_loss_gate=_dielectric_loss_gate(
            sample.scene,
            sample.frequency_hz,
        ),
        reactance_gate=_reactance_gate(sample.frequency_hz),
    )
    complex_dtype = (
        torch.complex128 if dtype == torch.float64 else torch.complex64
    )
    target_impedance = torch.as_tensor(
        sample.target_impedance,
        dtype=complex_dtype,
        device=device,
    )
    target_channels = torch.as_tensor(
        sample.target_dissipation_channels,
        dtype=complex_dtype,
        device=device,
    )
    resistance_error = (
        resistance - target_impedance.real
    ) / max(normalizer.resistance_scale, 1e-12)
    reactance_error = (
        reactance - target_impedance.imag
    ) / max(normalizer.reactance_scale, 1e-12)
    impedance_loss = torch.mean(resistance_error**2) + torch.mean(
        reactance_error**2
    )
    channel_scale = torch.clamp(
        torch.sqrt(torch.mean(torch.abs(target_channels) ** 2)),
        min=torch.as_tensor(1e-12, dtype=dtype, device=device),
    )
    channel_loss = torch.mean(
        torch.abs(channels - target_channels) ** 2
    ) / channel_scale**2
    return impedance_loss + float(channel_loss_weight) * channel_loss


def _legacy_train_tensor_hybrid_residual_surrogate(
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
    device: str = "cpu",
):
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
    ):
        raise ValueError("invalid tensor hybrid training configuration")

    baseline_segments = {int(sample.baseline_segments) for sample in samples}
    if len(baseline_segments) != 1:
        raise ValueError("all tensor hybrid samples must use one baseline resolution")
    baseline_segments = baseline_segments.pop()

    all_samples = samples + validation_samples
    for sample in all_samples:
        if not isinstance(sample, TensorHybridTeacherSample):
            raise TypeError(
                "tensor trainer requires TensorHybridTeacherSample instances"
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
    normalizer = HybridNormalizer.fit(samples)
    torch.manual_seed(int(seed))
    model = HybridPhysicsFactoredResidualNet(
        coil_dim=TENSOR_COIL_FEATURE_DIM,
        coil_pair_dim=TENSOR_PAIR_FEATURE_DIM,
        package_dim=TENSOR_PACKAGE_FEATURE_DIM,
        cross_dim=TENSOR_CROSS_FEATURE_DIM,
        package_pair_dim=TENSOR_PAIR_FEATURE_DIM,
        hidden_dim=hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
    ).to(resolved_device).double()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    rng = np.random.default_rng(int(seed))
    best_state = None
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    final_loss = float("inf")
    stopped_early = False
    epochs_run = 0

    for epoch in range(1, int(epochs) + 1):
        model.train()
        total = 0.0
        for index in rng.permutation(len(samples)):
            optimizer.zero_grad(set_to_none=True)
            loss = _tensor_sample_loss(
                model,
                normalizer,
                samples[int(index)],
                channel_loss_weight=channel_loss_weight,
                device=resolved_device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total += float(loss.detach().cpu())
        final_loss = total / len(samples)
        epochs_run = epoch

        model.eval()
        evaluation_set = validation_samples if validation_samples else samples
        with torch.no_grad():
            score = float(
                np.mean(
                    [
                        float(
                            _tensor_sample_loss(
                                model,
                                normalizer,
                                sample,
                                channel_loss_weight=channel_loss_weight,
                                device=resolved_device,
                            ).detach().cpu()
                        )
                        for sample in evaluation_set
                    ]
                )
            )
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
            if stale >= patience:
                stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
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


def train_tensor_hybrid_residual_surrogate(
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
    batch_size: int = 1,
    precision: str = "float64",
    device: str = "cpu",
):
    """Train tensor FAST ports, optionally using true vectorized batches.

    The historical default remains CPU/FP64 with one sample per optimizer
    update. Set ``batch_size > 1`` or ``precision='auto'``/``float32`` or
    ``device='auto'`` to use the accelerated same-topology batching path.
    """
    normalized_precision = str(precision).strip().lower()
    legacy_precision = normalized_precision in {"float64", "fp64", "double"}
    use_accelerated = (
        int(batch_size) != 1
        or not legacy_precision
        or str(device).strip().lower() == "auto"
    )
    if use_accelerated:
        from .performance import train_tensor_hybrid_residual_surrogate_accelerated

        return train_tensor_hybrid_residual_surrogate_accelerated(
            samples,
            validation_samples=validation_samples,
            hidden_dim=hidden_dim,
            factor_rank=factor_rank,
            depth=depth,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            channel_loss_weight=channel_loss_weight,
            patience=patience,
            min_improvement=min_improvement,
            seed=seed,
            geometry_domain=geometry_domain,
            batch_size=batch_size,
            precision=precision,
            device=device,
        )
    return _legacy_train_tensor_hybrid_residual_surrogate(
        samples,
        validation_samples=validation_samples,
        hidden_dim=hidden_dim,
        factor_rank=factor_rank,
        depth=depth,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        channel_loss_weight=channel_loss_weight,
        patience=patience,
        min_improvement=min_improvement,
        seed=seed,
        geometry_domain=geometry_domain,
        device=device,
    )
