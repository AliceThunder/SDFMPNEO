"""Standard mini-batch training for the neural Joule-tensor surrogate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy

import numpy as np

from .network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .physical_layer import decode_heat_source_torch, torch_quadratic_feature
from .surrogate import NeuralTensorSurrogate


@dataclass(frozen=True)
class NeuralTrainingConfig:
    epochs: int = 500
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    heat_loss_weight: float = 0.25
    patience: int = 50
    seed: int = 0
    dtype: str = "float32"
    gradient_clip_norm: float | None = 10.0

    def __post_init__(self) -> None:
        if int(self.epochs) < 1 or int(self.batch_size) < 1 or int(self.patience) < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("invalid optimizer settings")
        if float(self.heat_loss_weight) < 0:
            raise ValueError("heat_loss_weight must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NeuralTrainingReport:
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    train_loss: float
    validation_loss: float
    test_coefficient_rmse: float
    test_relative_packed_error: float
    stopped_early: bool
    device: str


def _coefficient_normalization(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(beta, axis=0)
    scale = np.std(beta, axis=0)
    scale = np.maximum(scale, 1e-12)
    return mean, scale


def _dtype(torch, name: str):
    return torch.float32 if name == "float32" else torch.float64


def train_tensor_surrogate(
    dataset,
    pod,
    *,
    operating_lower: np.ndarray,
    operating_upper: np.ndarray,
    network_config: ResidualMLPConfig | None = None,
    training_config: NeuralTrainingConfig | None = None,
    device: str | None = None,
):
    """Train only the local map ``(a,g)->beta`` with ordinary AdamW.

    Expensive electromagnetic physics is absent from this loop.  The optional
    heat loss uses the already stored tensor labels and a freshly sampled real
    operating vector, so it adds no EM solves.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("install sdfmpneo[neural] to train the neural electrothermal ROM") from exc

    cfg = NeuralTrainingConfig() if training_config is None else training_config
    lo = np.asarray(operating_lower, dtype=float).reshape(-1)
    hi = np.asarray(operating_upper, dtype=float).reshape(-1)
    if lo.shape != (dataset.current_dimension,) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("operating bounds do not match dataset current dimension")

    train_ids = dataset.indices("train")
    val_ids = dataset.indices("validation")
    test_ids = dataset.indices("test")
    x_train = dataset.inputs[train_ids]
    y_train = dataset.outputs[train_ids]
    beta_train = pod.encode(y_train)
    beta_mean, beta_scale = _coefficient_normalization(beta_train)
    input_normalizer = FeatureNormalizer.fit(x_train)

    if network_config is None:
        network_config = ResidualMLPConfig(
            input_dimension=dataset.thermal_rank + dataset.geometry_dimension,
            output_dimension=pod.rank,
        )
    if network_config.input_dimension != dataset.inputs.shape[1] or network_config.output_dimension != pod.rank:
        raise ValueError("network dimensions do not match dataset/POD")

    model = build_residual_mlp(network_config, input_normalizer)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _dtype(torch, cfg.dtype)
    model = model.to(device=resolved_device, dtype=dtype)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    mean_t = torch.as_tensor(pod.mean, dtype=dtype, device=resolved_device)
    basis_t = torch.as_tensor(pod.basis, dtype=dtype, device=resolved_device)
    beta_mean_t = torch.as_tensor(beta_mean, dtype=dtype, device=resolved_device)
    beta_scale_t = torch.as_tensor(beta_scale, dtype=dtype, device=resolved_device)
    lo_t = torch.as_tensor(lo, dtype=dtype, device=resolved_device)
    hi_t = torch.as_tensor(hi, dtype=dtype, device=resolved_device)
    rng = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    def tensors(ids):
        return (
            torch.as_tensor(dataset.inputs[ids], dtype=dtype, device=resolved_device),
            torch.as_tensor(dataset.outputs[ids], dtype=dtype, device=resolved_device),
            torch.as_tensor(pod.encode(dataset.outputs[ids]), dtype=dtype, device=resolved_device),
        )

    x_val, y_val, beta_val = tensors(val_ids)
    x_test, y_test, beta_test = tensors(test_ids)

    def normalized_target(beta):
        return (beta - beta_mean_t) / beta_scale_t

    def batch_loss(xb, yb, beta_b, *, random_operating: bool):
        predicted_normalized = model(xb)
        target_normalized = normalized_target(beta_b)
        coefficient_loss = torch.mean((predicted_normalized - target_normalized) ** 2)
        if cfg.heat_loss_weight == 0.0:
            return coefficient_loss
        predicted_beta = beta_mean_t + beta_scale_t * predicted_normalized
        if random_operating:
            u = lo_t + (hi_t - lo_t) * torch.rand(
                (xb.shape[0], dataset.current_dimension), dtype=dtype, device=resolved_device
            )
        else:
            center = 0.5 * (lo_t + hi_t)
            u = center.unsqueeze(0).expand(xb.shape[0], -1)
        predicted_q = decode_heat_source_torch(
            predicted_beta,
            mean_t,
            basis_t,
            dataset.thermal_rank,
            u,
        )
        true_packed = yb.reshape(
            xb.shape[0], dataset.thermal_rank, dataset.packed_symmetric_size
        )
        feature = torch_quadratic_feature(u)
        true_q = torch.einsum("brs,bs->br", true_packed, feature)
        heat_scale = torch.sqrt(torch.mean(true_q**2)).clamp_min(torch.finfo(dtype).eps)
        heat_loss = torch.mean(((predicted_q - true_q) / heat_scale) ** 2)
        return coefficient_loss + float(cfg.heat_loss_weight) * heat_loss

    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    last_train = float("inf")
    last_validation = float("inf")
    stale = 0
    epochs_completed = 0

    for epoch in range(int(cfg.epochs)):
        model.train()
        shuffled = rng.permutation(train_ids)
        total = 0.0
        count = 0
        for start in range(0, len(shuffled), int(cfg.batch_size)):
            ids = shuffled[start:start + int(cfg.batch_size)]
            xb, yb, beta_b = tensors(ids)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(xb, yb, beta_b, random_operating=True)
            loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.gradient_clip_norm))
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
            count += len(ids)
        last_train = total / max(count, 1)

        model.eval()
        with torch.no_grad():
            val_loss = batch_loss(x_val, y_val, beta_val, random_operating=False)
            last_validation = float(val_loss.detach().cpu())
        epochs_completed = epoch + 1
        threshold = best_validation - 1e-10 * max(1.0, abs(best_validation)) if np.isfinite(best_validation) else float("inf")
        if not np.isfinite(best_validation) or last_validation < threshold:
            best_validation = last_validation
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= int(cfg.patience):
            break

    model.load_state_dict(best_state)
    model.eval()
    surrogate = NeuralTensorSurrogate(
        model,
        pod,
        state_dimension=dataset.thermal_rank,
        geometry_dimension=dataset.geometry_dimension,
        coefficient_mean=beta_mean,
        coefficient_scale=beta_scale,
    )

    with torch.no_grad():
        predicted_normalized = model(x_test)
        predicted_beta = beta_mean_t + beta_scale_t * predicted_normalized
        coefficient_rmse = torch.sqrt(torch.mean((predicted_beta - beta_test) ** 2))
        predicted_packed = mean_t.unsqueeze(0) + predicted_beta @ basis_t.T
        packed_error = torch.linalg.vector_norm(predicted_packed - y_test, dim=1)
        packed_scale = torch.linalg.vector_norm(y_test, dim=1).clamp_min(torch.finfo(dtype).eps)
        relative_packed = torch.max(packed_error / packed_scale)

    report = NeuralTrainingReport(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation),
        train_loss=float(last_train),
        validation_loss=float(last_validation),
        test_coefficient_rmse=float(coefficient_rmse.cpu()),
        test_relative_packed_error=float(relative_packed.cpu()),
        stopped_early=epochs_completed < int(cfg.epochs),
        device=str(resolved_device),
    )
    return surrogate, report


__all__ = [
    "NeuralTrainingConfig",
    "NeuralTrainingReport",
    "train_tensor_surrogate",
]
