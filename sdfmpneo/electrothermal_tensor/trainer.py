"""Standard mini-batch training for the neural Joule-tensor surrogate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
from contextlib import nullcontext

import numpy as np

from .network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .physical_layer import decode_heat_source_torch
from .surrogate import NeuralTensorSurrogate


@dataclass(frozen=True)
class NeuralTrainingConfig:
    epochs: int = 500
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    heat_loss_weight: float = 0.25
    patience: int = 50
    seed: int = 0
    dtype: str = "float32"
    gradient_clip_norm: float | None = 10.0
    evaluation_batch_size: int | None = None
    mixed_precision: bool = True
    validation_interval: int = 5
    preload_to_device: bool = True
    enable_tf32: bool = True

    def __post_init__(self) -> None:
        if int(self.epochs) < 1 or int(self.batch_size) < 1 or int(self.patience) < 1:
            raise ValueError("epochs, batch_size and patience must be positive")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("invalid optimizer settings")
        if float(self.heat_loss_weight) < 0:
            raise ValueError("heat_loss_weight must be non-negative")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) <= 0.0:
            raise ValueError("gradient_clip_norm must be positive when supplied")
        if self.evaluation_batch_size is not None and int(self.evaluation_batch_size) < 1:
            raise ValueError("evaluation_batch_size must be positive when supplied")
        if int(self.validation_interval) < 1:
            raise ValueError("validation_interval must be positive")

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
    mixed_precision: bool
    optimizer: str
    learning_rate_schedule: str
    training_config: dict
    network_config: dict


def _coefficient_normalization(beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(beta, axis=0)
    scale = np.std(beta, axis=0)
    scale = np.maximum(scale, 1e-12)
    return mean, scale


def _dtype(torch, name: str):
    return torch.float32 if name == "float32" else torch.float64


def _encode_dataset_outputs(dataset, pod, *, chunk_rows: int) -> np.ndarray:
    """Project wide frozen outputs to small POD coefficients without full RAM copy."""
    n = int(dataset.n_samples)
    beta = np.empty((n, pod.rank), dtype=np.float64)
    chunk = max(1, int(chunk_rows))
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        block = np.asarray(dataset.outputs[start:stop], dtype=np.float64)
        beta[start:stop] = pod.encode(block)
    return beta


def _resolve_device(torch, requested: str | None) -> str:
    if requested is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    value = str(requested)
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动退回 CPU。", flush=True)
        return "cpu"
    return value


def _adamw(torch, parameters, *, device: str, learning_rate: float, weight_decay: float):
    kwargs = {"lr": float(learning_rate), "weight_decay": float(weight_decay)}
    if device.startswith("cuda"):
        try:
            return torch.optim.AdamW(parameters, fused=True, **kwargs), "AdamW(fused)"
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(parameters, **kwargs), "AdamW"


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
    """Train only ``(a,g)->beta`` with a conventional mini-batch optimizer.

    EM physics is absent from this loop.  The wide frozen tensor data are read
    once to obtain POD coefficients.  By default the compact input/coefficient
    tensors then remain resident on the training device, avoiding a CPU->GPU
    copy for every mini-batch.
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
    inputs_all = np.asarray(dataset.inputs, dtype=float)
    projection_chunk = max(64, int(cfg.evaluation_batch_size or cfg.batch_size))
    print("训练准备：投影 Joule tensor 到 POD 系数……0%", flush=True)
    beta_all = _encode_dataset_outputs(dataset, pod, chunk_rows=projection_chunk)
    print("训练准备：投影 Joule tensor 到 POD 系数……100%", flush=True)
    beta_mean, beta_scale = _coefficient_normalization(beta_all[train_ids])
    input_normalizer = FeatureNormalizer.fit(inputs_all[train_ids])

    if network_config is None:
        network_config = ResidualMLPConfig(
            input_dimension=dataset.thermal_rank + dataset.geometry_dimension,
            output_dimension=pod.rank,
        )
    if network_config.input_dimension != inputs_all.shape[1] or network_config.output_dimension != pod.rank:
        raise ValueError("network dimensions do not match dataset/POD")

    rng = np.random.default_rng(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg.seed))

    resolved_device = _resolve_device(torch, device)
    dtype = _dtype(torch, cfg.dtype)
    if resolved_device.startswith("cuda") and dtype == torch.float32 and cfg.enable_tf32:
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except (AttributeError, RuntimeError):
            pass

    model = build_residual_mlp(network_config, input_normalizer)
    model = model.to(device=resolved_device, dtype=dtype)
    optimizer, optimizer_name = _adamw(
        torch,
        model.parameters(),
        device=resolved_device,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    use_amp = bool(
        cfg.mixed_precision
        and resolved_device.startswith("cuda")
        and dtype == torch.float32
    )
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    mean_t = torch.as_tensor(pod.mean, dtype=dtype, device=resolved_device)
    basis_t = torch.as_tensor(pod.basis, dtype=dtype, device=resolved_device)
    beta_mean_t = torch.as_tensor(beta_mean, dtype=dtype, device=resolved_device)
    beta_scale_t = torch.as_tensor(beta_scale, dtype=dtype, device=resolved_device)
    lo_t = torch.as_tensor(lo, dtype=dtype, device=resolved_device)
    hi_t = torch.as_tensor(hi, dtype=dtype, device=resolved_device)

    resident = bool(cfg.preload_to_device)
    if resident:
        inputs_t = torch.as_tensor(inputs_all, dtype=dtype, device=resolved_device)
        beta_t = torch.as_tensor(beta_all, dtype=dtype, device=resolved_device)
    else:
        inputs_t = beta_t = None

    def tensors(ids):
        index = np.asarray(ids, dtype=np.int64)
        if resident:
            index_t = torch.as_tensor(index, dtype=torch.long, device=resolved_device)
            return inputs_t.index_select(0, index_t), beta_t.index_select(0, index_t)
        return (
            torch.as_tensor(inputs_all[index], dtype=dtype, device=resolved_device),
            torch.as_tensor(beta_all[index], dtype=dtype, device=resolved_device),
        )

    def normalized_target(beta):
        return (beta - beta_mean_t) / beta_scale_t

    def autocast_context():
        if not use_amp:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    def batch_loss(xb, beta_b, *, random_operating: bool):
        with autocast_context():
            predicted_normalized = model(xb)
            target_normalized = normalized_target(beta_b)
            coefficient_loss = torch.mean((predicted_normalized - target_normalized) ** 2)
            if cfg.heat_loss_weight == 0.0:
                return coefficient_loss
            predicted_beta = beta_mean_t + beta_scale_t * predicted_normalized
            if random_operating:
                u = lo_t + (hi_t - lo_t) * torch.rand(
                    (xb.shape[0], dataset.current_dimension),
                    dtype=dtype,
                    device=resolved_device,
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
            target_q = decode_heat_source_torch(
                beta_b,
                mean_t,
                basis_t,
                dataset.thermal_rank,
                u,
            )
            heat_scale = torch.sqrt(torch.mean(target_q**2)).clamp_min(torch.finfo(dtype).eps)
            heat_loss = torch.mean(((predicted_q - target_q) / heat_scale) ** 2)
            return coefficient_loss + float(cfg.heat_loss_weight) * heat_loss

    evaluation_batch = int(cfg.evaluation_batch_size or max(cfg.batch_size, 1024))

    def split_loss(ids) -> float:
        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for start in range(0, len(ids), evaluation_batch):
                batch_ids = ids[start:start + evaluation_batch]
                xb, beta_b = tensors(batch_ids)
                loss = batch_loss(xb, beta_b, random_operating=False)
                total += float(loss.detach().cpu()) * len(batch_ids)
                count += len(batch_ids)
        return total / max(count, 1)

    best_state = copy.deepcopy(model.state_dict())
    best_validation = float("inf")
    best_epoch = 0
    last_train = float("inf")
    last_validation = float("inf")
    stale_epochs = 0
    epochs_completed = 0
    validation_interval = max(1, int(cfg.validation_interval))

    print(
        f"开始 MLP 训练：device={resolved_device}, optimizer={optimizer_name}, "
        f"AMP={'on' if use_amp else 'off'}, batch={cfg.batch_size}",
        flush=True,
    )

    for epoch in range(int(cfg.epochs)):
        model.train()
        shuffled = rng.permutation(train_ids)
        total = 0.0
        count = 0
        for start in range(0, len(shuffled), int(cfg.batch_size)):
            ids = shuffled[start:start + int(cfg.batch_size)]
            xb, beta_b = tensors(ids)
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(xb, beta_b, random_operating=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if cfg.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.gradient_clip_norm)
                    )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg.gradient_clip_norm)
                    )
                optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
            count += len(ids)
        last_train = total / max(count, 1)
        epochs_completed = epoch + 1

        validate_now = (
            epoch == 0
            or epochs_completed % validation_interval == 0
            or epochs_completed == int(cfg.epochs)
        )
        if validate_now:
            last_validation = split_loss(val_ids)
            threshold = (
                best_validation - 1e-10 * max(1.0, abs(best_validation))
                if np.isfinite(best_validation)
                else float("inf")
            )
            if not np.isfinite(best_validation) or last_validation < threshold:
                best_validation = last_validation
                best_epoch = epochs_completed
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += validation_interval

        percent = 100.0 * epochs_completed / int(cfg.epochs)
        val_text = "--" if not np.isfinite(last_validation) else f"{last_validation:.5g}"
        print(
            f"训练神经网络……{percent:5.1f}%  epoch={epochs_completed}/{cfg.epochs}  "
            f"train={last_train:.5g}  val={val_text}",
            flush=True,
        )
        if stale_epochs >= int(cfg.patience):
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

    coefficient_sq = 0.0
    coefficient_count = 0
    maximum_relative_packed = 0.0
    with torch.no_grad():
        for start in range(0, len(test_ids), evaluation_batch):
            ids = test_ids[start:start + evaluation_batch]
            xb, _ = tensors(ids)
            with autocast_context():
                predicted_normalized = model(xb)
                predicted_beta = beta_mean_t + beta_scale_t * predicted_normalized
            predicted = predicted_beta.detach().cpu().double().numpy()
            true_beta = beta_all[ids]
            beta_error = predicted - true_beta
            coefficient_sq += float(np.sum(beta_error**2))
            coefficient_count += int(beta_error.size)

            true_outputs = np.asarray(dataset.outputs[ids], dtype=np.float64)
            centered = true_outputs - pod.mean
            centered_sq = np.einsum("ij,ij->i", centered, centered, optimize=True)
            projected_sq = np.einsum("ij,ij->i", true_beta, true_beta, optimize=True)
            orthogonal_sq = np.maximum(centered_sq - projected_sq, 0.0)
            prediction_sq = orthogonal_sq + np.einsum(
                "ij,ij->i", beta_error, beta_error, optimize=True
            )
            packed_scale = np.maximum(
                np.linalg.norm(true_outputs, axis=1),
                np.finfo(float).tiny,
            )
            maximum_relative_packed = max(
                maximum_relative_packed,
                float(np.max(np.sqrt(prediction_sq) / packed_scale)),
            )

    coefficient_rmse = float(np.sqrt(coefficient_sq / max(1, coefficient_count)))
    if not np.isfinite(best_validation):
        best_validation = split_loss(val_ids)
        best_epoch = epochs_completed
    report = NeuralTrainingReport(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_loss=float(best_validation),
        train_loss=float(last_train),
        validation_loss=float(last_validation),
        test_coefficient_rmse=coefficient_rmse,
        test_relative_packed_error=float(maximum_relative_packed),
        stopped_early=epochs_completed < int(cfg.epochs),
        device=resolved_device,
        mixed_precision=use_amp,
        optimizer=optimizer_name,
        learning_rate_schedule="constant",
        training_config=cfg.to_dict(),
        network_config=network_config.to_dict(),
    )
    return surrogate, report


__all__ = [
    "NeuralTrainingConfig",
    "NeuralTrainingReport",
    "train_tensor_surrogate",
]
