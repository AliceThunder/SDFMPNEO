"""Physical-residual training for the unified neural Maxwell initial guess."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy

import numpy as np

from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp


@dataclass(frozen=True)
class MaxwellTrainingConfig:
    epochs: int = 1000
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    patience: int = 150
    validation_interval: int = 5
    seed: int = 17
    dtype: str = "float32"

    def __post_init__(self):
        if min(self.epochs, self.batch_size, self.patience, self.validation_interval) < 1:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MaxwellTrainingReport:
    epochs_completed: int
    best_epoch: int
    best_validation_residual_loss: float
    train_residual_loss: float
    validation_residual_loss: float
    test_residual_loss: float
    stopped_early: bool
    device: str
    training_config: dict
    network_config: dict


def residual_loss(torch, y, base, scale, Q, S, N, p, r):
    """Mean squared full-background relative Maxwell residual.

    Network arithmetic may be float32, but the residual quadratic contraction is
    intentionally evaluated in float64.  Near convergence the expression is a
    subtraction of O(1) terms; keeping it in float32 can create false zero loss.
    """
    coefficients = base + scale.unsqueeze(-1) * y.reshape((-1, p, 2 * r))
    work = coefficients.to(dtype=Q.dtype)
    quad = torch.einsum("bpi,bij,bpj->bp", work, Q, work)
    linear = torch.einsum("bpi,bpi->bp", work, S)
    residual2 = (N - 2.0 * linear + quad).clamp_min(0.0)
    return torch.mean(residual2 / N.clamp_min(torch.finfo(Q.dtype).tiny))


def train_maxwell_accelerator(
    dataset,
    *,
    network_settings=None,
    training_settings=None,
    device="cuda",
    monitor=None,
    checkpoint_path=None,
):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("install sdfmpneo[neural] to train the unified model") from exc

    cfg = MaxwellTrainingConfig(**(training_settings or {}))
    rank, n_rhs = dataset.reduced_rank, dataset.n_rhs
    train_ids = dataset.indices("train")
    validation_ids = dataset.indices("validation")
    test_ids = dataset.indices("test")
    if min(len(train_ids), len(validation_ids), len(test_ids)) < 1:
        raise ValueError("operator dataset must contain train/validation/test samples")

    normalizer = FeatureNormalizer.fit(dataset.features[train_ids])
    network_config = ResidualMLPConfig(
        input_dimension=dataset.features.shape[1],
        output_dimension=2 * rank * n_rhs,
        **dict(network_settings or {}),
    )
    model = build_residual_mlp(network_config, normalizer)
    actual_device = str(device)
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动退回 CPU。", flush=True)
        actual_device = "cpu"
    network_dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    model = model.to(device=actual_device, dtype=network_dtype)

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            fused=actual_device.startswith("cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

    features = torch.as_tensor(dataset.features, dtype=network_dtype, device=actual_device)
    baseline = torch.as_tensor(dataset.baseline, dtype=network_dtype, device=actual_device)
    coefficient_scale = torch.as_tensor(dataset.coefficient_scale, dtype=network_dtype, device=actual_device)
    # The physical residual quadratic form stays float64 even when the NN is float32.
    gram = torch.as_tensor(dataset.residual_gram, dtype=torch.float64, device=actual_device)
    linear = torch.as_tensor(dataset.residual_linear, dtype=torch.float64, device=actual_device)
    rhs_norm2 = torch.as_tensor(dataset.rhs_norm2, dtype=torch.float64, device=actual_device)

    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    start = 0
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    checkpoint_identity = {
        "network_config": network_config.to_dict(),
        "training_config": cfg.to_dict(),
        "feature_dimension": int(dataset.features.shape[1]),
        "reduced_rank": int(rank),
        "n_rhs": int(n_rhs),
    }

    if checkpoint is not None and checkpoint.is_file():
        try:
            try:
                saved = torch.load(checkpoint, map_location=actual_device, weights_only=False)
            except TypeError:
                saved = torch.load(checkpoint, map_location=actual_device)
            if saved.get("identity") != checkpoint_identity:
                raise ValueError("training configuration or physical feature dimensions changed")
            model.load_state_dict(saved["network"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            start = int(saved["epoch"])
            best = float(saved["best"])
            best_epoch = int(saved["best_epoch"])
            best_state = saved["best_state"]
            print(f"恢复 Maxwell residual 训练：epoch={start}", flush=True)
        except Exception as exc:
            print(f"训练检查点与当前统一模型不一致，重新训练网络：{exc}", flush=True)
            checkpoint.unlink(missing_ok=True)

    rng = np.random.default_rng(cfg.seed)

    def batch_loss(index):
        return residual_loss(
            torch,
            model(features[index]),
            baseline[index],
            coefficient_scale[index],
            gram[index],
            linear[index],
            rhs_norm2[index],
            n_rhs,
            rank,
        )

    def evaluate(ids):
        model.eval()
        total = 0.0
        count = 0
        batch = max(cfg.batch_size, 256)
        with torch.no_grad():
            for start_index in range(0, len(ids), batch):
                index = torch.as_tensor(
                    ids[start_index:start_index + batch],
                    dtype=torch.long,
                    device=actual_device,
                )
                loss = batch_loss(index)
                total += float(loss.cpu()) * len(index)
                count += len(index)
        return total / max(count, 1)

    last_train = float("inf")
    last_validation = float("inf")
    stale = 0
    completed = start
    for epoch in range(start, cfg.epochs):
        if monitor is not None:
            monitor.checkpoint()
        model.train()
        order = rng.permutation(train_ids)
        total = 0.0
        count = 0
        for start_index in range(0, len(order), cfg.batch_size):
            if monitor is not None:
                monitor.checkpoint()
            index = torch.as_tensor(
                order[start_index:start_index + cfg.batch_size],
                dtype=torch.long,
                device=actual_device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * len(index)
            count += len(index)
        last_train = total / max(count, 1)
        completed = epoch + 1

        validate = (
            completed == 1
            or completed % cfg.validation_interval == 0
            or completed == cfg.epochs
        )
        if validate:
            last_validation = evaluate(validation_ids)
            threshold = best - 1e-10 * max(1.0, abs(best)) if np.isfinite(best) else float("inf")
            if not np.isfinite(best) or last_validation < threshold:
                best = last_validation
                best_epoch = completed
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += cfg.validation_interval

        print(
            f"训练神经网络……{100 * completed / cfg.epochs:5.1f}%  "
            f"epoch={completed}/{cfg.epochs}  train={last_train:.5g}  val={last_validation:.5g}",
            flush=True,
        )
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="neural_training",
                    epoch=completed,
                    epoch_total=cfg.epochs,
                    train_loss=last_train,
                    validation_loss=None if not np.isfinite(last_validation) else last_validation,
                )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save(
                {
                    "identity": checkpoint_identity,
                    "epoch": completed,
                    "network": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best": best,
                    "best_epoch": best_epoch,
                    "best_state": best_state,
                },
                temporary,
            )
            temporary.replace(checkpoint)
        if stale >= cfg.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    test_loss = evaluate(test_ids)
    report = MaxwellTrainingReport(
        completed,
        best_epoch,
        float(best),
        float(last_train),
        float(last_validation),
        float(test_loss),
        completed < cfg.epochs,
        actual_device,
        cfg.to_dict(),
        network_config.to_dict(),
    )
    return model, report


__all__ = [
    "MaxwellTrainingConfig",
    "MaxwellTrainingReport",
    "residual_loss",
    "train_maxwell_accelerator",
]
