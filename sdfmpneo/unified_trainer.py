"""Solution-label-free training for the full-edge neural Maxwell preconditioner."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy

import numpy as np

from .unified_neural_operator import (
    EdgeMultiscaleConfig,
    build_edge_residual_operator,
    residual_features_from_diagonal,
    safe_diagonal_values,
)


@dataclass(frozen=True)
class MaxwellTrainingConfig:
    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 2e-3
    weight_decay: float = 1e-6
    patience: int = 60
    validation_interval: int = 2
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
    residual_vectors_per_operator: int


def _operator_coefficients(background, context, state):
    """Compact exact coefficients for A=C^T diag(h2) C + diag(d)."""
    sigma, eps, mu_inv, _, _, _ = background.cell_properties(context, state, em=True)
    h2 = np.asarray(background.face_cell_hodge @ mu_inv, float).ravel()
    hs = np.asarray(background.edge_cell_hodge @ sigma, float).ravel()
    he = np.asarray(background.edge_cell_hodge @ eps, float).ravel()
    d = -background.omega**2 * he + 1j * background.omega * hs
    curl_diag = np.asarray(background.curl.multiply(background.curl).T @ h2, float).ravel()
    diagonal = safe_diagonal_values(curl_diag + d)
    return h2, np.asarray(d, complex), diagonal


def _apply_numpy(curl, h2, diagonal_term, X):
    value = np.asarray(X, complex)
    vector = value.ndim == 1
    if vector:
        value = value[:, None]
    curl_value = curl @ value
    result = curl.T @ (h2[:, None] * curl_value) + diagonal_term[:, None] * value
    return np.asarray(result[:, 0] if vector else result, complex)


def _baseline_residual_bank(curl, h2, diagonal_term, diagonal, B, steps, rng):
    """Build B-like and intermediate physical residuals without solution labels."""
    B = np.asarray(B, complex)
    columns = []
    for p in range(B.shape[1]):
        r = B[:, p].copy()
        for _ in range(int(steps)):
            columns.append(r.copy())
            z = r / diagonal
            r = r - _apply_numpy(curl, h2, diagonal_term, z)
    if B.shape[1] > 1:
        for _ in range(max(1, int(steps) // 2)):
            weights = rng.normal(size=B.shape[1]) + 1j * rng.normal(size=B.shape[1])
            norm = np.linalg.norm(weights)
            if norm > 0:
                columns.append(np.asarray(B @ (weights / norm), complex))
    return np.column_stack(columns)


def _torch_sparse_complex(torch, matrix, device):
    coo = matrix.tocoo()
    indices = torch.as_tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.as_tensor(coo.data, dtype=torch.complex128, device=device)
    return torch.sparse_coo_tensor(indices, values, size=matrix.shape, device=device).coalesce()


def _torch_apply(torch, curl_t, h2, diagonal_term, Z):
    curl_value = torch.sparse.mm(curl_t, Z)
    weighted = h2[:, None] * curl_value
    return torch.sparse.mm(curl_t.transpose(0, 1), weighted) + diagonal_term[:, None] * Z


def _system_loss(torch, model, curl_t, h2, diagonal_term, diagonal, R, *, device, network_dtype):
    features, jacobi, scale = residual_features_from_diagonal(diagonal, R)
    x = torch.as_tensor(features, dtype=network_dtype, device=device)
    y = model(x).to(dtype=torch.float64)
    learned = torch.complex(y[..., 0], y[..., 1]).transpose(0, 1)
    baseline = torch.as_tensor(jacobi, dtype=torch.complex128, device=device)
    scale_t = torch.as_tensor(scale, dtype=torch.float64, device=device)
    Z = baseline + learned.to(torch.complex128) * scale_t[None, :]
    h2_t = torch.as_tensor(h2, dtype=torch.complex128, device=device)
    d_t = torch.as_tensor(diagonal_term, dtype=torch.complex128, device=device)
    R_t = torch.as_tensor(R, dtype=torch.complex128, device=device)
    after = R_t - _torch_apply(torch, curl_t, h2_t, d_t, Z)
    numerator = torch.sum(torch.abs(after) ** 2, dim=0)
    denominator = torch.sum(torch.abs(R_t) ** 2, dim=0).clamp_min(torch.finfo(torch.float64).tiny)
    return torch.mean(numerator / denominator)


def train_maxwell_accelerator(
    background,
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
    net_cfg = EdgeMultiscaleConfig(**dict(network_settings or {}))
    train_ids = dataset.indices("train")
    validation_ids = dataset.indices("validation")
    test_ids = dataset.indices("test")
    if min(len(train_ids), len(validation_ids), len(test_ids)) < 1:
        raise ValueError("operator dataset must contain train/validation/test samples")

    actual_device = str(device)
    if actual_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，自动退回 CPU。", flush=True)
        actual_device = "cpu"
    network_dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    model = build_edge_residual_operator(background, net_cfg).to(device=actual_device, dtype=network_dtype)

    try:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
            fused=actual_device.startswith("cuda"),
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
        )

    systems = []
    total = len(dataset.geometries)
    for i, (geometry, state) in enumerate(zip(dataset.geometries, dataset.states)):
        if monitor is not None:
            monitor.checkpoint()
        context = background.geometry_context(geometry, assemble_thermal=False)
        h2, diagonal_term, diagonal = _operator_coefficients(background, context, state)
        B = np.asarray(background.rhs_matrix(context), complex)
        systems.append((h2, diagonal_term, diagonal, B))
        if i == 0 or (i + 1) % max(1, total // 20) == 0 or i + 1 == total:
            print(
                f"缓存 Maxwell matrix-free 系数……{100 * (i + 1) / total:5.1f}% ({i + 1}/{total})",
                flush=True,
            )

    curl_t = _torch_sparse_complex(torch, background.curl, actual_device)

    def residual_bank(index):
        h2, diagonal_term, diagonal, B = systems[int(index)]
        rng = np.random.default_rng(int(dataset.seed) + 104729 * (int(index) + 1))
        R = _baseline_residual_bank(
            background.curl, h2, diagonal_term, diagonal, B, dataset.residual_steps, rng
        )
        return h2, diagonal_term, diagonal, R

    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    start = 0
    best = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    checkpoint_identity = {
        "network_config": net_cfg.to_dict(),
        "training_config": cfg.to_dict(),
        "n_edges": int(background.n_edges),
        "sample_count": int(total),
        "residual_steps": int(dataset.residual_steps),
        "operator_representation": "matrix_free_curl_hodge",
    }
    if checkpoint is not None and checkpoint.is_file():
        try:
            try:
                saved = torch.load(checkpoint, map_location=actual_device, weights_only=False)
            except TypeError:
                saved = torch.load(checkpoint, map_location=actual_device)
            if saved.get("identity") != checkpoint_identity:
                raise ValueError("training configuration or edge topology changed")
            model.load_state_dict(saved["network"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            start = int(saved["epoch"])
            best = float(saved["best"])
            best_epoch = int(saved["best_epoch"])
            best_state = saved["best_state"]
            print(f"恢复 Maxwell residual-correction 训练：epoch={start}", flush=True)
        except Exception as exc:
            print(f"训练检查点与当前统一模型不一致，重新训练网络：{exc}", flush=True)
            checkpoint.unlink(missing_ok=True)

    def system_loss(index):
        h2, diagonal_term, diagonal, R = residual_bank(index)
        return _system_loss(
            torch, model, curl_t, h2, diagonal_term, diagonal, R,
            device=actual_device, network_dtype=network_dtype,
        )

    def evaluate(ids):
        model.eval()
        values = []
        with torch.no_grad():
            for idx in ids:
                values.append(float(system_loss(int(idx)).detach().cpu()))
        return float(np.mean(values)) if values else float("inf")

    stale = 0
    last_train = float("inf")
    last_validation = float("inf")
    completed = start
    epoch_rng = np.random.default_rng(cfg.seed)

    for epoch in range(start, cfg.epochs):
        if monitor is not None:
            monitor.checkpoint()
        model.train()
        order = epoch_rng.permutation(train_ids)
        losses = []
        optimizer.zero_grad(set_to_none=True)
        pending_losses = []
        for position, idx in enumerate(order):
            if monitor is not None:
                monitor.checkpoint()
            loss = system_loss(int(idx))
            pending_losses.append(loss)
            losses.append(float(loss.detach().cpu()))
            flush = len(pending_losses) >= cfg.batch_size or position + 1 == len(order)
            if flush:
                torch.stack(pending_losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_losses.clear()

        last_train = float(np.mean(losses)) if losses else float("inf")
        completed = epoch + 1
        validate = completed == 1 or completed % cfg.validation_interval == 0 or completed == cfg.epochs
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
            f"训练 Maxwell neural residual corrector……{100 * completed / cfg.epochs:5.1f}%  "
            f"epoch={completed}/{cfg.epochs}  train={last_train:.5g}  val={last_validation:.5g}",
            flush=True,
        )
        if monitor is not None:
            with monitor._lock:
                monitor.data.update(
                    phase="neural_training", epoch=completed, epoch_total=cfg.epochs,
                    train_loss=last_train,
                    validation_loss=None if not np.isfinite(last_validation) else last_validation,
                )
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
            torch.save({
                "identity": checkpoint_identity,
                "epoch": completed,
                "network": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best": best,
                "best_epoch": best_epoch,
                "best_state": best_state,
            }, temporary)
            temporary.replace(checkpoint)
        if stale >= cfg.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    test_loss = evaluate(test_ids)
    _, _, _, first_R = residual_bank(0)
    report = MaxwellTrainingReport(
        completed, best_epoch, float(best), float(last_train), float(last_validation),
        float(test_loss), completed < cfg.epochs, actual_device, cfg.to_dict(),
        net_cfg.to_dict(), int(first_R.shape[1]),
    )
    return model, report


__all__ = ["MaxwellTrainingConfig", "MaxwellTrainingReport", "train_maxwell_accelerator"]
