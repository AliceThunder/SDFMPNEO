"""Matrix-aware POD training for the geometry-only EM tensor surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib

import numpy as np

from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .unified_tensor_surrogate import UnifiedTensorSurrogate, pack_tensors, tensor_block_sizes


@dataclass(frozen=True)
class TensorTrainingReport:
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    test_loss: float
    pod_rank: int
    pod_relative_tail_error: float
    test_relative_tensor_error: float
    test_z_relative_error: float
    test_d_relative_error: float
    test_h_relative_error: float
    audit_relative_tensor_error: float
    audit_z_relative_error: float
    audit_d_relative_error: float
    audit_h_relative_error: float
    maximum_test_zd_projection_correction: float
    maximum_test_h_projection_correction: float
    maximum_audit_zd_projection_correction: float
    maximum_audit_h_projection_correction: float
    device: str
    network_config: dict
    training_config: dict


def _resolve_device(torch, requested):
    value = "cuda" if requested is None else str(requested)
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA 不可用，tensor surrogate 自动退回 CPU。", flush=True)
        return "cpu"
    return value


def _matrix_weights(n_ports):
    n = int(n_ports)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    pair_w = np.asarray([1.0 if i == j else 2.0 for i, j in pairs], float)
    z = np.concatenate([pair_w, pair_w])
    off = n * (n - 1) // 2
    h = np.concatenate([np.ones(n), 2.0 * np.ones(off), 2.0 * np.ones(off)])
    return z, h


def _fit_output_pod(outputs, *, relative_tail_tolerance):
    y = np.asarray(outputs, float)
    if y.ndim != 2 or y.shape[0] < 1:
        raise ValueError("tensor POD requires a nonempty output matrix")
    tol = float(relative_tail_tolerance)
    if not 0.0 < tol < 1.0:
        raise ValueError("pod_relative_tail_tolerance must lie in (0, 1)")
    mean = np.mean(y, axis=0)
    scale = np.maximum(np.std(y, axis=0), 1e-12)
    normalized = (y - mean) / scale
    _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
    total = float(np.sum(singular**2))
    if total <= np.finfo(float).tiny:
        basis = np.zeros((y.shape[1], 1), float)
        basis[0, 0] = 1.0
        return mean, scale, basis, 0.0
    cumulative = np.cumsum(singular**2)
    tails = np.sqrt(np.maximum(total - cumulative, 0.0) / total)
    valid = np.flatnonzero(tails <= tol)
    rank = int(valid[0] + 1) if valid.size else len(singular)
    basis = vt[:rank].T.copy()
    tail = float(tails[rank - 1])
    return mean, scale, basis, tail


def _unpack_z_torch(torch, packed, n):
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    m = len(pairs)
    dtype = torch.complex64 if packed.dtype == torch.float32 else torch.complex128
    z = torch.zeros((*packed.shape[:-1], n, n), dtype=dtype, device=packed.device)
    for k, (i, j) in enumerate(pairs):
        value = torch.complex(packed[..., k], packed[..., m + k])
        z[..., i, j] = value
        z[..., j, i] = value
    return z


def _unpack_h_torch(torch, packed, n):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    dtype = torch.complex64 if packed.dtype == torch.float32 else torch.complex128
    h = torch.zeros((*packed.shape[:-1], n, n), dtype=dtype, device=packed.device)
    for i in range(n):
        h[..., i, i] = torch.complex(packed[..., i], torch.zeros_like(packed[..., i]))
    start = n
    for k, (i, j) in enumerate(pairs):
        value = torch.complex(
            packed[..., start + k], packed[..., start + len(pairs) + k]
        )
        h[..., i, j] = value
        h[..., j, i] = value.conj()
    return h


def _physics_penalty(torch, physical, n_ports, thermal_rank, phi_min, phi_max):
    n = int(n_ports)
    r = int(thermal_rank)
    z_size, h_size, _ = tensor_block_sizes(n, r)
    z = _unpack_z_torch(torch, physical[..., :z_size], n)
    d = _unpack_h_torch(torch, physical[..., z_size:z_size + h_size], n)
    penalty = torch.mean(torch.relu(-torch.linalg.eigvalsh(d).real) ** 2)
    outward = z.real.to(d.dtype) - d
    penalty = penalty + torch.mean(
        torch.relu(-torch.linalg.eigvalsh(outward).real) ** 2
    )
    start = z_size + h_size
    for j in range(r):
        h = _unpack_h_torch(
            torch,
            physical[..., start + j * h_size:start + (j + 1) * h_size],
            n,
        )
        low = h - float(phi_min[j]) * d
        high = float(phi_max[j]) * d - h
        penalty = penalty + torch.mean(torch.relu(-torch.linalg.eigvalsh(low).real) ** 2)
        penalty = penalty + torch.mean(torch.relu(-torch.linalg.eigvalsh(high).real) ** 2)
    return penalty / max(1, 2 + 2 * r)


def _relative_numpy(diff, truth, weights):
    num = float(np.sum(np.asarray(diff, float) ** 2 * weights))
    den = max(float(np.sum(np.asarray(truth, float) ** 2 * weights)), np.finfo(float).tiny)
    return float(np.sqrt(num / den))


def train_matrix_tensor_surrogate(
    dataset,
    phi_min,
    phi_max,
    *,
    network_settings=None,
    training_settings=None,
    device="cuda",
    monitor=None,
    checkpoint_path=None,
):
    """Fit training-only POD and optimize per-matrix relative errors."""
    import torch

    cfg = {
        "epochs": 240,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "patience": 40,
        "validation_interval": 2,
        "gradient_clip_norm": 10.0,
        "physics_penalty_weight": 0.05,
        "z_weight": 1.0,
        "d_weight": 1.0,
        "h_weight": 1.0,
        "pod_relative_tail_tolerance": 1e-4,
        "seed": 17,
        "dtype": "float32",
    }
    cfg.update(dict(training_settings or {}))
    train_ids = dataset.indices("train")
    val_ids = dataset.indices("validation")
    test_ids = dataset.indices("test")
    audit_ids = dataset.indices("audit")
    if min(len(train_ids), len(val_ids), len(test_ids), len(audit_ids)) < 1:
        raise ValueError("tensor dataset requires nonempty train/validation/test/audit splits")

    input_norm = FeatureNormalizer.fit(dataset.inputs[train_ids])
    output_mean, output_scale, pod_basis, pod_tail = _fit_output_pod(
        dataset.outputs[train_ids],
        relative_tail_tolerance=float(cfg["pod_relative_tail_tolerance"]),
    )
    pod_rank = int(pod_basis.shape[1])
    net_settings = dict(network_settings or {})
    net_settings.pop("input_dimension", None)
    net_settings.pop("output_dimension", None)
    net_cfg = ResidualMLPConfig(
        input_dimension=dataset.inputs.shape[1],
        output_dimension=pod_rank,
        **net_settings,
    )
    network = build_residual_mlp(net_cfg, input_norm)

    resolved = _resolve_device(torch, device)
    dtype = torch.float32 if str(cfg["dtype"]) == "float32" else torch.float64
    network = network.to(device=resolved, dtype=dtype)
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    x = torch.as_tensor(dataset.inputs, dtype=dtype, device=resolved)
    target = torch.as_tensor(dataset.outputs, dtype=dtype, device=resolved)
    out_mean = torch.as_tensor(output_mean, dtype=dtype, device=resolved)
    out_scale = torch.as_tensor(output_scale, dtype=dtype, device=resolved)
    pod_t = torch.as_tensor(pod_basis, dtype=dtype, device=resolved)
    z_w_np, h_w_np = _matrix_weights(dataset.n_ports)
    z_w = torch.as_tensor(z_w_np, dtype=dtype, device=resolved)
    h_w = torch.as_tensor(h_w_np, dtype=dtype, device=resolved)
    phi_min = np.asarray(phi_min, float).reshape(-1)
    phi_max = np.asarray(phi_max, float).reshape(-1)

    n = int(dataset.n_ports)
    r = int(dataset.thermal_rank)
    z_size, h_size, _ = tensor_block_sizes(n, r)

    def relative_block(diff, truth, weights):
        numerator = torch.sum(diff * diff * weights, dim=-1)
        denominator = torch.sum(truth * truth * weights, dim=-1).clamp_min(
            torch.finfo(dtype).eps
        )
        return numerator / denominator

    def physical_prediction(xb):
        beta = network(xb)
        normalized = beta @ pod_t.T
        return out_mean + out_scale * normalized

    def batch_loss(ids):
        ids_t = torch.as_tensor(
            np.asarray(ids, np.int64), dtype=torch.long, device=resolved
        )
        xb = x.index_select(0, ids_t)
        truth = target.index_select(0, ids_t)
        pred = physical_prediction(xb)
        diff = pred - truth
        loss_z = torch.mean(
            relative_block(diff[..., :z_size], truth[..., :z_size], z_w)
        )
        loss_d = torch.mean(
            relative_block(
                diff[..., z_size:z_size + h_size],
                truth[..., z_size:z_size + h_size],
                h_w,
            )
        )
        start = z_size + h_size
        h_losses = []
        for j in range(r):
            sl = slice(start + j * h_size, start + (j + 1) * h_size)
            h_losses.append(
                torch.mean(relative_block(diff[..., sl], truth[..., sl], h_w))
            )
        loss_h = torch.stack(h_losses).mean()
        loss = (
            float(cfg["z_weight"]) * loss_z
            + float(cfg["d_weight"]) * loss_d
            + float(cfg["h_weight"]) * loss_h
        )
        if float(cfg["physics_penalty_weight"]) > 0.0:
            physical_scale = torch.mean(truth * truth).clamp_min(torch.finfo(dtype).eps)
            penalty = _physics_penalty(torch, pred, n, r, phi_min, phi_max) / physical_scale
            loss = loss + float(cfg["physics_penalty_weight"]) * penalty
        return loss

    rng = np.random.default_rng(int(cfg["seed"]))
    torch.manual_seed(int(cfg["seed"]))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(cfg["seed"]))

    pod_signature = hashlib.sha256(
        output_mean.tobytes() + output_scale.tobytes() + pod_basis.tobytes()
    ).hexdigest()
    checkpoint = None if checkpoint_path is None else Path(checkpoint_path)
    best_state = copy.deepcopy(network.state_dict())
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 0
    if checkpoint is not None and checkpoint.is_file():
        try:
            saved = torch.load(checkpoint, map_location=resolved, weights_only=False)
            compatible = (
                int(saved.get("schema_version", -1)) == 2
                and saved.get("pod_signature") == pod_signature
                and int(saved.get("pod_rank", -1)) == pod_rank
            )
            if compatible:
                network.load_state_dict(saved["network"])
                optimizer.load_state_dict(saved["optimizer"])
                best_state = saved["best_network"]
                best_val = float(saved["best_validation_loss"])
                best_epoch = int(saved["best_epoch"])
                stale = int(saved.get("stale", 0))
                start_epoch = int(saved["epoch"])
                print(f"恢复 geometry→tensor 检查点：epoch={start_epoch}", flush=True)
        except (OSError, RuntimeError, ValueError, KeyError):
            print("geometry→tensor 检查点不兼容，重新训练。", flush=True)

    interval = max(1, int(cfg["validation_interval"]))
    epochs_completed = start_epoch
    last_val = float("inf")
    for epoch in range(start_epoch, int(cfg["epochs"])):
        if monitor is not None:
            monitor.checkpoint()
        network.train()
        order = rng.permutation(train_ids)
        total = 0.0
        count = 0
        for start in range(0, len(order), int(cfg["batch_size"])):
            ids = order[start:start + int(cfg["batch_size"])]
            loss = batch_loss(ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.get("gradient_clip_norm") is not None:
                torch.nn.utils.clip_grad_norm_(
                    network.parameters(), float(cfg["gradient_clip_norm"])
                )
            optimizer.step()
            total += float(loss.detach().cpu()) * len(ids)
            count += len(ids)
        train_loss = total / max(count, 1)
        epochs_completed = epoch + 1
        validate = (
            epochs_completed == 1
            or epochs_completed % interval == 0
            or epochs_completed == int(cfg["epochs"])
        )
        if validate:
            network.eval()
            with torch.no_grad():
                last_val = float(batch_loss(val_ids).detach().cpu())
            if last_val < best_val - 1e-10 * max(1.0, abs(best_val)):
                best_val = last_val
                best_epoch = epochs_completed
                best_state = copy.deepcopy(network.state_dict())
                stale = 0
            else:
                stale += interval
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema_version": 2,
                        "pod_signature": pod_signature,
                        "pod_rank": pod_rank,
                        "epoch": epochs_completed,
                        "best_epoch": best_epoch,
                        "best_validation_loss": best_val,
                        "network": network.state_dict(),
                        "best_network": best_state,
                        "optimizer": optimizer.state_dict(),
                        "stale": stale,
                    },
                    checkpoint,
                )
            if monitor is not None:
                with monitor._lock:
                    monitor.data.update(
                        phase="tensor_training",
                        epoch=epochs_completed,
                        train_loss=train_loss,
                        validation_loss=last_val,
                        tensor_pod_rank=pod_rank,
                    )
            print(
                f"训练 geometry→tensor POD-MLP……epoch={epochs_completed}/{cfg['epochs']}  "
                f"train={train_loss:.5g} val={last_val:.5g} POD={pod_rank}",
                flush=True,
            )
            if stale >= int(cfg["patience"]):
                break

    network.load_state_dict(best_state)
    network.eval()
    surrogate = UnifiedTensorSurrogate(
        network,
        output_mean,
        output_scale,
        pod_basis,
        n,
        phi_min,
        phi_max,
    )
    with torch.no_grad():
        test_loss = float(batch_loss(test_ids).detach().cpu())

    def split_metrics(ids):
        overall = []
        z_errors = []
        d_errors = []
        h_errors = []
        zd_projection = []
        h_projection = []
        for idx in ids:
            decoded = surrogate.predict_from_encoded(dataset.inputs[idx])
            pred = pack_tensors(decoded.z_field, decoded.d_vol, decoded.modal_h)
            truth = dataset.outputs[idx]
            diff = pred - truth
            overall.append(
                float(np.linalg.norm(diff) / max(np.linalg.norm(truth), np.finfo(float).tiny))
            )
            z_errors.append(_relative_numpy(diff[:z_size], truth[:z_size], z_w_np))
            d_errors.append(
                _relative_numpy(
                    diff[z_size:z_size + h_size],
                    truth[z_size:z_size + h_size],
                    h_w_np,
                )
            )
            start = z_size + h_size
            per_h = []
            for j in range(r):
                sl = slice(start + j * h_size, start + (j + 1) * h_size)
                per_h.append(_relative_numpy(diff[sl], truth[sl], h_w_np))
            h_errors.append(max(per_h))
            zd_projection.append(decoded.zd_projection_correction)
            h_projection.append(decoded.h_projection_correction)
        return {
            "overall": max(overall),
            "z": max(z_errors),
            "d": max(d_errors),
            "h": max(h_errors),
            "zd_projection": max(zd_projection),
            "h_projection": max(h_projection),
        }

    test_metrics = split_metrics(test_ids)
    audit_metrics = split_metrics(audit_ids)
    report = TensorTrainingReport(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_loss=float(best_val),
        test_loss=test_loss,
        pod_rank=pod_rank,
        pod_relative_tail_error=pod_tail,
        test_relative_tensor_error=test_metrics["overall"],
        test_z_relative_error=test_metrics["z"],
        test_d_relative_error=test_metrics["d"],
        test_h_relative_error=test_metrics["h"],
        audit_relative_tensor_error=audit_metrics["overall"],
        audit_z_relative_error=audit_metrics["z"],
        audit_d_relative_error=audit_metrics["d"],
        audit_h_relative_error=audit_metrics["h"],
        maximum_test_zd_projection_correction=test_metrics["zd_projection"],
        maximum_test_h_projection_correction=test_metrics["h_projection"],
        maximum_audit_zd_projection_correction=audit_metrics["zd_projection"],
        maximum_audit_h_projection_correction=audit_metrics["h_projection"],
        device=resolved,
        network_config=net_cfg.to_dict(),
        training_config=dict(cfg),
    )
    return surrogate, report


__all__ = ["TensorTrainingReport", "train_matrix_tensor_surrogate"]
