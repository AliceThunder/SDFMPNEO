"""Matrix-aware POD training for the geometry-only EM tensor surrogate."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib

import numpy as np

from .electrothermal_tensor.network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .unified_tensor_surrogate import (
    UnifiedTensorSurrogate,
    UnifiedSpatialTensorSurrogate,
    decode_physical_tensors,
    decode_spatial_tensors,
    decode_geometry_encoding,
    encode_geometry_invariant,
    pack_tensors,
    pack_spatial_tensors,
    pack_spatial_global_tensors,
    pack_whitened_field_factors,
    tensor_block_sizes,
    unpack_spatial_tensors,
    spatial_cell_features,
    whiten_cell_joule_tensors,
)


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
    total = float(np.sum(singular ** 2))
    if total <= np.finfo(float).tiny:
        basis = np.zeros((y.shape[1], 1), float)
        basis[0, 0] = 1.0
        return mean, scale, basis, 0.0
    cumulative = np.cumsum(singular ** 2)
    tails = np.sqrt(np.maximum(total - cumulative, 0.0) / total)
    valid = np.flatnonzero(tails <= tol)
    rank = int(valid[0] + 1) if valid.size else len(singular)
    return mean, scale, vt[:rank].T.copy(), float(tails[rank - 1])


def _fit_output_pod_dual(outputs, *, relative_tail_tolerance):
    """Training-only POD via the sample Gram matrix for very wide outputs."""
    y = np.asarray(outputs, float)
    if y.ndim != 2 or y.shape[0] < 1:
        raise ValueError("tensor POD requires a nonempty output matrix")
    tol = float(relative_tail_tolerance)
    if not 0.0 < tol < 1.0:
        raise ValueError("pod_relative_tail_tolerance must lie in (0, 1)")
    mean = np.mean(y, axis=0)
    scale = np.maximum(np.std(y, axis=0), 1e-12)
    normalized = (y - mean) / scale
    gram = normalized @ normalized.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    vectors = vectors[:, order]
    singular = np.sqrt(eigenvalues)
    total = float(np.sum(eigenvalues))
    if total <= np.finfo(float).tiny:
        basis = np.zeros((y.shape[1], 1), float)
        basis[0, 0] = 1.0
        return mean, scale, basis, 0.0

    positive = singular > np.sqrt(np.finfo(float).eps) * singular[0]
    singular = singular[positive]
    vectors = vectors[:, positive]
    if singular.size == 0:
        basis = np.zeros((y.shape[1], 1), float)
        basis[0, 0] = 1.0
        return mean, scale, basis, 0.0

    cumulative = np.cumsum(singular ** 2)
    tails = np.sqrt(np.maximum(total - cumulative, 0.0) / total)
    valid = np.flatnonzero(tails <= tol)
    rank = int(valid[0] + 1) if valid.size else len(singular)
    u = vectors[:, :rank]
    s = singular[:rank]
    basis = normalized.T @ (u / s[None, :])
    # Re-orthogonalize the small POD span to remove dual-form roundoff.
    basis, _ = np.linalg.qr(basis, mode="reduced")
    return mean, scale, np.asarray(basis, float), float(tails[rank - 1])


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
        value = torch.complex(packed[..., start + k], packed[..., start + len(pairs) + k])
        h[..., i, j] = value
        h[..., j, i] = value.conj()
    return h



def _unpack_real_symmetric_torch(torch, packed, n):
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    if packed.shape[-1] != len(pairs):
        raise ValueError("real symmetric torch packed size mismatch")
    out = torch.zeros(
        (*packed.shape[:-1], n, n),
        dtype=packed.dtype,
        device=packed.device,
    )
    for k, (i, j) in enumerate(pairs):
        out[..., i, j] = packed[..., k]
        out[..., j, i] = packed[..., k]
    return out


def _spatial_global_physics_penalty(torch, physical, n_ports):
    n = int(n_ports)
    h_size = n * n
    s_size = n * (n + 1) // 2
    if physical.shape[-1] != h_size + 2 * s_size:
        raise ValueError("spatial global physical width mismatch")
    d = _unpack_h_torch(
        torch,
        physical[..., :h_size],
        n,
    )
    outward_real = _unpack_real_symmetric_torch(
        torch,
        physical[..., h_size:h_size + s_size],
        n,
    )
    outward = torch.complex(
        outward_real,
        -d.imag,
    )
    d_penalty = torch.mean(
        torch.relu(
            -torch.linalg.eigvalsh(d).real
        ) ** 2
    )
    outward_penalty = torch.mean(
        torch.relu(
            -torch.linalg.eigvalsh(outward).real
        ) ** 2
    )
    return 0.5 * (d_penalty + outward_penalty)


def _physics_penalty(torch, physical, n_ports, thermal_rank, phi_min, phi_max):
    n = int(n_ports)
    r = int(thermal_rank)
    z_size, h_size, _ = tensor_block_sizes(n, r)
    z = _unpack_z_torch(torch, physical[..., :z_size], n)
    d = _unpack_h_torch(torch, physical[..., z_size:z_size + h_size], n)
    penalty = torch.mean(torch.relu(-torch.linalg.eigvalsh(d).real) ** 2)
    outward = z.real.to(d.dtype) - d
    penalty = penalty + torch.mean(torch.relu(-torch.linalg.eigvalsh(outward).real) ** 2)
    start = z_size + h_size
    for j in range(r):
        h = _unpack_h_torch(torch, physical[..., start + j * h_size:start + (j + 1) * h_size], n)
        lower = phi_min[..., j].reshape(-1, 1, 1).to(d.dtype)
        upper = phi_max[..., j].reshape(-1, 1, 1).to(d.dtype)
        penalty = penalty + torch.mean(torch.relu(-torch.linalg.eigvalsh(h - lower * d).real) ** 2)
        penalty = penalty + torch.mean(torch.relu(-torch.linalg.eigvalsh(upper * d - h).real) ** 2)
    return penalty / max(1, 2 + 2 * r)


def _relative_numpy(diff, truth, weights=None):
    d = np.asarray(diff, float)
    t = np.asarray(truth, float)
    if weights is None:
        num = float(np.sum(d * d))
        den = max(float(np.sum(t * t)), np.finfo(float).tiny)
    else:
        num = float(np.sum(d * d * weights))
        den = max(float(np.sum(t * t * weights)), np.finfo(float).tiny)
    return float(np.sqrt(num / den))


def train_matrix_tensor_surrogate(
    dataset,
    *,
    network_settings=None,
    training_settings=None,
    device="cuda",
    monitor=None,
    checkpoint_path=None,
):
    """Fit training-only POD and MLP; modal feasibility uses per-geometry bounds."""
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
    bounds_min = torch.as_tensor(dataset.phi_min, dtype=dtype, device=resolved)
    bounds_max = torch.as_tensor(dataset.phi_max, dtype=dtype, device=resolved)
    out_mean = torch.as_tensor(output_mean, dtype=dtype, device=resolved)
    out_scale = torch.as_tensor(output_scale, dtype=dtype, device=resolved)
    pod_t = torch.as_tensor(pod_basis, dtype=dtype, device=resolved)
    z_w_np, h_w_np = _matrix_weights(dataset.n_ports)
    z_w = torch.as_tensor(z_w_np, dtype=dtype, device=resolved)
    h_w = torch.as_tensor(h_w_np, dtype=dtype, device=resolved)
    n = int(dataset.n_ports)
    r = int(dataset.thermal_rank)
    z_size, h_size, _ = tensor_block_sizes(n, r)

    def relative_block(diff, truth, weights):
        numerator = torch.sum(diff * diff * weights, dim=-1)
        denominator = torch.sum(truth * truth * weights, dim=-1).clamp_min(torch.finfo(dtype).eps)
        return numerator / denominator

    def physical_prediction(xb):
        return out_mean + out_scale * (network(xb) @ pod_t.T)

    def batch_loss(ids):
        ids_t = torch.as_tensor(np.asarray(ids, np.int64), dtype=torch.long, device=resolved)
        pred = physical_prediction(x.index_select(0, ids_t))
        truth = target.index_select(0, ids_t)
        diff = pred - truth
        loss_z = torch.mean(relative_block(diff[..., :z_size], truth[..., :z_size], z_w))
        loss_d = torch.mean(relative_block(
            diff[..., z_size:z_size + h_size], truth[..., z_size:z_size + h_size], h_w
        ))
        start = z_size + h_size
        h_losses = []
        for j in range(r):
            sl = slice(start + j * h_size, start + (j + 1) * h_size)
            h_losses.append(torch.mean(relative_block(diff[..., sl], truth[..., sl], h_w)))
        loss_h = torch.stack(h_losses).mean() if h_losses else torch.zeros((), dtype=dtype, device=resolved)
        loss = float(cfg["z_weight"]) * loss_z + float(cfg["d_weight"]) * loss_d + float(cfg["h_weight"]) * loss_h
        if float(cfg["physics_penalty_weight"]) > 0.0:
            physical_scale = torch.mean(truth * truth).clamp_min(torch.finfo(dtype).eps)
            penalty = _physics_penalty(
                torch,
                pred,
                n,
                r,
                bounds_min.index_select(0, ids_t),
                bounds_max.index_select(0, ids_t),
            ) / physical_scale
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
                int(saved.get("schema_version", -1)) == 3
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
                torch.nn.utils.clip_grad_norm_(network.parameters(), float(cfg["gradient_clip_norm"]))
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
            if (not np.isfinite(best_val)) or last_val < best_val - 1e-10 * max(1.0, abs(best_val)):
                best_val = last_val
                best_epoch = epochs_completed
                best_state = copy.deepcopy(network.state_dict())
                stale = 0
            else:
                stale += interval
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "schema_version": 3,
                    "pod_signature": pod_signature,
                    "pod_rank": pod_rank,
                    "epoch": epochs_completed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val,
                    "network": network.state_dict(),
                    "best_network": best_state,
                    "optimizer": optimizer.state_dict(),
                    "stale": stale,
                }, checkpoint)
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
    surrogate = UnifiedTensorSurrogate(network, output_mean, output_scale, pod_basis, n, r)
    with torch.no_grad():
        test_loss = float(batch_loss(test_ids).detach().cpu())

    def evaluate(ids):
        decoded = []
        zd = []
        hc = []
        with torch.no_grad():
            for idx in ids:
                pred = physical_prediction(x[idx:idx + 1]).detach().cpu().numpy()[0]
                tensor = decode_physical_tensors(
                    pred,
                    n,
                    dataset.phi_min[idx],
                    dataset.phi_max[idx],
                )
                decoded.append(pack_tensors(tensor.z_field, tensor.d_vol, tensor.modal_h))
                zd.append(tensor.zd_projection_correction)
                hc.append(tensor.h_projection_correction)
        decoded = np.asarray(decoded, float)
        truth = dataset.outputs[np.asarray(ids, int)]
        diff = decoded - truth
        start = z_size + h_size
        h_weights = np.tile(h_w_np, r) if r else np.empty(0)
        full_weights = np.concatenate([z_w_np, h_w_np, h_weights])
        return {
            "tensor": _relative_numpy(diff, truth, full_weights),
            "z": _relative_numpy(diff[:, :z_size], truth[:, :z_size], z_w_np),
            "d": _relative_numpy(diff[:, z_size:z_size + h_size], truth[:, z_size:z_size + h_size], h_w_np),
            "h": _relative_numpy(diff[:, start:], truth[:, start:], h_weights) if r else 0.0,
            "zd": max(zd or [0.0]),
            "hc": max(hc or [0.0]),
        }

    test = evaluate(test_ids)
    audit = evaluate(audit_ids)
    return surrogate, TensorTrainingReport(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_loss=float(best_val),
        test_loss=test_loss,
        pod_rank=pod_rank,
        pod_relative_tail_error=pod_tail,
        test_relative_tensor_error=test["tensor"],
        test_z_relative_error=test["z"],
        test_d_relative_error=test["d"],
        test_h_relative_error=test["h"],
        audit_relative_tensor_error=audit["tensor"],
        audit_z_relative_error=audit["z"],
        audit_d_relative_error=audit["d"],
        audit_h_relative_error=audit["h"],
        maximum_test_zd_projection_correction=test["zd"],
        maximum_test_h_projection_correction=test["hc"],
        maximum_audit_zd_projection_correction=audit["zd"],
        maximum_audit_h_projection_correction=audit["hc"],
        device=resolved,
        network_config=net_cfg.to_dict(),
        training_config=dict(cfg),
    )



def train_spatial_tensor_surrogate(
    dataset,
    *,
    background,
    network_settings=None,
    training_settings=None,
    device="cuda",
    monitor=None,
    checkpoint_path=None,
):
    """Fit the production two-head spatial surrogate.

    The global MLP predicts reciprocal/passive Z/D.  A second coordinate-
    conditioned field MLP predicts the whitened local Joule shape.  The
    production decoder projects that field to PSD and enforces exact
    sum(H_cell)=D, so no world-grid spatial POD is used.
    """
    import torch

    cfg = {
        "epochs": 240,
        "batch_size": 16,
        "field_batch_size": 4096,
        "field_cells_per_geometry": 2048,
        "learning_rate": 1e-3,
        "global_learning_rate": 5e-4,
        "field_learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "global_weight_decay": 1e-3,
        "field_weight_decay": 1e-6,
        "patience": 40,
        "validation_interval": 2,
        "gradient_clip_norm": 10.0,
        "physics_penalty_weight": 0.05,
        "z_weight": 1.0,
        "d_weight": 1.0,
        "outward_weight": 1.0,
        "spatial_weight": 1.0,
        "field_density_weight": 1.0,
        "field_shape_weight": 1.0,
        "refit_all_truth": True,
        "refit_epochs": 24,
        "refit_learning_rate_factor": 0.25,
        "seed": 17,
        "dtype": "float32",
    }
    incoming = dict(training_settings or {})
    if "spatial_weight" not in incoming and "h_weight" in incoming:
        incoming["spatial_weight"] = incoming["h_weight"]
    cfg.update(incoming)

    train_ids = dataset.indices("train")
    val_ids = dataset.indices("validation")
    test_ids = dataset.indices("test")
    audit_ids = dataset.indices("audit")
    if min(len(train_ids), len(val_ids), len(test_ids), len(audit_ids)) < 1:
        raise ValueError(
            "spatial tensor dataset requires nonempty train/validation/test/audit splits"
        )

    n = int(dataset.n_ports)
    m = int(dataset.n_cells)
    if int(background.n_cells) != m:
        raise ValueError(
            "spatial tensor dataset cell count differs from training background"
        )
    z_size, h_size, _ = tensor_block_sizes(n, 0)
    symmetric_size = n * (n + 1) // 2
    global_dim = h_size + 2 * symmetric_size
    field_dim = 1 + n * n

    geometries = [
        decode_geometry_encoding(row, n)
        for row in np.asarray(dataset.inputs, float)
    ]
    global_inputs_np = np.asarray(
        [
            encode_geometry_invariant(g)
            for g in geometries
        ],
        float,
    )
    global_targets_np = []
    for packed in np.asarray(dataset.outputs, float):
        z_truth, d_truth, _cells_truth = unpack_spatial_tensors(
            packed,
            n,
            m,
        )
        global_targets_np.append(
            pack_spatial_global_tensors(
                z_truth,
                d_truth,
            )
        )
    global_targets_np = np.asarray(
        global_targets_np,
        float,
    )

    seed = int(cfg["seed"])
    sample_count = max(1, int(cfg["field_cells_per_geometry"]))

    def sampled_field_rows(geometry_ids, seed_offset):
        local_rng = np.random.default_rng(seed + int(seed_offset))
        feature_rows = []
        target_rows = []
        for sample_id in np.asarray(geometry_ids, int):
            _z, d, cells = unpack_spatial_tensors(
                dataset.outputs[int(sample_id)],
                n,
                m,
            )
            whitened = whiten_cell_joule_tensors(cells, d)
            k = min(m, sample_count)
            if k >= m:
                cell_ids = np.arange(m, dtype=int)
            else:
                importance_count = k // 2
                uniform_count = k - importance_count
                score = np.sqrt(
                    np.sum(
                        np.abs(whitened) ** 2,
                        axis=(1, 2),
                    )
                )
                score = np.asarray(score, float)
                if (
                    importance_count > 0
                    and np.sum(score) > np.finfo(float).tiny
                ):
                    weighted = local_rng.choice(
                        m,
                        size=importance_count,
                        replace=False,
                        p=score / np.sum(score),
                    )
                else:
                    weighted = np.empty(0, dtype=int)
                available = np.setdiff1d(
                    np.arange(m, dtype=int),
                    weighted,
                    assume_unique=False,
                )
                uniform = local_rng.choice(
                    available,
                    size=min(uniform_count, available.size),
                    replace=False,
                )
                cell_ids = np.unique(
                    np.concatenate((weighted, uniform))
                )
                if cell_ids.size < k:
                    remaining = np.setdiff1d(
                        np.arange(m, dtype=int),
                        cell_ids,
                        assume_unique=False,
                    )
                    fill = local_rng.choice(
                        remaining,
                        size=k - cell_ids.size,
                        replace=False,
                    )
                    cell_ids = np.concatenate((cell_ids, fill))

            feature_rows.append(
                spatial_cell_features(
                    background,
                    geometries[int(sample_id)],
                    cell_ids,
                )
            )
            target = pack_whitened_field_factors(
                whitened[cell_ids],
                total_cells=m,
            )
            target_rows.append(target)
        features = np.vstack(feature_rows).astype(
            float,
            copy=False,
        )
        targets = np.vstack(target_rows).astype(
            float,
            copy=False,
        )
        # Shape/orientation is irrelevant where local power density is tiny.
        # Weight its loss by the dimensionless density while always learning
        # log-density itself across the whole domain.
        shape_weight = np.minimum(
            np.exp(np.clip(targets[:, 0], -30.0, 30.0)),
            10.0,
        )
        shape_weight = np.maximum(shape_weight, 1e-3)
        shape_weight /= max(
            float(np.mean(shape_weight)),
            np.finfo(float).tiny,
        )
        return features, targets, shape_weight

    (
        field_train_x_np,
        field_train_y_np,
        field_train_shape_weight_np,
    ) = sampled_field_rows(
        train_ids,
        104729,
    )
    (
        field_val_x_np,
        field_val_y_np,
        field_val_shape_weight_np,
    ) = sampled_field_rows(
        val_ids,
        130363,
    )

    def output_normalizer(values):
        value = np.asarray(values, float)
        mean = np.mean(value, axis=0)
        spread = np.std(value, axis=0)
        magnitude = np.max(np.abs(value), axis=0)
        scale = np.maximum(
            spread,
            np.maximum(1e-6 * magnitude, 1e-12),
        )
        return mean, scale

    global_output_mean, global_output_scale = output_normalizer(
        global_targets_np[train_ids]
    )
    global_targets_normalized_np = (
        global_targets_np - global_output_mean
    ) / global_output_scale

    field_output_mean, field_output_scale = output_normalizer(
        field_train_y_np
    )
    field_train_y_normalized_np = (
        field_train_y_np - field_output_mean
    ) / field_output_scale
    field_val_y_normalized_np = (
        field_val_y_np - field_output_mean
    ) / field_output_scale

    global_norm = FeatureNormalizer.fit(
        global_inputs_np[train_ids]
    )
    field_norm = FeatureNormalizer.fit(field_train_x_np)

    net_settings = dict(network_settings or {})
    if (
        isinstance(net_settings.get("global"), dict)
        or isinstance(net_settings.get("field"), dict)
    ):
        shared = {
            k: v
            for k, v in net_settings.items()
            if k not in {"global", "field"}
        }
        global_settings = {
            **shared,
            **dict(net_settings.get("global", {})),
        }
        field_settings = {
            **shared,
            **dict(net_settings.get("field", {})),
        }
    else:
        global_settings = dict(net_settings)
        field_settings = dict(net_settings)
    for settings_part in (global_settings, field_settings):
        settings_part.pop("input_dimension", None)
        settings_part.pop("output_dimension", None)
    global_cfg = ResidualMLPConfig(
        input_dimension=global_inputs_np.shape[1],
        output_dimension=global_dim,
        **global_settings,
    )
    field_cfg = ResidualMLPConfig(
        input_dimension=field_train_x_np.shape[1],
        output_dimension=field_dim,
        **field_settings,
    )

    resolved = _resolve_device(torch, device)
    dtype = (
        torch.float32
        if str(cfg["dtype"]) == "float32"
        else torch.float64
    )
    global_network = build_residual_mlp(
        global_cfg,
        global_norm,
    ).to(device=resolved, dtype=dtype)
    field_network = build_residual_mlp(
        field_cfg,
        field_norm,
    ).to(device=resolved, dtype=dtype)

    global_optimizer = torch.optim.AdamW(
        global_network.parameters(),
        lr=float(
            cfg.get(
                "global_learning_rate",
                cfg["learning_rate"],
            )
        ),
        weight_decay=float(
            cfg.get(
                "global_weight_decay",
                cfg["weight_decay"],
            )
        ),
    )
    field_optimizer = torch.optim.AdamW(
        field_network.parameters(),
        lr=float(
            cfg.get(
                "field_learning_rate",
                cfg["learning_rate"],
            )
        ),
        weight_decay=float(
            cfg.get(
                "field_weight_decay",
                cfg["weight_decay"],
            )
        ),
    )

    geometry_x = torch.as_tensor(
        global_inputs_np,
        dtype=dtype,
        device=resolved,
    )
    global_target = torch.as_tensor(
        global_targets_np,
        dtype=dtype,
        device=resolved,
    )
    global_target_normalized = torch.as_tensor(
        global_targets_normalized_np,
        dtype=dtype,
        device=resolved,
    )
    global_output_mean_t = torch.as_tensor(
        global_output_mean,
        dtype=dtype,
        device=resolved,
    )
    global_output_scale_t = torch.as_tensor(
        global_output_scale,
        dtype=dtype,
        device=resolved,
    )
    field_train_x = torch.as_tensor(
        field_train_x_np,
        dtype=dtype,
        device=resolved,
    )
    field_train_y = torch.as_tensor(
        field_train_y_normalized_np,
        dtype=dtype,
        device=resolved,
    )
    field_val_x = torch.as_tensor(
        field_val_x_np,
        dtype=dtype,
        device=resolved,
    )
    field_val_y = torch.as_tensor(
        field_val_y_normalized_np,
        dtype=dtype,
        device=resolved,
    )
    field_train_shape_weight = torch.as_tensor(
        field_train_shape_weight_np,
        dtype=dtype,
        device=resolved,
    )
    field_val_shape_weight = torch.as_tensor(
        field_val_shape_weight_np,
        dtype=dtype,
        device=resolved,
    )

    z_w_np, h_w_np = _matrix_weights(n)

    def global_loss(ids):
        ids_t = torch.as_tensor(
            np.asarray(ids, np.int64),
            dtype=torch.long,
            device=resolved,
        )
        truth = global_target.index_select(0, ids_t)
        truth_normalized = global_target_normalized.index_select(
            0,
            ids_t,
        )
        pred_normalized = global_network(
            geometry_x.index_select(0, ids_t)
        )
        diff = pred_normalized - truth_normalized
        d_loss = torch.mean(
            diff[..., :h_size] ** 2
        )
        outward_loss = torch.mean(
            diff[
                ...,
                h_size:h_size + symmetric_size,
            ] ** 2
        )
        reactance_loss = torch.mean(
            diff[
                ...,
                h_size + symmetric_size:,
            ] ** 2
        )
        loss = (
            float(cfg["d_weight"]) * d_loss
            + float(cfg["outward_weight"]) * outward_loss
            + float(cfg["z_weight"]) * reactance_loss
        )
        if float(cfg["physics_penalty_weight"]) > 0.0:
            physical_pred = (
                global_output_mean_t
                + global_output_scale_t * pred_normalized
            )
            physical_scale = torch.mean(
                truth * truth
            ).clamp_min(torch.finfo(dtype).eps)
            loss = loss + float(cfg["physics_penalty_weight"]) * (
                _spatial_global_physics_penalty(
                    torch,
                    physical_pred,
                    n,
                )
                / physical_scale
            )
        return loss

    def field_loss(xb, yb, shape_weight):
        pred = field_network(xb)
        diff = pred - yb
        density_loss = torch.mean(
            diff[..., 0] ** 2
        )
        shape_error = torch.mean(
            diff[..., 1:] ** 2,
            dim=-1,
        )
        weighted_shape = torch.sum(
            shape_weight * shape_error
        ) / torch.sum(shape_weight).clamp_min(
            torch.finfo(dtype).eps
        )
        return (
            float(cfg["field_density_weight"])
            * density_loss
            + float(cfg["field_shape_weight"])
            * weighted_shape
        )

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    signature_hasher = hashlib.sha256()
    signature_hasher.update(
        np.ascontiguousarray(global_inputs_np).tobytes()
    )
    signature_hasher.update(
        np.ascontiguousarray(global_targets_np).tobytes()
    )
    signature_hasher.update(
        np.ascontiguousarray(field_train_x_np).tobytes()
    )
    signature_hasher.update(
        np.ascontiguousarray(field_train_y_np).tobytes()
    )
    signature_hasher.update(
        np.ascontiguousarray(global_output_mean).tobytes()
    )
    signature_hasher.update(
        np.ascontiguousarray(global_output_scale).tobytes()
    )
    signature_hasher.update(
        np.ascontiguousarray(field_output_mean).tobytes()
    )
    signature_hasher.update(
        np.ascontiguousarray(field_output_scale).tobytes()
    )
    training_signature = signature_hasher.hexdigest()

    checkpoint = (
        None
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    best_global = copy.deepcopy(
        global_network.state_dict()
    )
    best_field = copy.deepcopy(
        field_network.state_dict()
    )
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    start_epoch = 0

    if checkpoint is not None and checkpoint.is_file():
        try:
            saved = torch.load(
                checkpoint,
                map_location=resolved,
                weights_only=False,
            )
            compatible = (
                int(saved.get("schema_version", -1)) == 7
                and saved.get("representation")
                == "cellwise_joule_neural_field_v3"
                and saved.get("training_signature")
                == training_signature
            )
            if compatible:
                global_network.load_state_dict(
                    saved["global_network"]
                )
                field_network.load_state_dict(
                    saved["field_network"]
                )
                global_optimizer.load_state_dict(
                    saved["global_optimizer"]
                )
                field_optimizer.load_state_dict(
                    saved["field_optimizer"]
                )
                best_global = saved["best_global_network"]
                best_field = saved["best_field_network"]
                best_val = float(
                    saved["best_validation_loss"]
                )
                best_epoch = int(saved["best_epoch"])
                stale = int(saved.get("stale", 0))
                start_epoch = int(saved["epoch"])
                print(
                    "恢复 geometry→spatial-Joule neural-field "
                    f"检查点：epoch={start_epoch}",
                    flush=True,
                )
            else:
                print(
                    "spatial-Joule 旧 POD 检查点与 v2 neural-field "
                    "架构不兼容，重新训练网络（truth cache 保留）。",
                    flush=True,
                )
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
        ):
            print(
                "spatial-Joule neural-field 检查点不可用，重新训练网络。",
                flush=True,
            )

    interval = max(
        1,
        int(cfg["validation_interval"]),
    )
    geometry_batch = max(
        1,
        int(cfg["batch_size"]),
    )
    field_batch = max(
        1,
        int(cfg["field_batch_size"]),
    )
    epochs_completed = start_epoch
    last_train = float("inf")
    last_val = float("inf")

    for epoch in range(
        start_epoch,
        int(cfg["epochs"]),
    ):
        if monitor is not None:
            monitor.checkpoint()

        global_network.train()
        field_network.train()

        order = rng.permutation(train_ids)
        global_total = 0.0
        global_count = 0
        for start_batch in range(
            0,
            len(order),
            geometry_batch,
        ):
            ids = order[
                start_batch:start_batch + geometry_batch
            ]
            loss = global_loss(ids)
            global_optimizer.zero_grad(
                set_to_none=True
            )
            loss.backward()
            if cfg.get(
                "gradient_clip_norm"
            ) is not None:
                torch.nn.utils.clip_grad_norm_(
                    global_network.parameters(),
                    float(cfg["gradient_clip_norm"]),
                )
            global_optimizer.step()
            global_total += (
                float(loss.detach().cpu())
                * len(ids)
            )
            global_count += len(ids)

        field_order = rng.permutation(
            field_train_x.shape[0]
        )
        field_total = 0.0
        field_count = 0
        for start_batch in range(
            0,
            len(field_order),
            field_batch,
        ):
            ids_np = field_order[
                start_batch:start_batch + field_batch
            ]
            ids_t = torch.as_tensor(
                ids_np,
                dtype=torch.long,
                device=resolved,
            )
            loss = field_loss(
                field_train_x.index_select(0, ids_t),
                field_train_y.index_select(0, ids_t),
                field_train_shape_weight.index_select(
                    0,
                    ids_t,
                ),
            )
            field_optimizer.zero_grad(
                set_to_none=True
            )
            loss.backward()
            if cfg.get(
                "gradient_clip_norm"
            ) is not None:
                torch.nn.utils.clip_grad_norm_(
                    field_network.parameters(),
                    float(cfg["gradient_clip_norm"]),
                )
            field_optimizer.step()
            field_total += (
                float(loss.detach().cpu())
                * len(ids_np)
            )
            field_count += len(ids_np)

        global_train = (
            global_total / max(global_count, 1)
        )
        field_train = (
            field_total / max(field_count, 1)
        )
        last_train = (
            global_train
            + float(cfg["spatial_weight"])
            * field_train
        )
        epochs_completed = epoch + 1

        validate = (
            epochs_completed == 1
            or epochs_completed % interval == 0
            or epochs_completed == int(cfg["epochs"])
        )
        if validate:
            global_network.eval()
            field_network.eval()
            with torch.no_grad():
                val_global = float(
                    global_loss(val_ids).detach().cpu()
                )
                val_field_total = 0.0
                val_field_count = 0
                for start_batch in range(
                    0,
                    field_val_x.shape[0],
                    field_batch,
                ):
                    xb = field_val_x[
                        start_batch:start_batch + field_batch
                    ]
                    yb = field_val_y[
                        start_batch:start_batch + field_batch
                    ]
                    wb = field_val_shape_weight[
                        start_batch:start_batch + field_batch
                    ]
                    value = float(
                        field_loss(xb, yb, wb)
                        .detach()
                        .cpu()
                    )
                    val_field_total += (
                        value * xb.shape[0]
                    )
                    val_field_count += xb.shape[0]
                val_field = (
                    val_field_total
                    / max(val_field_count, 1)
                )
                last_val = (
                    val_global
                    + float(cfg["spatial_weight"])
                    * val_field
                )

            if (
                not np.isfinite(best_val)
                or last_val
                < best_val
                - 1e-10 * max(1.0, abs(best_val))
            ):
                best_val = last_val
                best_epoch = epochs_completed
                best_global = copy.deepcopy(
                    global_network.state_dict()
                )
                best_field = copy.deepcopy(
                    field_network.state_dict()
                )
                stale = 0
            else:
                stale += interval

            if checkpoint is not None:
                checkpoint.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                torch.save(
                    {
                        "schema_version": 7,
                        "representation":
                            "cellwise_joule_neural_field_v3",
                        "training_signature":
                            training_signature,
                        "epoch": epochs_completed,
                        "best_epoch": best_epoch,
                        "best_validation_loss":
                            best_val,
                        "global_network":
                            global_network.state_dict(),
                        "field_network":
                            field_network.state_dict(),
                        "best_global_network":
                            best_global,
                        "best_field_network":
                            best_field,
                        "global_optimizer":
                            global_optimizer.state_dict(),
                        "field_optimizer":
                            field_optimizer.state_dict(),
                        "stale": stale,
                    },
                    checkpoint,
                )

            if monitor is not None:
                with monitor._lock:
                    monitor.data.update(
                        phase="spatial_tensor_training",
                        epoch=epochs_completed,
                        train_loss=last_train,
                        validation_loss=last_val,
                        global_validation_loss=val_global,
                        field_validation_loss=val_field,
                        tensor_pod_rank=0,
                        tensor_representation=(
                            "cellwise_joule_neural_field_v2"
                        ),
                    )
            print(
                "训练 geometry→spatial-Joule neural field……"
                f"epoch={epochs_completed}/{cfg['epochs']}  "
                f"train={last_train:.5g} "
                f"val={last_val:.5g} "
                f"global={val_global:.5g} "
                f"field={val_field:.5g}",
                flush=True,
            )
            if stale >= int(cfg["patience"]):
                break

    global_network.load_state_dict(best_global)
    field_network.load_state_dict(best_field)

    refit_epochs_completed = 0
    if bool(cfg.get("refit_all_truth", True)):
        refit_ids = np.arange(
            len(dataset.inputs),
            dtype=int,
        )
        (
            field_refit_x_np,
            field_refit_y_np,
            field_refit_shape_weight_np,
        ) = sampled_field_rows(
            refit_ids,
            161803,
        )
        field_refit_y_normalized_np = (
            field_refit_y_np - field_output_mean
        ) / field_output_scale
        field_refit_x = torch.as_tensor(
            field_refit_x_np,
            dtype=dtype,
            device=resolved,
        )
        field_refit_y = torch.as_tensor(
            field_refit_y_normalized_np,
            dtype=dtype,
            device=resolved,
        )
        field_refit_shape_weight = torch.as_tensor(
            field_refit_shape_weight_np,
            dtype=dtype,
            device=resolved,
        )

        factor = float(
            cfg.get(
                "refit_learning_rate_factor",
                0.25,
            )
        )
        if not 0.0 < factor <= 1.0:
            raise ValueError(
                "refit_learning_rate_factor must lie in (0,1]"
            )
        # Best weights may come from an earlier epoch than the optimizer's
        # current moment state.  Rebuild AdamW so the refit starts from a
        # coherent (weights, optimizer-state) pair.
        global_optimizer = torch.optim.AdamW(
            global_network.parameters(),
            lr=float(
                cfg.get(
                    "global_learning_rate",
                    cfg["learning_rate"],
                )
            ) * factor,
            weight_decay=float(
                cfg.get(
                    "global_weight_decay",
                    cfg["weight_decay"],
                )
            ),
        )
        field_optimizer = torch.optim.AdamW(
            field_network.parameters(),
            lr=float(
                cfg.get(
                    "field_learning_rate",
                    cfg["learning_rate"],
                )
            ) * factor,
            weight_decay=float(
                cfg.get(
                    "field_weight_decay",
                    cfg["weight_decay"],
                )
            ),
        )

        requested_refit = max(
            1,
            int(cfg.get("refit_epochs", 24)),
        )
        for refit_epoch in range(requested_refit):
            if monitor is not None:
                monitor.checkpoint()
            global_network.train()
            field_network.train()

            order = rng.permutation(refit_ids)
            for start_batch in range(
                0,
                len(order),
                geometry_batch,
            ):
                ids = order[
                    start_batch:
                    start_batch + geometry_batch
                ]
                loss = global_loss(ids)
                global_optimizer.zero_grad(
                    set_to_none=True
                )
                loss.backward()
                if cfg.get(
                    "gradient_clip_norm"
                ) is not None:
                    torch.nn.utils.clip_grad_norm_(
                        global_network.parameters(),
                        float(
                            cfg[
                                "gradient_clip_norm"
                            ]
                        ),
                    )
                global_optimizer.step()

            field_order = rng.permutation(
                field_refit_x.shape[0]
            )
            for start_batch in range(
                0,
                len(field_order),
                field_batch,
            ):
                ids_np = field_order[
                    start_batch:
                    start_batch + field_batch
                ]
                ids_t = torch.as_tensor(
                    ids_np,
                    dtype=torch.long,
                    device=resolved,
                )
                loss = field_loss(
                    field_refit_x.index_select(
                        0,
                        ids_t,
                    ),
                    field_refit_y.index_select(
                        0,
                        ids_t,
                    ),
                    field_refit_shape_weight.index_select(
                        0,
                        ids_t,
                    ),
                )
                field_optimizer.zero_grad(
                    set_to_none=True
                )
                loss.backward()
                if cfg.get(
                    "gradient_clip_norm"
                ) is not None:
                    torch.nn.utils.clip_grad_norm_(
                        field_network.parameters(),
                        float(
                            cfg[
                                "gradient_clip_norm"
                            ]
                        ),
                    )
                field_optimizer.step()

            refit_epochs_completed = refit_epoch + 1
            if (
                monitor is not None
                and (
                    refit_epochs_completed == 1
                    or refit_epochs_completed % 4 == 0
                    or refit_epochs_completed
                    == requested_refit
                )
            ):
                global_network.eval()
                field_network.eval()
                with torch.no_grad():
                    refit_global_loss = float(
                        global_loss(refit_ids)
                        .detach()
                        .cpu()
                    )
                    # Deterministic diagnostic subset; training still uses
                    # every sampled refit field row above.
                    diagnostic_count = min(
                        field_batch,
                        field_refit_x.shape[0],
                    )
                    refit_field_loss = float(
                        field_loss(
                            field_refit_x[
                                :diagnostic_count
                            ],
                            field_refit_y[
                                :diagnostic_count
                            ],
                            field_refit_shape_weight[
                                :diagnostic_count
                            ],
                        )
                        .detach()
                        .cpu()
                    )
                with monitor._lock:
                    monitor.data.update(
                        phase="spatial_tensor_refit",
                        refit_epoch=(
                            refit_epochs_completed
                        ),
                        refit_global_loss=(
                            refit_global_loss
                        ),
                        refit_field_loss=(
                            refit_field_loss
                        ),
                    )

    global_network.eval()
    field_network.eval()

    surrogate = UnifiedSpatialTensorSurrogate(
        global_network,
        field_network,
        n,
        m,
        global_output_mean=global_output_mean,
        global_output_scale=global_output_scale,
        field_output_mean=field_output_mean,
        field_output_scale=field_output_scale,
    )

    full_weights = np.concatenate(
        (
            z_w_np,
            h_w_np,
            np.tile(h_w_np, m),
        )
    )

    def evaluate(ids):
        decoded_rows = []
        truth_rows = []
        zd = []
        spatial_correction = []
        for sample_id in np.asarray(ids, int):
            predicted = surrogate.predict(
                geometries[int(sample_id)],
                background=background,
            )
            decoded_rows.append(
                pack_spatial_tensors(
                    predicted.z_field,
                    predicted.d_vol,
                    predicted.cell_h,
                )
            )
            truth_rows.append(
                np.asarray(
                    dataset.outputs[int(sample_id)],
                    float,
                )
            )
            zd.append(
                float(
                    predicted.zd_projection_correction
                )
            )
            spatial_correction.append(
                float(
                    predicted.spatial_projection_correction
                )
            )
        decoded = np.asarray(decoded_rows, float)
        truth = np.asarray(truth_rows, float)
        diff = decoded - truth
        return {
            "tensor": _relative_numpy(
                diff,
                truth,
                full_weights,
            ),
            "z": _relative_numpy(
                diff[:, :z_size],
                truth[:, :z_size],
                z_w_np,
            ),
            "d": _relative_numpy(
                diff[:, z_size:global_dim],
                truth[:, z_size:global_dim],
                h_w_np,
            ),
            "spatial": _relative_numpy(
                diff[:, global_dim:],
                truth[:, global_dim:],
                np.tile(h_w_np, m),
            ),
            "zd": max(zd or [0.0]),
            "spatial_correction": max(
                spatial_correction or [0.0]
            ),
        }

    test = evaluate(test_ids)
    audit = evaluate(audit_ids)
    report_cfg = dict(cfg)
    report_cfg["representation"] = (
        "cellwise_joule_neural_field_v2"
    )
    report_cfg["field_training_points"] = int(
        field_train_x_np.shape[0]
    )
    report_cfg["field_validation_points"] = int(
        field_val_x_np.shape[0]
    )
    report_cfg["final_refit_all_truth"] = bool(
        cfg.get("refit_all_truth", True)
    )
    report_cfg["final_refit_epochs_completed"] = int(
        refit_epochs_completed
    )

    return surrogate, TensorTrainingReport(
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_validation_loss=float(best_val),
        test_loss=float(test["tensor"]),
        pod_rank=0,
        pod_relative_tail_error=0.0,
        test_relative_tensor_error=test["tensor"],
        test_z_relative_error=test["z"],
        test_d_relative_error=test["d"],
        test_h_relative_error=test["spatial"],
        audit_relative_tensor_error=audit["tensor"],
        audit_z_relative_error=audit["z"],
        audit_d_relative_error=audit["d"],
        audit_h_relative_error=audit["spatial"],
        maximum_test_zd_projection_correction=test["zd"],
        maximum_test_h_projection_correction=(
            test["spatial_correction"]
        ),
        maximum_audit_zd_projection_correction=audit["zd"],
        maximum_audit_h_projection_correction=(
            audit["spatial_correction"]
        ),
        device=resolved,
        network_config={
            "global": global_cfg.to_dict(),
            "field": field_cfg.to_dict(),
        },
        training_config=report_cfg,
    )


__all__ = ["TensorTrainingReport", "train_matrix_tensor_surrogate", "train_spatial_tensor_surrogate"]
