from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
import math

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("controlled vNext training requires the 'neural' extra") from exc

from .device import resolve_torch_device
from .hybrid_neural import HybridNormalizer, HybridPhysicsFactoredResidualNet
from .hybrid_spatial_neural import HybridSpatialLossShapeNet
from .performance import (
    _batch_loss,
    _record_from_sample,
    _stack_records,
    _topology_buckets,
    resolve_training_dtype,
)
from .spatial_performance import (
    _batched_spatial_shape_loss,
    _sample_signature,
)
from .tensor_features import (
    TENSOR_COIL_FEATURE_DIM,
    TENSOR_PACKAGE_FEATURE_DIM,
    TENSOR_PAIR_FEATURE_DIM,
    TENSOR_CROSS_FEATURE_DIM,
)
from .tensor_neural import TensorHybridNeuralResidualArtifact, _sample_tensor_ranges
from .tensor_spatial_neural import (
    TensorHybridSpatialLossArtifact,
    _sample_end_to_end_error,
    tensor_port_fingerprint,
)
from .tensor_training_data import TensorHybridTeacherSample
from .tensor_spatial_training_data import TensorHybridSpatialTeacherSample
from .training_control import TrainingControl, TrainingStopRequested
from .workflow_cache import canonical_json


CHECKPOINT_SCHEMA = 1


def _hash(value) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _cpu_state_dict(state):
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def _port_signature(config, cache_key: str) -> str:
    cfg = dict(config)
    return _hash(
        {
            "schema": CHECKPOINT_SCHEMA,
            "kind": "tensor_port",
            "cache_key": cache_key,
            "model": cfg.get("model", {}),
            "optimizer": cfg.get("optimizer", {}),
            "batch_size": cfg.get("batch_size", 16),
            "precision": cfg.get("precision", "auto"),
            "geometry_domain": cfg.get("geometry_domain"),
            "validation_fraction": cfg.get("validation_fraction"),
        }
    )


def _spatial_signature(config, cache_key: str, port_fingerprint: str) -> str:
    cfg = dict(config)
    return _hash(
        {
            "schema": CHECKPOINT_SCHEMA,
            "kind": "tensor_spatial",
            "cache_key": cache_key,
            "port_fingerprint": port_fingerprint,
            "model": cfg.get("model", {}),
            "optimizer": cfg.get("optimizer", {}),
            "normalization": cfg.get("normalization", {}),
            "batch_size": cfg.get("batch_size", 8),
            "validation_fraction": cfg.get("validation_fraction"),
        }
    )


def deterministic_split(samples, fraction: float, seed: int):
    samples = tuple(samples)
    fraction = float(fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation fraction must lie in [0,1)")
    if not samples or fraction == 0.0 or len(samples) == 1:
        return samples, ()
    count = max(1, int(round(len(samples) * fraction)))
    count = min(count, len(samples) - 1)
    order = np.random.default_rng(int(seed) + 991).permutation(len(samples))
    validation = set(int(index) for index in order[:count])
    return (
        tuple(sample for index, sample in enumerate(samples) if index not in validation),
        tuple(sample for index, sample in enumerate(samples) if index in validation),
    )


def _schedule_from_buckets(buckets, batch_size: int, rng):
    keys = list(buckets)
    rng.shuffle(keys)
    result = []
    for key in keys:
        order = rng.permutation(np.asarray(buckets[key], dtype=int))
        for start in range(0, len(order), int(batch_size)):
            result.append([int(value) for value in order[start : start + int(batch_size)]])
    return result


def _save_port_checkpoint(
    path,
    *,
    signature,
    cache_key,
    status,
    model,
    optimizer,
    normalizer,
    baseline_segments,
    material_domain,
    geometry_domain,
    epoch,
    schedule,
    next_batch,
    epoch_total,
    epoch_seen,
    best_state,
    best_score,
    best_epoch,
    stale,
    history,
    rng,
):
    _atomic_torch_save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "kind": "tensor_port",
            "signature": signature,
            "cache_key": cache_key,
            "status": status,
            "model_config": {
                "coil_dim": model.coil_dim,
                "coil_pair_dim": model.coil_pair_dim,
                "package_dim": model.package_dim,
                "cross_dim": model.cross_dim,
                "package_pair_dim": model.package_pair_dim,
                "hidden_dim": model.hidden_dim,
                "factor_rank": model.factor_rank,
                "depth": model.depth,
            },
            "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "model_state": _cpu_state_dict(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "normalizer": normalizer.to_dict(),
            "baseline_segments": int(baseline_segments),
            "material_domain": material_domain,
            "geometry_domain": geometry_domain,
            "epoch": int(epoch),
            "schedule": schedule,
            "next_batch": int(next_batch),
            "epoch_total": float(epoch_total),
            "epoch_seen": int(epoch_seen),
            "best_state": None if best_state is None else _cpu_state_dict(best_state),
            "best_score": float(best_score),
            "best_epoch": int(best_epoch),
            "stale": int(stale),
            "history": list(history),
            "rng_state": deepcopy(rng.bit_generator.state),
        },
        path,
    )


def _build_port_model(model_config, *, dtype, device):
    return HybridPhysicsFactoredResidualNet(
        coil_dim=int(model_config.get("coil_dim", TENSOR_COIL_FEATURE_DIM)),
        coil_pair_dim=int(model_config.get("coil_pair_dim", TENSOR_PAIR_FEATURE_DIM)),
        package_dim=int(model_config.get("package_dim", TENSOR_PACKAGE_FEATURE_DIM)),
        cross_dim=int(model_config.get("cross_dim", TENSOR_CROSS_FEATURE_DIM)),
        package_pair_dim=int(model_config.get("package_pair_dim", TENSOR_PAIR_FEATURE_DIM)),
        hidden_dim=int(model_config.get("hidden_dim", 64)),
        factor_rank=int(model_config.get("factor_rank", 4)),
        depth=int(model_config.get("depth", 2)),
    ).to(device=device, dtype=dtype)


def train_port_controlled(
    samples,
    *,
    config,
    cache_key: str,
    checkpoint_path,
    control: TrainingControl,
    resume: bool = True,
):
    samples = tuple(samples)
    if not samples:
        raise ValueError("port training requires teacher samples")
    if any(not isinstance(sample, TensorHybridTeacherSample) for sample in samples):
        raise TypeError("port training requires TensorHybridTeacherSample values")
    cfg = dict(config)
    model_cfg = dict(cfg.get("model", {}) or {})
    optimizer_cfg = dict(cfg.get("optimizer", {}) or {})
    epochs = int(cfg.get("epochs", 200))
    batch_size = int(cfg.get("batch_size", 16))
    patience = int(cfg.get("patience", 30))
    min_improvement = float(cfg.get("min_improvement", 1e-5))
    channel_loss_weight = float(cfg.get("channel_loss_weight", 1.0))
    checkpoint_every_batches = max(1, int(cfg.get("checkpoint_every_batches", 1)))
    seed = int(cfg.get("seed", 17))
    device = resolve_torch_device(cfg.get("device", "auto"))
    dtype = resolve_training_dtype(cfg.get("precision", "auto"), device=device)
    geometry_domain = cfg.get("geometry_domain")
    validation_fraction = float(cfg.get("validation_fraction", 0.15))
    training_samples, validation_samples = deterministic_split(
        samples,
        validation_fraction,
        seed,
    )
    signature = _port_signature(cfg, cache_key)
    checkpoint_path = Path(checkpoint_path)

    payload = None
    if resume and checkpoint_path.is_file():
        candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if candidate.get("signature") == signature and candidate.get("cache_key") == cache_key:
            payload = candidate

    if payload is None:
        normalizer = HybridNormalizer.fit(training_samples)
        baseline_segments = {int(sample.baseline_segments) for sample in training_samples}
        if len(baseline_segments) != 1:
            raise ValueError("all port samples must use one baseline resolution")
        baseline_segments = baseline_segments.pop()
        material_domain = _sample_tensor_ranges(training_samples)
        model = _build_port_model(model_cfg, dtype=dtype, device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1e-6)),
        )
        epoch = 1
        schedule = None
        next_batch = 0
        epoch_total = 0.0
        epoch_seen = 0
        best_state = None
        best_score = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        rng = np.random.default_rng(seed)
    else:
        normalizer = HybridNormalizer.from_dict(payload["normalizer"])
        baseline_segments = int(payload["baseline_segments"])
        material_domain = payload["material_domain"]
        saved_dtype = str(payload.get("model_dtype", "float64"))
        dtype = torch.float32 if saved_dtype == "float32" else torch.float64
        model = _build_port_model(payload["model_config"], dtype=dtype, device=device)
        model.load_state_dict(payload["model_state"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1e-6)),
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        epoch = int(payload["epoch"])
        schedule = payload.get("schedule")
        next_batch = int(payload.get("next_batch", 0))
        epoch_total = float(payload.get("epoch_total", 0.0))
        epoch_seen = int(payload.get("epoch_seen", 0))
        best_state = payload.get("best_state")
        best_score = float(payload.get("best_score", float("inf")))
        best_epoch = int(payload.get("best_epoch", 0))
        stale = int(payload.get("stale", 0))
        history = list(payload.get("history", []))
        rng = np.random.default_rng(seed)
        rng.bit_generator.state = payload["rng_state"]
        if payload.get("status") == "completed":
            selected = best_state if best_state is not None else payload["model_state"]
            model.load_state_dict(selected)
            model.eval()
            return TensorHybridNeuralResidualArtifact(
                model,
                normalizer,
                baseline_segments=baseline_segments,
                material_domain=material_domain,
                geometry_domain=geometry_domain,
                device=device,
            ), history

    training_records = tuple(_record_from_sample(normalizer, sample) for sample in training_samples)
    validation_records = tuple(_record_from_sample(normalizer, sample) for sample in validation_samples)
    training_buckets = _topology_buckets(training_records)
    validation_buckets = _topology_buckets(validation_records)

    def save(status):
        _save_port_checkpoint(
            checkpoint_path,
            signature=signature,
            cache_key=cache_key,
            status=status,
            model=model,
            optimizer=optimizer,
            normalizer=normalizer,
            baseline_segments=baseline_segments,
            material_domain=material_domain,
            geometry_domain=geometry_domain,
            epoch=epoch,
            schedule=schedule,
            next_batch=next_batch,
            epoch_total=epoch_total,
            epoch_seen=epoch_seen,
            best_state=best_state,
            best_score=best_score,
            best_epoch=best_epoch,
            stale=stale,
            history=history,
            rng=rng,
        )

    while epoch <= epochs:
        model.train()
        if schedule is None:
            schedule = _schedule_from_buckets(training_buckets, batch_size, rng)
            next_batch = 0
            epoch_total = 0.0
            epoch_seen = 0
        for batch_number in range(next_batch, len(schedule)):
            try:
                control.checkpoint(
                    phase="port_training",
                    epoch=epoch,
                    epochs=epochs,
                    batch=batch_number + 1,
                    batches=len(schedule),
                )
            except TrainingStopRequested:
                save("stopped")
                raise
            indices = np.asarray(schedule[batch_number], dtype=int)
            batch = _stack_records(
                training_records,
                indices,
                dtype=dtype,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model,
                normalizer,
                batch,
                channel_loss_weight=channel_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimizer_cfg.get("gradient_clip_norm", 10.0)),
            )
            optimizer.step()
            count = len(indices)
            epoch_total += float(loss.detach().cpu()) * count
            epoch_seen += count
            next_batch = batch_number + 1
            if next_batch % checkpoint_every_batches == 0:
                save("running")

        train_loss = epoch_total / max(epoch_seen, 1)
        model.eval()
        score_total = 0.0
        score_count = 0
        eval_records = validation_records if validation_records else training_records
        eval_buckets = validation_buckets if validation_records else training_buckets
        with torch.no_grad():
            for key in eval_buckets:
                order = np.asarray(eval_buckets[key], dtype=int)
                for start in range(0, len(order), batch_size):
                    indices = order[start : start + batch_size]
                    batch = _stack_records(eval_records, indices, dtype=dtype, device=device)
                    score = _batch_loss(
                        model,
                        normalizer,
                        batch,
                        channel_loss_weight=channel_loss_weight,
                    )
                    score_total += float(score.detach().cpu()) * len(indices)
                    score_count += len(indices)
        validation_loss = score_total / max(score_count, 1)
        if validation_loss < best_score - min_improvement:
            best_score = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model.state_dict())
            stale = 0
        else:
            stale += 1
        row = {
            "phase": "port_training",
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": float(train_loss),
            "validation_loss": float(validation_loss),
            "best_validation_loss": float(best_score),
            "best_epoch": int(best_epoch),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
        }
        history.append(row)
        control.emit(**row)
        epoch += 1
        schedule = None
        next_batch = 0
        epoch_total = 0.0
        epoch_seen = 0
        save("running")
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    artifact = TensorHybridNeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=baseline_segments,
        material_domain=material_domain,
        geometry_domain=geometry_domain,
        device=device,
    )
    save("completed")
    return artifact, history


def _save_spatial_checkpoint(
    path,
    *,
    signature,
    cache_key,
    port_fingerprint,
    status,
    model,
    optimizer,
    epoch,
    schedule,
    next_batch,
    epoch_total,
    epoch_seen,
    best_state,
    best_score,
    best_shape_score,
    best_epoch,
    stale,
    history,
    rng,
    normalization,
):
    _atomic_torch_save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "kind": "tensor_spatial",
            "signature": signature,
            "cache_key": cache_key,
            "port_fingerprint": port_fingerprint,
            "status": status,
            "model_config": {
                "hidden_dim": model.hidden_dim,
                "coil_pair_dim": model.coil_pair_dim,
                "cross_dim": model.cross_dim,
                "field_hidden_dim": model.field_hidden_dim,
                "factor_rank": model.factor_rank,
                "depth": model.depth,
            },
            "model_state": _cpu_state_dict(model.state_dict()),
            "optimizer_state": optimizer.state_dict(),
            "epoch": int(epoch),
            "schedule": schedule,
            "next_batch": int(next_batch),
            "epoch_total": float(epoch_total),
            "epoch_seen": int(epoch_seen),
            "best_state": None if best_state is None else _cpu_state_dict(best_state),
            "best_score": None if best_score is None else float(best_score),
            "best_shape_score": None if best_shape_score is None else float(best_shape_score),
            "best_epoch": int(best_epoch),
            "stale": int(stale),
            "history": list(history),
            "rng_state": deepcopy(rng.bit_generator.state),
            "normalization": dict(normalization),
        },
        path,
    )


def train_spatial_controlled(
    port_artifact,
    samples,
    *,
    config,
    cache_key: str,
    checkpoint_path,
    control: TrainingControl,
    resume: bool = True,
):
    samples = tuple(samples)
    if not samples:
        raise ValueError("spatial training requires teacher samples")
    if any(not isinstance(sample, TensorHybridSpatialTeacherSample) for sample in samples):
        raise TypeError("spatial training requires TensorHybridSpatialTeacherSample values")
    cfg = dict(config)
    model_cfg = dict(cfg.get("model", {}) or {})
    optimizer_cfg = dict(cfg.get("optimizer", {}) or {})
    normalization = dict(cfg.get("normalization", {}) or {})
    epochs = int(cfg.get("epochs", 120))
    batch_size = int(cfg.get("batch_size", 8))
    patience = int(cfg.get("patience", 20))
    validation_interval = max(1, int(cfg.get("validation_interval", 1)))
    min_improvement = float(cfg.get("min_improvement", 1e-5))
    checkpoint_every_batches = max(1, int(cfg.get("checkpoint_every_batches", 1)))
    seed = int(cfg.get("seed", 47))
    device = resolve_torch_device(cfg.get("device", "auto"))
    validation_fraction = float(cfg.get("validation_fraction", 0.15))
    end_to_end_validation = bool(cfg.get("end_to_end_validation", True))
    training_samples, validation_samples = deterministic_split(samples, validation_fraction, seed)
    port_artifact.model.to(device)
    port_artifact.device = device
    port_model = port_artifact.model
    port_model.eval()
    for parameter in port_model.parameters():
        parameter.requires_grad_(False)
    port_fp = tensor_port_fingerprint(port_artifact)
    signature = _spatial_signature(cfg, cache_key, port_fp)
    checkpoint_path = Path(checkpoint_path)

    payload = None
    if resume and checkpoint_path.is_file():
        candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if candidate.get("signature") == signature and candidate.get("port_fingerprint") == port_fp:
            payload = candidate

    dtype = next(port_model.parameters()).dtype
    if payload is None:
        model = HybridSpatialLossShapeNet(
            port_model.hidden_dim,
            port_model.coil_pair_dim,
            port_model.cross_dim,
            field_hidden_dim=int(model_cfg.get("field_hidden_dim", 64)),
            factor_rank=int(model_cfg.get("factor_rank", 4)),
            depth=int(model_cfg.get("depth", 2)),
        ).to(device=device, dtype=dtype)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1e-6)),
        )
        epoch = 1
        schedule = None
        next_batch = 0
        epoch_total = 0.0
        epoch_seen = 0
        best_state = None
        best_score = None
        best_shape_score = None
        best_epoch = 0
        stale = 0
        history = []
        rng = np.random.default_rng(seed)
    else:
        model = HybridSpatialLossShapeNet(**payload["model_config"]).to(device=device, dtype=dtype)
        model.load_state_dict(payload["model_state"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1e-6)),
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        epoch = int(payload["epoch"])
        schedule = payload.get("schedule")
        next_batch = int(payload.get("next_batch", 0))
        epoch_total = float(payload.get("epoch_total", 0.0))
        epoch_seen = int(payload.get("epoch_seen", 0))
        best_state = payload.get("best_state")
        best_score = payload.get("best_score")
        best_shape_score = payload.get("best_shape_score")
        best_epoch = int(payload.get("best_epoch", 0))
        stale = int(payload.get("stale", 0))
        history = list(payload.get("history", []))
        rng = np.random.default_rng(seed)
        rng.bit_generator.state = payload["rng_state"]
        if payload.get("status") == "completed":
            selected = best_state if best_state is not None else payload["model_state"]
            model.load_state_dict(selected)
            model.eval()
            return TensorHybridSpatialLossArtifact(
                port_artifact,
                model,
                **normalization,
                device=device,
            ), history

    def make_buckets(values):
        buckets = defaultdict(list)
        for index, sample in enumerate(values):
            buckets[_sample_signature(sample)].append(index)
        return buckets

    training_buckets = make_buckets(training_samples)
    validation_buckets = make_buckets(validation_samples)

    def save(status):
        _save_spatial_checkpoint(
            checkpoint_path,
            signature=signature,
            cache_key=cache_key,
            port_fingerprint=port_fp,
            status=status,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            schedule=schedule,
            next_batch=next_batch,
            epoch_total=epoch_total,
            epoch_seen=epoch_seen,
            best_state=best_state,
            best_score=best_score,
            best_shape_score=best_shape_score,
            best_epoch=best_epoch,
            stale=stale,
            history=history,
            rng=rng,
            normalization=normalization,
        )

    while epoch <= epochs:
        model.train()
        if schedule is None:
            schedule = _schedule_from_buckets(training_buckets, batch_size, rng)
            next_batch = 0
            epoch_total = 0.0
            epoch_seen = 0
        for batch_number in range(next_batch, len(schedule)):
            try:
                control.checkpoint(
                    phase="spatial_training",
                    epoch=epoch,
                    epochs=epochs,
                    batch=batch_number + 1,
                    batches=len(schedule),
                )
            except TrainingStopRequested:
                save("stopped")
                raise
            batch = tuple(training_samples[index] for index in schedule[batch_number])
            optimizer.zero_grad(set_to_none=True)
            loss = _batched_spatial_shape_loss(
                model,
                port_artifact,
                batch,
                device=device,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimizer_cfg.get("gradient_clip_norm", 10.0)),
            )
            optimizer.step()
            epoch_total += float(loss.detach().cpu()) * len(batch)
            epoch_seen += len(batch)
            next_batch = batch_number + 1
            if next_batch % checkpoint_every_batches == 0:
                save("running")

        train_loss = epoch_total / max(epoch_seen, 1)
        shape_score = None
        score = None
        if validation_samples and (epoch % validation_interval == 0 or epoch == epochs):
            model.eval()
            total = 0.0
            seen = 0
            with torch.no_grad():
                for key in validation_buckets:
                    indices = validation_buckets[key]
                    for start in range(0, len(indices), batch_size):
                        batch = tuple(
                            validation_samples[index]
                            for index in indices[start : start + batch_size]
                        )
                        value = _batched_spatial_shape_loss(
                            model,
                            port_artifact,
                            batch,
                            device=device,
                        )
                        total += float(value.detach().cpu()) * len(batch)
                        seen += len(batch)
            shape_score = total / max(seen, 1)
            if end_to_end_validation:
                probe = TensorHybridSpatialLossArtifact(
                    port_artifact,
                    model,
                    **normalization,
                    device=device,
                )
                score = float(
                    np.mean(
                        [_sample_end_to_end_error(probe, sample) for sample in validation_samples]
                    )
                )
            else:
                score = shape_score
            if best_score is None or score < float(best_score) - min_improvement:
                best_score = score
                best_shape_score = shape_score
                best_epoch = epoch
                best_state = _cpu_state_dict(model.state_dict())
                stale = 0
            else:
                stale += 1
        elif not validation_samples:
            best_epoch = epoch
            best_state = _cpu_state_dict(model.state_dict())

        row = {
            "phase": "spatial_training",
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": float(train_loss),
            "validation_loss": None if score is None else float(score),
            "validation_shape_loss": None if shape_score is None else float(shape_score),
            "best_validation_loss": None if best_score is None else float(best_score),
            "best_epoch": int(best_epoch),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
        }
        history.append(row)
        control.emit(**row)
        epoch += 1
        schedule = None
        next_batch = 0
        epoch_total = 0.0
        epoch_seen = 0
        save("running")
        if validation_samples and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    artifact = TensorHybridSpatialLossArtifact(
        port_artifact,
        model,
        **normalization,
        device=device,
    )
    save("completed")
    return artifact, history
