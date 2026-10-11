from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("generation-2 Spatial training requires the 'neural' extra") from exc

from .device import resolve_torch_device
from .generation2_objectives import GENERATION2_SPATIAL_TRAINING_CONTRACT
from .generation2_port import Generation2PortArtifact, generation2_port_fingerprint
from .generation2_spatial import (
    GENERATION2_SPATIAL_MODEL_GENERATION,
    Generation2SpatialArtifact,
    Generation2SpatialNet,
    generation2_batched_spatial_shape_loss,
    generation2_spatial_end_to_end_error,
)
from .spatial_consistent_training import resolve_spatial_normalization
from .training_control import TrainingControl, TrainingStopRequested
from .workflow_cache import canonical_json


GENERATION2_SPATIAL_CHECKPOINT_SCHEMA = 1


def _hash(payload) -> str:
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _cpu_state_dict(state):
    return {
        key: value.detach().cpu().clone()
        for key, value in state.items()
    }


def _atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _signature(config, *, cache_key: str, partition_fingerprint: str, port_fingerprint: str):
    cfg = dict(config)
    return _hash(
        {
            "schema": GENERATION2_SPATIAL_CHECKPOINT_SCHEMA,
            "model_generation": GENERATION2_SPATIAL_MODEL_GENERATION,
            "training_contract": GENERATION2_SPATIAL_TRAINING_CONTRACT,
            "cache_key": str(cache_key),
            "partition_fingerprint": str(partition_fingerprint),
            "port_fingerprint": str(port_fingerprint),
            "model": cfg.get("model", {}),
            "optimizer": cfg.get("optimizer", {}),
            "normalization": cfg.get("normalization", {}),
            "batch_size": int(cfg.get("batch_size", 8)),
        }
    )


def _sample_signature(sample):
    return len(sample.scene.coils), len(sample.scene.packages)


def _buckets(samples):
    result = defaultdict(list)
    for index, sample in enumerate(samples):
        result[_sample_signature(sample)].append(index)
    return result


def _schedule(buckets, batch_size: int, rng):
    keys = list(buckets)
    rng.shuffle(keys)
    batches = []
    for key in keys:
        order = rng.permutation(np.asarray(buckets[key], dtype=int))
        for start in range(0, len(order), int(batch_size)):
            batches.append([int(value) for value in order[start : start + int(batch_size)]])
    rng.shuffle(batches)
    return batches


def _evaluate_shape(model, port_artifact, samples, buckets, *, batch_size, device, normalization):
    total = 0.0
    seen = 0
    model.eval()
    with torch.no_grad():
        for key in buckets:
            indices = buckets[key]
            for start in range(0, len(indices), int(batch_size)):
                current = indices[start : start + int(batch_size)]
                batch = tuple(samples[index] for index in current)
                value = generation2_batched_spatial_shape_loss(
                    model,
                    port_artifact,
                    batch,
                    device=device,
                    normalization=normalization,
                )
                total += float(value.detach().cpu()) * len(batch)
                seen += len(batch)
    return total / max(seen, 1)


def _model_config(model: Generation2SpatialNet):
    return {
        "port_hidden_dim": model.port_hidden_dim,
        "context_hidden_dim": model.hidden_dim,
        "context_rounds": model.context_rounds,
        "context_depth": model.context_depth,
        "field_hidden_dim": model.field_hidden_dim,
        "factor_rank": model.factor_rank,
        "depth": model.depth,
    }


def _build_model(port_artifact, config, *, device):
    cfg = dict(config.get("model", {}) or {})
    dtype = next(port_artifact.model.parameters()).dtype
    return Generation2SpatialNet(
        port_artifact.model.hidden_dim,
        context_hidden_dim=int(cfg.get("context_hidden_dim", port_artifact.model.hidden_dim)),
        context_rounds=int(cfg.get("context_rounds", 1)),
        context_depth=int(cfg.get("context_depth", 1)),
        field_hidden_dim=int(cfg.get("field_hidden_dim", 128)),
        factor_rank=int(cfg.get("factor_rank", 4)),
        depth=int(cfg.get("depth", 3)),
    ).to(device=device, dtype=dtype)


def train_generation2_spatial_controlled(
    port_artifact: Generation2PortArtifact,
    training_samples,
    validation_samples,
    *,
    config,
    cache_key: str,
    partition_fingerprint: str,
    checkpoint_path,
    control: TrainingControl,
    resume: bool = True,
):
    """Train canonical Spatial shape on explicit shared workflow subsets."""
    if not isinstance(port_artifact, Generation2PortArtifact):
        raise TypeError("generation-2 Spatial training requires Generation2PortArtifact")
    training_samples = tuple(training_samples)
    validation_samples = tuple(validation_samples)
    if not training_samples or not validation_samples:
        raise ValueError("generation-2 Spatial training requires train and validation samples")

    cfg = dict(config)
    optimizer_cfg = dict(cfg.get("optimizer", {}) or {})
    normalization = resolve_spatial_normalization(cfg.get("normalization", {}) or {})
    epochs = int(cfg.get("epochs", 140))
    batch_size = int(cfg.get("batch_size", 8))
    patience = int(cfg.get("patience", 20))
    validation_interval = max(1, int(cfg.get("validation_interval", 1)))
    end_to_end_interval = max(
        1,
        int(cfg.get("end_to_end_validation_interval", validation_interval)),
    )
    min_improvement = float(cfg.get("min_improvement", 1e-5))
    checkpoint_every_batches = max(1, int(cfg.get("checkpoint_every_batches", 1)))
    seed = int(cfg.get("seed", 47))
    device = resolve_torch_device(cfg.get("device", "auto"))
    port_artifact.model.to(device)
    port_artifact.device = device
    port_artifact.model.eval()
    for parameter in port_artifact.model.parameters():
        parameter.requires_grad_(False)

    port_fingerprint = generation2_port_fingerprint(port_artifact)
    signature = _signature(
        cfg,
        cache_key=cache_key,
        partition_fingerprint=partition_fingerprint,
        port_fingerprint=port_fingerprint,
    )
    checkpoint_path = Path(checkpoint_path)
    payload = None
    if resume and checkpoint_path.is_file():
        candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if candidate.get("signature") == signature:
            payload = candidate

    if payload is None:
        model = _build_model(port_artifact, cfg, device=device)
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
        best_shape_score = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        rng = np.random.default_rng(seed)
    else:
        dtype = next(port_artifact.model.parameters()).dtype
        model = Generation2SpatialNet(**payload["model_config"]).to(
            device=device,
            dtype=dtype,
        )
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
        best_shape_score = float(payload.get("best_shape_score", float("inf")))
        best_epoch = int(payload.get("best_epoch", 0))
        stale = int(payload.get("stale", 0))
        history = list(payload.get("history", []))
        rng = np.random.default_rng(seed)
        rng.bit_generator.state = payload["rng_state"]
        if payload.get("status") == "completed":
            selected = best_state if best_state is not None else payload["model_state"]
            model.load_state_dict(selected)
            model.eval()
            return Generation2SpatialArtifact(
                port_artifact,
                model,
                normalization=normalization,
                device=device,
            ), history

    training_buckets = _buckets(training_samples)
    validation_buckets = _buckets(validation_samples)

    def save(status):
        _atomic_save(
            {
                "schema": GENERATION2_SPATIAL_CHECKPOINT_SCHEMA,
                "kind": "generation2_spatial",
                "signature": signature,
                "status": str(status),
                "model_generation": GENERATION2_SPATIAL_MODEL_GENERATION,
                "training_contract": GENERATION2_SPATIAL_TRAINING_CONTRACT,
                "partition_fingerprint": str(partition_fingerprint),
                "port_fingerprint": port_fingerprint,
                "model_config": _model_config(model),
                "model_state": _cpu_state_dict(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "normalization": dict(normalization),
                "epoch": int(epoch),
                "schedule": schedule,
                "next_batch": int(next_batch),
                "epoch_total": float(epoch_total),
                "epoch_seen": int(epoch_seen),
                "best_state": None if best_state is None else _cpu_state_dict(best_state),
                "best_shape_score": float(best_shape_score),
                "best_epoch": int(best_epoch),
                "stale": int(stale),
                "history": list(history),
                "rng_state": deepcopy(rng.bit_generator.state),
            },
            checkpoint_path,
        )

    while epoch <= epochs:
        model.train()
        if schedule is None:
            schedule = _schedule(training_buckets, batch_size, rng)
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
            indices = schedule[batch_number]
            batch = tuple(training_samples[index] for index in indices)
            optimizer.zero_grad(set_to_none=True)
            loss = generation2_batched_spatial_shape_loss(
                model,
                port_artifact,
                batch,
                device=device,
                normalization=normalization,
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

        train_shape = epoch_total / max(epoch_seen, 1)
        validation_shape = None
        validation_end_to_end = None
        if epoch % validation_interval == 0 or epoch == epochs:
            validation_shape = _evaluate_shape(
                model,
                port_artifact,
                validation_samples,
                validation_buckets,
                batch_size=batch_size,
                device=device,
                normalization=normalization,
            )
            if validation_shape < best_shape_score - min_improvement:
                best_shape_score = float(validation_shape)
                best_epoch = int(epoch)
                best_state = _cpu_state_dict(model.state_dict())
                stale = 0
            else:
                stale += 1

        if epoch % end_to_end_interval == 0 or epoch == epochs:
            probe = Generation2SpatialArtifact(
                port_artifact,
                model,
                normalization=normalization,
                device=device,
            )
            validation_end_to_end = float(
                np.mean(
                    [
                        generation2_spatial_end_to_end_error(probe, sample)
                        for sample in validation_samples
                    ]
                )
            )

        row = {
            "phase": "spatial_training",
            "model_generation": GENERATION2_SPATIAL_MODEL_GENERATION,
            "epoch": int(epoch),
            "epochs": int(epochs),
            "train_loss": float(train_shape),
            "train_shape_loss": float(train_shape),
            "validation_loss": (
                None if validation_shape is None else float(validation_shape)
            ),
            "validation_shape_loss": (
                None if validation_shape is None else float(validation_shape)
            ),
            "validation_end_to_end_loss": (
                None if validation_end_to_end is None else float(validation_end_to_end)
            ),
            "best_validation_loss": (
                None if not np.isfinite(best_shape_score) else float(best_shape_score)
            ),
            "best_validation_shape_loss": (
                None if not np.isfinite(best_shape_score) else float(best_shape_score)
            ),
            "best_epoch": int(best_epoch),
            "device": device,
            "dtype": str(next(port_artifact.model.parameters()).dtype).replace("torch.", ""),
            "spatial_training_contract": GENERATION2_SPATIAL_TRAINING_CONTRACT,
        }
        history.append(row)
        control.emit(**row)
        epoch += 1
        schedule = None
        next_batch = 0
        epoch_total = 0.0
        epoch_seen = 0
        save("running")
        if validation_shape is not None and stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    artifact = Generation2SpatialArtifact(
        port_artifact,
        model,
        normalization=normalization,
        device=device,
    )
    save("completed")
    return artifact, history


__all__ = [
    "GENERATION2_SPATIAL_CHECKPOINT_SCHEMA",
    "train_generation2_spatial_controlled",
]
