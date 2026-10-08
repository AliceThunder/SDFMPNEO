from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("controlled vNext training requires the 'neural' extra") from exc

from ._controlled_training_core import (
    _cpu_state_dict,
    _hash,
    _save_spatial_checkpoint,
    _schedule_from_buckets,
    _spatial_signature,
    deterministic_split,
)
from .device import resolve_torch_device
from .hybrid_spatial_neural import HybridSpatialLossShapeNet
from .spatial_consistent_training import (
    SPATIAL_TRAINING_CONTRACT,
    clear_spatial_consistency_caches,
    consistent_batched_spatial_shape_loss,
    resolve_spatial_normalization,
)
from .spatial_performance import _sample_signature
from .tensor_spatial_neural import (
    TensorHybridSpatialLossArtifact,
    _sample_end_to_end_error,
    tensor_port_fingerprint,
)
from .tensor_spatial_training_data import TensorHybridSpatialTeacherSample
from .training_control import TrainingControl, TrainingStopRequested


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
    """Controlled tensor spatial training with inference-identical normalization."""
    samples = tuple(samples)
    if not samples:
        raise ValueError("spatial training requires teacher samples")
    if any(not isinstance(sample, TensorHybridSpatialTeacherSample) for sample in samples):
        raise TypeError("spatial training requires TensorHybridSpatialTeacherSample values")

    cfg = dict(config)
    model_cfg = dict(cfg.get("model", {}) or {})
    optimizer_cfg = dict(cfg.get("optimizer", {}) or {})
    normalization = resolve_spatial_normalization(cfg.get("normalization", {}) or {})
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
    training_samples, validation_samples = deterministic_split(
        samples,
        validation_fraction,
        seed,
    )

    port_artifact.model.to(device)
    port_artifact.device = device
    port_model = port_artifact.model
    port_model.eval()
    for parameter in port_model.parameters():
        parameter.requires_grad_(False)

    clear_spatial_consistency_caches()
    port_fp = tensor_port_fingerprint(port_artifact)
    legacy_signature = _spatial_signature(cfg, cache_key, port_fp)
    signature = _hash(
        {
            "legacy_signature": legacy_signature,
            "spatial_training_contract": SPATIAL_TRAINING_CONTRACT,
        }
    )
    checkpoint_path = Path(checkpoint_path)

    payload = None
    if resume and checkpoint_path.is_file():
        candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            candidate.get("signature") == signature
            and candidate.get("port_fingerprint") == port_fp
        ):
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
        model = HybridSpatialLossShapeNet(**payload["model_config"]).to(
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
            loss = consistent_batched_spatial_shape_loss(
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

        train_loss = epoch_total / max(epoch_seen, 1)
        shape_score = None
        end_to_end_score = None
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
                        value = consistent_batched_spatial_shape_loss(
                            model,
                            port_artifact,
                            batch,
                            device=device,
                            normalization=normalization,
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
                end_to_end_score = float(
                    np.mean(
                        [
                            _sample_end_to_end_error(probe, sample)
                            for sample in validation_samples
                        ]
                    )
                )

            # Select the checkpoint using the quantity controlled by this
            # network. End-to-end error also contains the frozen port model's
            # error and is therefore diagnostic rather than an early-stop key.
            if best_score is None or shape_score < float(best_score) - min_improvement:
                best_score = shape_score
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
            "validation_loss": None if shape_score is None else float(shape_score),
            "validation_shape_loss": None if shape_score is None else float(shape_score),
            "validation_end_to_end_loss": (
                None if end_to_end_score is None else float(end_to_end_score)
            ),
            "best_validation_loss": None if best_score is None else float(best_score),
            "best_epoch": int(best_epoch),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "spatial_training_contract": SPATIAL_TRAINING_CONTRACT,
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
