from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("generation-2 controlled training requires the 'neural' extra") from exc

from .device import resolve_torch_device
from .generation2_features import (
    GENERATION2_FEATURE_SCHEMA,
    encode_generation2_scene,
)
from .generation2_objectives import (
    GENERATION2_PORT_TRAINING_CONTRACT,
    generation2_port_losses,
)
from .generation2_port import (
    GENERATION2_PORT_MODEL_GENERATION,
    Generation2Normalizer,
    Generation2PortArtifact,
    Generation2PortNet,
    forward_generation2_port_batch,
    generation2_material_domain,
)
from .hybrid_neural import _dielectric_loss_gate, _reactance_gate
from .performance import resolve_training_dtype
from .training_control import TrainingControl, TrainingStopRequested
from .workflow_cache import canonical_json


GENERATION2_CHECKPOINT_SCHEMA = 1


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


def _zero_last_linear(module) -> None:
    linears = [layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
    if not linears:
        raise TypeError("expected a module containing at least one Linear layer")
    torch.nn.init.zeros_(linears[-1].weight)
    if linears[-1].bias is not None:
        torch.nn.init.zeros_(linears[-1].bias)


def _initialize_baseline_residual_heads(model: Generation2PortNet) -> None:
    """Make the fresh model reproduce analytic R/X before learning corrections."""
    torch.nn.init.zeros_(model.resistance_log_diag_head.weight)
    torch.nn.init.zeros_(model.resistance_log_diag_head.bias)
    _zero_last_linear(model.resistance_log_pair_head)
    torch.nn.init.zeros_(model.reactance_diag_head.weight)
    torch.nn.init.zeros_(model.reactance_diag_head.bias)
    _zero_last_linear(model.reactance_pair_head)


@dataclass(frozen=True)
class _PortRecord:
    topology: tuple[int, int]
    normalized: tuple[np.ndarray, ...]
    baseline_resistance: np.ndarray
    baseline_reactance: np.ndarray
    target_impedance: np.ndarray
    target_channels: np.ndarray
    dielectric_loss_gate: float
    reactance_gate: float


def _record(normalizer: Generation2Normalizer, sample) -> _PortRecord:
    encoded = encode_generation2_scene(sample.scene, sample.frequency_hz)
    return _PortRecord(
        (len(sample.scene.coils), len(sample.scene.packages)),
        tuple(np.asarray(value, dtype=float) for value in normalizer.normalize(encoded)),
        np.asarray(sample.baseline_resistance, dtype=float),
        np.asarray(sample.baseline_reactance, dtype=float),
        np.asarray(sample.target_impedance, dtype=complex),
        np.asarray(sample.target_dissipation_channels, dtype=complex),
        float(_dielectric_loss_gate(sample.scene, sample.frequency_hz)),
        float(_reactance_gate(sample.frequency_hz)),
    )


def _buckets(records):
    result = defaultdict(list)
    for index, record in enumerate(records):
        result[record.topology].append(index)
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


def _stack(records, indices, *, dtype, device):
    selected = [records[int(index)] for index in indices]
    if len({record.topology for record in selected}) != 1:
        raise ValueError("generation-2 Port batch contains mixed topologies")
    complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
    return {
        "normalized": tuple(
            torch.as_tensor(
                np.stack([record.normalized[position] for record in selected], axis=0),
                dtype=dtype,
                device=device,
            )
            for position in range(5)
        ),
        "baseline_resistance": torch.as_tensor(
            np.stack([record.baseline_resistance for record in selected]),
            dtype=dtype,
            device=device,
        ),
        "baseline_reactance": torch.as_tensor(
            np.stack([record.baseline_reactance for record in selected]),
            dtype=dtype,
            device=device,
        ),
        "target_impedance": torch.as_tensor(
            np.stack([record.target_impedance for record in selected]),
            dtype=complex_dtype,
            device=device,
        ),
        "target_channels": torch.as_tensor(
            np.stack([record.target_channels for record in selected]),
            dtype=complex_dtype,
            device=device,
        ),
        "dielectric_loss_gate": torch.as_tensor(
            [record.dielectric_loss_gate for record in selected],
            dtype=dtype,
            device=device,
        ),
        "reactance_gate": torch.as_tensor(
            [record.reactance_gate for record in selected],
            dtype=dtype,
            device=device,
        ),
    }


def _losses(model, normalizer, batch, config):
    resistance, reactance, channels = forward_generation2_port_batch(
        model,
        *batch["normalized"],
        batch["baseline_resistance"],
        batch["baseline_reactance"],
        reactance_scale=normalizer.reactance_scale,
        dielectric_loss_gate=batch["dielectric_loss_gate"],
        reactance_gate=batch["reactance_gate"],
    )
    return generation2_port_losses(
        resistance,
        reactance,
        channels,
        batch["target_impedance"],
        batch["target_channels"],
        resistance_weight=float(config.get("resistance_weight", 1.0)),
        reactance_weight=float(config.get("reactance_weight", 1.0)),
        channel_weight=float(config.get("channel_loss_weight", 2.0)),
    )


def _evaluate(model, normalizer, records, buckets, *, batch_size, dtype, device, config):
    totals = {
        "loss": 0.0,
        "resistance_loss": 0.0,
        "reactance_loss": 0.0,
        "channel_loss": 0.0,
    }
    seen = 0
    model.eval()
    with torch.no_grad():
        for key in buckets:
            indices = buckets[key]
            for start in range(0, len(indices), int(batch_size)):
                current = indices[start : start + int(batch_size)]
                batch = _stack(records, current, dtype=dtype, device=device)
                values = _losses(model, normalizer, batch, config).detached()
                count = len(current)
                for name in totals:
                    totals[name] += values[name] * count
                seen += count
    return {name: value / max(seen, 1) for name, value in totals.items()}


def _signature(config, *, cache_key: str, partition_fingerprint: str) -> str:
    cfg = dict(config)
    payload = {
        "checkpoint_schema": GENERATION2_CHECKPOINT_SCHEMA,
        "model_generation": GENERATION2_PORT_MODEL_GENERATION,
        "feature_schema": GENERATION2_FEATURE_SCHEMA,
        "training_contract": GENERATION2_PORT_TRAINING_CONTRACT,
        "cache_key": str(cache_key),
        "partition_fingerprint": str(partition_fingerprint),
        "model": cfg.get("model", {}),
        "optimizer": cfg.get("optimizer", {}),
        "batch_size": int(cfg.get("batch_size", 32)),
        "precision": str(cfg.get("precision", "auto")),
        "resistance_weight": float(cfg.get("resistance_weight", 1.0)),
        "reactance_weight": float(cfg.get("reactance_weight", 1.0)),
        "channel_loss_weight": float(cfg.get("channel_loss_weight", 2.0)),
        "geometry_domain": cfg.get("geometry_domain"),
    }
    return _hash(payload)


def _model_config(model: Generation2PortNet):
    return {
        "coil_dim": model.coil_dim,
        "coil_pair_dim": model.coil_pair_dim,
        "package_dim": model.package_dim,
        "cross_dim": model.cross_dim,
        "package_pair_dim": model.package_pair_dim,
        "hidden_dim": model.hidden_dim,
        "factor_rank": model.factor_rank,
        "depth": model.depth,
        "interaction_rounds": model.interaction_rounds,
        "resistance_log_limit": model.resistance_log_limit,
    }


def _build_model(config, *, dtype, device):
    model_cfg = dict(config.get("model", {}) or {})
    model = Generation2PortNet(
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        factor_rank=int(model_cfg.get("factor_rank", 4)),
        depth=int(model_cfg.get("depth", 2)),
        interaction_rounds=int(model_cfg.get("interaction_rounds", 3)),
        resistance_log_limit=float(model_cfg.get("resistance_log_limit", 4.0)),
    ).to(device=device, dtype=dtype)
    _initialize_baseline_residual_heads(model)
    return model


def train_generation2_port_controlled(
    training_samples,
    validation_samples,
    *,
    domain_samples=None,
    config,
    cache_key: str,
    partition_fingerprint: str,
    checkpoint_path,
    control: TrainingControl,
    resume: bool = True,
):
    """Train Gen2 Port with explicit, shared workflow subsets and decomposed metrics."""
    training_samples = tuple(training_samples)
    validation_samples = tuple(validation_samples)
    domain_samples = tuple(
        training_samples + validation_samples
        if domain_samples is None
        else domain_samples
    )
    if not training_samples or not validation_samples:
        raise ValueError("generation-2 Port training requires train and validation samples")

    cfg = dict(config)
    optimizer_cfg = dict(cfg.get("optimizer", {}) or {})
    epochs = int(cfg.get("epochs", 200))
    batch_size = int(cfg.get("batch_size", 32))
    patience = int(cfg.get("patience", 20))
    min_improvement = float(cfg.get("min_improvement", 1e-5))
    checkpoint_every_batches = max(1, int(cfg.get("checkpoint_every_batches", 1)))
    seed = int(cfg.get("seed", 17))
    device = resolve_torch_device(cfg.get("device", "auto"))
    dtype = resolve_training_dtype(cfg.get("precision", "auto"), device=device)
    geometry_domain = cfg.get("geometry_domain")
    signature = _signature(
        cfg,
        cache_key=cache_key,
        partition_fingerprint=partition_fingerprint,
    )
    checkpoint_path = Path(checkpoint_path)

    payload = None
    if resume and checkpoint_path.is_file():
        candidate = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if candidate.get("signature") == signature:
            payload = candidate

    if payload is None:
        normalizer = Generation2Normalizer.fit(training_samples)
        baseline_segments = {int(sample.baseline_segments) for sample in domain_samples}
        if len(baseline_segments) != 1:
            raise ValueError("all generation-2 Port samples must use one baseline resolution")
        baseline_segments = baseline_segments.pop()
        material_domain = generation2_material_domain(domain_samples)
        model = _build_model(cfg, dtype=dtype, device=device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1e-5)),
        )
        epoch = 1
        schedule = None
        next_batch = 0
        epoch_totals = None
        epoch_seen = 0
        best_state = None
        best_score = float("inf")
        best_epoch = 0
        stale = 0
        history = []
        rng = np.random.default_rng(seed)
    else:
        normalizer = Generation2Normalizer.from_dict(payload["normalizer"])
        baseline_segments = int(payload["baseline_segments"])
        material_domain = payload["material_domain"]
        saved_dtype = str(payload.get("model_dtype", "float64"))
        dtype = torch.float32 if saved_dtype == "float32" else torch.float64
        model = Generation2PortNet(**payload["model_config"]).to(
            device=device,
            dtype=dtype,
        )
        model.load_state_dict(payload["model_state"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1e-5)),
        )
        optimizer.load_state_dict(payload["optimizer_state"])
        epoch = int(payload["epoch"])
        schedule = payload.get("schedule")
        next_batch = int(payload.get("next_batch", 0))
        epoch_totals = payload.get("epoch_totals")
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
            return Generation2PortArtifact(
                model,
                normalizer,
                baseline_segments=baseline_segments,
                material_domain=material_domain,
                geometry_domain=geometry_domain,
                partition_fingerprint=partition_fingerprint,
                device=device,
            ), history

    training_records = tuple(_record(normalizer, sample) for sample in training_samples)
    validation_records = tuple(_record(normalizer, sample) for sample in validation_samples)
    training_buckets = _buckets(training_records)
    validation_buckets = _buckets(validation_records)

    def save(status):
        _atomic_save(
            {
                "schema": GENERATION2_CHECKPOINT_SCHEMA,
                "kind": "generation2_port",
                "signature": signature,
                "status": str(status),
                "model_generation": GENERATION2_PORT_MODEL_GENERATION,
                "feature_schema": GENERATION2_FEATURE_SCHEMA,
                "training_contract": GENERATION2_PORT_TRAINING_CONTRACT,
                "partition_fingerprint": str(partition_fingerprint),
                "model_config": _model_config(model),
                "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
                "model_state": _cpu_state_dict(model.state_dict()),
                "optimizer_state": optimizer.state_dict(),
                "normalizer": normalizer.to_dict(),
                "baseline_segments": int(baseline_segments),
                "material_domain": material_domain,
                "epoch": int(epoch),
                "schedule": schedule,
                "next_batch": int(next_batch),
                "epoch_totals": epoch_totals,
                "epoch_seen": int(epoch_seen),
                "best_state": None if best_state is None else _cpu_state_dict(best_state),
                "best_score": float(best_score),
                "best_epoch": int(best_epoch),
                "stale": int(stale),
                "history": list(history),
                "rng_state": deepcopy(rng.bit_generator.state),
            },
            checkpoint_path,
        )

    metric_names = ("loss", "resistance_loss", "reactance_loss", "channel_loss")
    while epoch <= epochs:
        model.train()
        if schedule is None:
            schedule = _schedule(training_buckets, batch_size, rng)
            next_batch = 0
            epoch_totals = {name: 0.0 for name in metric_names}
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
            indices = schedule[batch_number]
            batch = _stack(training_records, indices, dtype=dtype, device=device)
            optimizer.zero_grad(set_to_none=True)
            losses = _losses(model, normalizer, batch, cfg)
            losses.composite.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(optimizer_cfg.get("gradient_clip_norm", 10.0)),
            )
            optimizer.step()
            values = losses.detached()
            count = len(indices)
            for name in metric_names:
                epoch_totals[name] += values[name] * count
            epoch_seen += count
            next_batch = batch_number + 1
            if next_batch % checkpoint_every_batches == 0:
                save("running")

        train_metrics = {
            name: epoch_totals[name] / max(epoch_seen, 1)
            for name in metric_names
        }
        validation_metrics = _evaluate(
            model,
            normalizer,
            validation_records,
            validation_buckets,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            config=cfg,
        )
        validation_loss = float(validation_metrics["loss"])
        if validation_loss < best_score - min_improvement:
            best_score = validation_loss
            best_epoch = epoch
            best_state = _cpu_state_dict(model.state_dict())
            stale = 0
        else:
            stale += 1

        row = {
            "phase": "port_training",
            "model_generation": GENERATION2_PORT_MODEL_GENERATION,
            "epoch": int(epoch),
            "epochs": int(epochs),
            "train_loss": float(train_metrics["loss"]),
            "train_resistance_loss": float(train_metrics["resistance_loss"]),
            "train_reactance_loss": float(train_metrics["reactance_loss"]),
            "train_channel_loss": float(train_metrics["channel_loss"]),
            "validation_loss": validation_loss,
            "validation_resistance_loss": float(validation_metrics["resistance_loss"]),
            "validation_reactance_loss": float(validation_metrics["reactance_loss"]),
            "validation_channel_loss": float(validation_metrics["channel_loss"]),
            "best_validation_loss": float(best_score),
            "best_epoch": int(best_epoch),
            "device": device,
            "dtype": str(dtype).replace("torch.", ""),
            "port_training_contract": GENERATION2_PORT_TRAINING_CONTRACT,
        }
        history.append(row)
        control.emit(**row)

        epoch += 1
        schedule = None
        next_batch = 0
        epoch_totals = None
        epoch_seen = 0
        save("running")
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    artifact = Generation2PortArtifact(
        model,
        normalizer,
        baseline_segments=baseline_segments,
        material_domain=material_domain,
        geometry_domain=geometry_domain,
        partition_fingerprint=partition_fingerprint,
        device=device,
    )
    save("completed")
    return artifact, history


__all__ = [
    "GENERATION2_CHECKPOINT_SCHEMA",
    "train_generation2_port_controlled",
]
