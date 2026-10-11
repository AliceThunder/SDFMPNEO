from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .generation2_bundle import publish_generation2_bundle
from .generation2_objectives import numpy_relative_from_error_energy
from .generation2_spatial import (
    generation2_batched_spatial_shape_loss,
    generation2_spatial_end_to_end_error,
)
from .generation2_spatial_training import train_generation2_spatial_controlled
from .generation2_split import generation2_partition
from .generation2_training import train_generation2_port_controlled
from .training_control import (
    SessionPaths,
    TrainingControl,
    TrainingStopRequested,
    write_control_command,
)
from .workflow import (
    _new_session,
    _prepare_cache,
    _resolve,
    _runtime_config,
    _write_worker_snapshot,
)
from .workflow_cache import canonical_json


GENERATION2_WORKFLOW_CONTRACT = 1


def _training_dataset_key(cache_key: str, count: int, partition_fingerprint: str, config) -> str:
    payload = {
        "workflow_contract": GENERATION2_WORKFLOW_CONTRACT,
        "model_generation": 2,
        "teacher_cache_key": str(cache_key),
        "count": int(count),
        "partition_fingerprint": str(partition_fingerprint),
        "port_objective": {
            "resistance_weight": float(config["PORT_TRAINING"].get("resistance_weight", 1.0)),
            "reactance_weight": float(config["PORT_TRAINING"].get("reactance_weight", 1.0)),
            "channel_loss_weight": float(config["PORT_TRAINING"].get("channel_loss_weight", 2.0)),
        },
        "spatial_objective": "teacher_channel_canonical_shape_v1",
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _metric(row, name):
    value = row.get(name)
    return None if value is None else float(value)


def _port_history_summary(history, configured_epochs: int):
    rows = list(history)
    result = {
        "epochs_completed": len(rows),
        "configured_epochs": int(configured_epochs),
    }
    if not rows:
        return result
    last = rows[-1]
    best_epoch = int(last.get("best_epoch") or last.get("epoch") or len(rows))
    best = next(
        (row for row in rows if int(row.get("epoch", -1)) == best_epoch),
        last,
    )
    result.update(
        {
            "final_epoch": int(last.get("epoch") or len(rows)),
            "best_epoch": best_epoch,
            "stopped_early": int(last.get("epoch") or len(rows)) < int(configured_epochs),
            "final_train_loss": _metric(last, "train_loss"),
            "final_validation_loss": _metric(last, "validation_loss"),
            "best_validation_loss": _metric(last, "best_validation_loss"),
            "best_epoch_train_loss": _metric(best, "train_loss"),
            "best_epoch_validation_loss": _metric(best, "validation_loss"),
            "best_epoch_train_resistance_loss": _metric(best, "train_resistance_loss"),
            "best_epoch_validation_resistance_loss": _metric(best, "validation_resistance_loss"),
            "best_epoch_train_reactance_loss": _metric(best, "train_reactance_loss"),
            "best_epoch_validation_reactance_loss": _metric(best, "validation_reactance_loss"),
            "best_epoch_train_channel_loss": _metric(best, "train_channel_loss"),
            "best_epoch_validation_channel_loss": _metric(best, "validation_channel_loss"),
            "final_train_resistance_loss": _metric(last, "train_resistance_loss"),
            "final_validation_resistance_loss": _metric(last, "validation_resistance_loss"),
            "final_train_reactance_loss": _metric(last, "train_reactance_loss"),
            "final_validation_reactance_loss": _metric(last, "validation_reactance_loss"),
            "final_train_channel_loss": _metric(last, "train_channel_loss"),
            "final_validation_channel_loss": _metric(last, "validation_channel_loss"),
            "device": last.get("device"),
            "dtype": last.get("dtype"),
        }
    )
    return result


def _spatial_history_summary(history, configured_epochs: int):
    rows = list(history)
    result = {
        "epochs_completed": len(rows),
        "configured_epochs": int(configured_epochs),
    }
    if not rows:
        return result
    last = rows[-1]
    best_epoch = int(last.get("best_epoch") or last.get("epoch") or len(rows))
    best = next(
        (row for row in rows if int(row.get("epoch", -1)) == best_epoch),
        last,
    )
    end_rows = [row for row in rows if row.get("validation_end_to_end_loss") is not None]
    best_end = (
        min(end_rows, key=lambda row: float(row["validation_end_to_end_loss"]))
        if end_rows
        else None
    )
    result.update(
        {
            "final_epoch": int(last.get("epoch") or len(rows)),
            "best_epoch": best_epoch,
            "stopped_early": int(last.get("epoch") or len(rows)) < int(configured_epochs),
            "final_train_shape_loss": _metric(last, "train_shape_loss"),
            "final_validation_shape_loss": _metric(last, "validation_shape_loss"),
            "best_validation_shape_loss": _metric(last, "best_validation_shape_loss"),
            "best_epoch_train_shape_loss": _metric(best, "train_shape_loss"),
            "best_epoch_validation_shape_loss": _metric(best, "validation_shape_loss"),
            "final_validation_end_to_end_loss": _metric(last, "validation_end_to_end_loss"),
            "best_validation_end_to_end_loss": (
                None if best_end is None else _metric(best_end, "validation_end_to_end_loss")
            ),
            "best_end_to_end_epoch": (
                None if best_end is None else int(best_end.get("epoch") or 0)
            ),
            "device": last.get("device"),
            "dtype": last.get("dtype"),
        }
    )
    return result


def _relative_matrix(predicted, target) -> float:
    predicted = np.asarray(predicted)
    target = np.asarray(target)
    error = float(np.mean(np.abs(predicted - target) ** 2))
    energy = float(np.mean(np.abs(predicted) ** 2 + np.abs(target) ** 2))
    return numpy_relative_from_error_energy(error, energy)


def _evaluate_port_test(port, samples, config):
    values = []
    for sample in samples:
        prediction = port.predict_structured(sample.scene, sample.frequency_hz)
        target = np.asarray(sample.target_impedance, dtype=complex)
        resistance = _relative_matrix(prediction.impedance.real, target.real)
        reactance = _relative_matrix(prediction.impedance.imag, target.imag)
        channels = _relative_matrix(
            prediction.dissipation_channels,
            sample.target_dissipation_channels,
        )
        composite = (
            float(config.get("resistance_weight", 1.0)) * resistance
            + float(config.get("reactance_weight", 1.0)) * reactance
            + float(config.get("channel_loss_weight", 2.0)) * channels
        )
        values.append((composite, resistance, reactance, channels))
    array = np.asarray(values, dtype=float)
    return {
        "count": len(values),
        "loss": float(np.mean(array[:, 0])),
        "resistance_loss": float(np.mean(array[:, 1])),
        "reactance_loss": float(np.mean(array[:, 2])),
        "channel_loss": float(np.mean(array[:, 3])),
    }


def _spatial_buckets(samples):
    result = {}
    for index, sample in enumerate(samples):
        key = (len(sample.scene.coils), len(sample.scene.packages))
        result.setdefault(key, []).append(index)
    return result


def _evaluate_spatial_shape(spatial, samples, config):
    batch_size = int(config.get("batch_size", 8))
    buckets = _spatial_buckets(samples)
    total = 0.0
    seen = 0
    spatial.model.eval()
    with __import__("torch").no_grad():
        for indices in buckets.values():
            for start in range(0, len(indices), batch_size):
                current = indices[start : start + batch_size]
                batch = tuple(samples[index] for index in current)
                value = generation2_batched_spatial_shape_loss(
                    spatial.model,
                    spatial.port_artifact,
                    batch,
                    device=spatial.device,
                    normalization=spatial.normalization,
                )
                total += float(value.detach().cpu()) * len(batch)
                seen += len(batch)
    return total / max(seen, 1)


def _evaluate_spatial_test(spatial, samples, config):
    shape = _evaluate_spatial_shape(spatial, samples, config)
    end_to_end = float(
        np.mean(
            [generation2_spatial_end_to_end_error(spatial, sample) for sample in samples]
        )
    )
    return {
        "count": len(samples),
        "shape_loss": float(shape),
        "end_to_end_loss": end_to_end,
    }


def run_generation2_training_worker(config, session_dir) -> int:
    config = _runtime_config(config)
    root = Path(config["ROOT"])
    files = dict(config["FILES"])
    training_cfg = dict(config.get("TRAINING", {}) or {})
    if int(training_cfg.get("generation", 2)) != 2:
        raise ValueError("generation2 workflow requires TRAINING.generation=2")
    control_cfg = dict(config.get("CONTROL", {}) or {})
    session_dir = Path(session_dir).resolve(strict=False)
    session_dir.mkdir(parents=True, exist_ok=True)
    paths = SessionPaths.from_root(session_dir)
    if not paths.control.exists():
        write_control_command(paths.control, "run")

    with TrainingControl(
        session_dir,
        heartbeat_interval_s=float(control_cfg.get("heartbeat_interval_s", 0.5)),
        fsync_metrics=bool(control_cfg.get("fsync_metrics", False)),
    ) as control:
        try:
            cache, generated = _prepare_cache(config, control)
            count = int(config["DATA"]["count"])
            include_spatial = bool(config["TRUTH"].get("include_spatial", False))
            if include_spatial:
                port_samples = tuple(sample.port for sample in generated)
                spatial_samples = tuple(generated)
            else:
                port_samples = tuple(generated)
                spatial_samples = ()

            partition = generation2_partition(
                count,
                validation_fraction=float(training_cfg.get("validation_fraction", 0.10)),
                test_fraction=float(training_cfg.get("test_fraction", 0.10)),
                seed=int(training_cfg.get("split_seed", 2027)),
            )
            partition_fingerprint = partition.fingerprint()
            dataset_key = _training_dataset_key(
                cache.key,
                count,
                partition_fingerprint,
                config,
            )
            port_train = partition.subset(port_samples, "train")
            port_validation = partition.subset(port_samples, "validation")
            port_test = partition.subset(port_samples, "test")
            spatial_train = (
                partition.subset(spatial_samples, "train") if include_spatial else ()
            )
            spatial_validation = (
                partition.subset(spatial_samples, "validation") if include_spatial else ()
            )
            spatial_test = (
                partition.subset(spatial_samples, "test") if include_spatial else ()
            )
            control.emit(
                phase="starting",
                model_generation=2,
                dataset_key=dataset_key,
                partition_fingerprint=partition_fingerprint,
                train_samples=len(port_train),
                validation_samples=len(port_validation),
                test_samples=len(port_test),
                message="generation-2 immutable train/validation/test partition ready",
            )

            resume = bool(training_cfg.get("resume", True))
            checkpoint_root = _resolve(root, files["checkpoint_dir"])
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            port_checkpoint = checkpoint_root / str(
                files.get("port_checkpoint", "generation2_port.training.pt")
            )
            spatial_checkpoint = checkpoint_root / str(
                files.get("spatial_checkpoint", "generation2_spatial.training.pt")
            )

            control.checkpoint(
                phase="port_training",
                message="training generation-2 tensor FAST Port surrogate",
                dataset_key=dataset_key,
            )
            port, port_history = train_generation2_port_controlled(
                port_train,
                port_validation,
                domain_samples=port_samples,
                config=config["PORT_TRAINING"],
                cache_key=dataset_key,
                partition_fingerprint=partition_fingerprint,
                checkpoint_path=port_checkpoint,
                control=control,
                resume=resume,
            )
            port_artifact_path = _resolve(root, files["port_artifact"])
            port.save(port_artifact_path)
            port_summary = _port_history_summary(
                port_history,
                int(config["PORT_TRAINING"].get("epochs", 200)),
            )
            port_test_metrics = _evaluate_port_test(
                port,
                port_test,
                config["PORT_TRAINING"],
            )
            control.emit(
                phase="port_training",
                event="summary",
                port_artifact=str(port_artifact_path),
                **port_summary,
                test_metrics=port_test_metrics,
            )

            spatial = None
            spatial_history = []
            spatial_summary = {}
            spatial_test_metrics = None
            spatial_cfg = dict(config.get("SPATIAL_TRAINING", {}) or {})
            if bool(spatial_cfg.get("enabled", include_spatial)):
                if not include_spatial:
                    raise ValueError("SPATIAL_TRAINING.enabled requires TRUTH.include_spatial=true")
                control.checkpoint(
                    phase="spatial_training",
                    message="training generation-2 canonical continuous Spatial surrogate",
                )
                spatial, spatial_history = train_generation2_spatial_controlled(
                    port,
                    spatial_train,
                    spatial_validation,
                    config=spatial_cfg,
                    cache_key=dataset_key,
                    partition_fingerprint=partition_fingerprint,
                    checkpoint_path=spatial_checkpoint,
                    control=control,
                    resume=resume,
                )
                spatial_artifact_path = _resolve(root, files["spatial_artifact"])
                spatial.save(spatial_artifact_path)
                spatial_summary = _spatial_history_summary(
                    spatial_history,
                    int(spatial_cfg.get("epochs", 140)),
                )
                spatial_test_metrics = _evaluate_spatial_test(
                    spatial,
                    spatial_test,
                    spatial_cfg,
                )
                control.emit(
                    phase="spatial_training",
                    event="summary",
                    spatial_artifact=str(spatial_artifact_path),
                    **spatial_summary,
                    test_metrics=spatial_test_metrics,
                )

            control.checkpoint(
                phase="publishing",
                message="publishing generation-2 tensor bundle",
            )
            bundle_path = _resolve(root, files["bundle"])
            manifest = publish_generation2_bundle(
                bundle_path,
                port,
                spatial_artifact=spatial,
                metadata={
                    "teacher_cache_key": cache.key,
                    "training_dataset_key": dataset_key,
                    "teacher_samples": count,
                    "partition": {
                        "fingerprint": partition_fingerprint,
                        "train": len(port_train),
                        "validation": len(port_validation),
                        "test": len(port_test),
                    },
                    "config_fingerprint": sha256(
                        canonical_json(config).encode("utf-8")
                    ).hexdigest(),
                },
                overwrite=bool(files.get("overwrite_bundle", True)),
            )
            summary_path = _resolve(root, files["summary"])
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary = {
                "model_generation": 2,
                "bundle": str(bundle_path),
                "cache_key": cache.key,
                "dataset_key": dataset_key,
                "partition": {
                    "fingerprint": partition_fingerprint,
                    "train_count": len(port_train),
                    "validation_count": len(port_validation),
                    "test_count": len(port_test),
                },
                "port_epochs": len(port_history),
                "spatial_epochs": len(spatial_history),
                "port_training": port_summary,
                "spatial_training": spatial_summary,
                "test": {
                    "port": port_test_metrics,
                    "spatial": spatial_test_metrics,
                },
                "manifest": manifest,
            }
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            control.finish(
                "completed",
                phase="completed",
                message="generation-2 training completed",
                bundle=str(bundle_path),
                summary=str(summary_path),
            )
            return 0
        except TrainingStopRequested:
            control.finish(
                "stopped",
                message="generation-2 training stopped at a safe checkpoint; restart to resume",
            )
            return 2


def launch(config, argv=None, *, runner_path=None):
    parser = argparse.ArgumentParser(description="SDF-MPNEO vNext Generation-2 workflow")
    parser.add_argument(
        "--mode",
        choices=("gui", "train"),
        default=None,
        help="optional override of RUN.mode from run.py",
    )
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore Gen2 model checkpoints; the energy-v3 teacher cache is still reused",
    )
    args = parser.parse_args(argv)

    if args.worker_config:
        payload = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
        return run_generation2_training_worker(payload["config"], payload["session_dir"])

    cfg = _runtime_config(deepcopy(config))
    cfg.setdefault("TRAINING", {})["generation"] = 2
    if args.fresh:
        cfg["TRAINING"]["resume"] = False
    mode = args.mode or str(cfg.get("RUN", {}).get("mode", "gui"))
    runner = Path(runner_path or sys.argv[0]).resolve()
    if mode == "gui":
        from .training_gui import run_training_gui

        return run_training_gui(runner, cfg)

    root = Path(cfg["ROOT"])
    log_root = _resolve(root, cfg["FILES"]["log_dir"])
    session = _new_session(log_root)
    session.mkdir(parents=True, exist_ok=True)
    paths = SessionPaths.from_root(session)
    write_control_command(paths.control, "run")
    _write_worker_snapshot(paths.snapshot, cfg, session)
    return run_generation2_training_worker(cfg, session)


__all__ = [
    "GENERATION2_WORKFLOW_CONTRACT",
    "run_generation2_training_worker",
    "launch",
]
