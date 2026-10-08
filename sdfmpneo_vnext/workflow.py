from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid

from .controlled_training import train_port_controlled, train_spatial_controlled
from .tensor_bundle import publish_tensor_bundle
from .training_control import (
    SessionPaths,
    TrainingControl,
    TrainingStopRequested,
    write_control_command,
)
from .workflow_cache import TensorTeacherCache, canonical_json, teacher_cache_key


def _resolve(root: Path, value) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _runtime_config(config):
    cfg = deepcopy(config)
    root = Path(cfg.get("ROOT", ".")).expanduser().resolve(strict=False)
    cfg["ROOT"] = str(root)
    runtime_device = str(cfg.get("RUNTIME", {}).get("device", "auto"))
    for name in ("PORT_TRAINING", "SPATIAL_TRAINING"):
        section = cfg.get(name)
        if isinstance(section, dict) and str(section.get("device", "inherit")) == "inherit":
            section["device"] = runtime_device
    return cfg


def _training_dataset_key(cache_key: str, count: int, config) -> str:
    """Identity of the ordered samples and deterministic train/validation policy.

    Optimizer/model changes are handled by trainer checkpoint signatures. This
    key only adds partition/objective semantics that must not silently inherit
    an old optimizer trajectory. Epoch limits and accelerator choice are
    intentionally absent so a run can be extended or moved between CPU/GPU.
    """
    port = dict(config.get("PORT_TRAINING", {}) or {})
    spatial = dict(config.get("SPATIAL_TRAINING", {}) or {})
    payload = {
        "teacher_cache_key": cache_key,
        "count": int(count),
        "port": {
            "seed": int(port.get("seed", 17)),
            "validation_fraction": float(port.get("validation_fraction", 0.15)),
            "channel_loss_weight": float(port.get("channel_loss_weight", 1.0)),
        },
        "spatial": {
            "seed": int(spatial.get("seed", 47)),
            "validation_fraction": float(spatial.get("validation_fraction", 0.15)),
            "end_to_end_validation": bool(spatial.get("end_to_end_validation", True)),
            "validation_interval": int(spatial.get("validation_interval", 1)),
        },
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _training_history_summary(history, configured_epochs: int):
    """Compact, JSON-safe quality summary for fresh or resumed training history."""
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
        (
            row
            for row in rows
            if int(row.get("epoch", -1)) == best_epoch
        ),
        last,
    )

    def metric(row, name):
        value = row.get(name)
        return None if value is None else float(value)

    best_validation_loss = metric(last, "best_validation_loss")
    if best_validation_loss is None:
        best_validation_loss = metric(best, "validation_loss")

    final_epoch = int(last.get("epoch") or len(rows))
    result.update(
        {
            "final_epoch": final_epoch,
            "best_epoch": best_epoch,
            "stopped_early": final_epoch < int(configured_epochs),
            "final_train_loss": metric(last, "train_loss"),
            "final_validation_loss": metric(last, "validation_loss"),
            "final_validation_shape_loss": metric(last, "validation_shape_loss"),
            "best_validation_loss": best_validation_loss,
            "best_epoch_train_loss": metric(best, "train_loss"),
            "best_epoch_validation_loss": metric(best, "validation_loss"),
            "best_epoch_validation_shape_loss": metric(best, "validation_shape_loss"),
            "device": last.get("device"),
            "dtype": last.get("dtype"),
        }
    )
    return result


def _prepare_cache(config, control):
    root = Path(config["ROOT"])
    cache_cfg = dict(config["CACHE"])
    cache_root = _resolve(root, cache_cfg["root"])
    policy = str(cache_cfg.get("policy", "reuse")).strip().lower()
    if policy not in {"reuse", "refresh", "readonly"}:
        raise ValueError("CACHE.policy must be reuse, refresh, or readonly")
    key = teacher_cache_key(config)
    target = cache_root / key
    if policy == "refresh" and target.exists():
        shutil.rmtree(target)
    cache = TensorTeacherCache(
        cache_root,
        config,
        verify_checksums=bool(cache_cfg.get("verify_checksums", True)),
    )
    count = int(config["DATA"]["count"])
    if policy == "readonly" and cache.missing_indices(count):
        raise FileNotFoundError(
            "readonly teacher cache is incomplete for the requested sample count"
        )
    control.emit(
        phase="teacher_cache",
        cache_key=cache.key,
        cache_path=str(cache.path),
        message="checking reusable teacher cache",
    )
    samples = cache.ensure(
        count,
        config=config,
        checkpoint=lambda: control.checkpoint(phase="teacher_cache"),
        progress=lambda row: control.emit(**row),
    )
    return cache, samples


def run_training_worker(config, session_dir) -> int:
    config = _runtime_config(config)
    root = Path(config["ROOT"])
    files = dict(config["FILES"])
    training_cfg = dict(config["TRAINING"])
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
            dataset_key = _training_dataset_key(cache.key, count, config)
            include_spatial = bool(config["TRUTH"].get("include_spatial", False))
            if include_spatial:
                port_samples = tuple(sample.port for sample in generated)
                spatial_samples = tuple(generated)
            else:
                port_samples = tuple(generated)
                spatial_samples = ()

            resume = bool(training_cfg.get("resume", True))
            checkpoint_root = _resolve(root, files["checkpoint_dir"])
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            port_checkpoint = checkpoint_root / str(files.get("port_checkpoint", "port.pt"))
            spatial_checkpoint = checkpoint_root / str(
                files.get("spatial_checkpoint", "spatial.pt")
            )

            control.checkpoint(
                phase="port_training",
                message="training tensor FAST port surrogate",
                dataset_key=dataset_key,
            )
            port, port_history = train_port_controlled(
                port_samples,
                config=config["PORT_TRAINING"],
                cache_key=dataset_key,
                checkpoint_path=port_checkpoint,
                control=control,
                resume=resume,
            )
            port_artifact_path = _resolve(root, files["port_artifact"])
            port_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            port.save(port_artifact_path)
            port_summary = _training_history_summary(
                port_history,
                int(config["PORT_TRAINING"].get("epochs", 200)),
            )
            control.emit(
                phase="port_training",
                event="summary",
                port_artifact=str(port_artifact_path),
                port_epochs_completed=len(port_history),
                **port_summary,
            )

            spatial = None
            spatial_history = []
            spatial_cfg = dict(config.get("SPATIAL_TRAINING", {}) or {})
            spatial_summary = _training_history_summary(
                spatial_history,
                int(spatial_cfg.get("epochs", 120)),
            )
            if bool(spatial_cfg.get("enabled", include_spatial)):
                if not include_spatial:
                    raise ValueError(
                        "SPATIAL_TRAINING.enabled requires TRUTH.include_spatial=true"
                    )
                control.checkpoint(
                    phase="spatial_training",
                    message="training continuous tensor spatial loss surrogate",
                )
                spatial, spatial_history = train_spatial_controlled(
                    port,
                    spatial_samples,
                    config=spatial_cfg,
                    cache_key=dataset_key,
                    checkpoint_path=spatial_checkpoint,
                    control=control,
                    resume=resume,
                )
                spatial_artifact_path = _resolve(root, files["spatial_artifact"])
                spatial_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                spatial.save(spatial_artifact_path)
                spatial_summary = _training_history_summary(
                    spatial_history,
                    int(spatial_cfg.get("epochs", 120)),
                )
                control.emit(
                    phase="spatial_training",
                    event="summary",
                    spatial_artifact=str(spatial_artifact_path),
                    spatial_epochs_completed=len(spatial_history),
                    **spatial_summary,
                )

            control.checkpoint(phase="publishing", message="publishing tensor bundle")
            bundle_path = _resolve(root, files["bundle"])
            manifest = publish_tensor_bundle(
                bundle_path,
                port,
                spatial_artifact=spatial,
                metadata={
                    "teacher_cache_key": cache.key,
                    "training_dataset_key": dataset_key,
                    "teacher_samples": count,
                    "config_fingerprint": sha256(
                        canonical_json(config).encode("utf-8")
                    ).hexdigest(),
                },
                overwrite=bool(files.get("overwrite_bundle", True)),
            )
            summary_path = _resolve(root, files["summary"])
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "bundle": str(bundle_path),
                        "cache_key": cache.key,
                        "dataset_key": dataset_key,
                        "port_epochs": len(port_history),
                        "spatial_epochs": len(spatial_history),
                        "port_training": port_summary,
                        "spatial_training": spatial_summary,
                        "manifest": manifest,
                    },
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ) + "\n",
                encoding="utf-8",
            )
            control.finish(
                "completed",
                phase="completed",
                message="training completed",
                bundle=str(bundle_path),
                summary=str(summary_path),
            )
            return 0
        except TrainingStopRequested:
            control.finish(
                "stopped",
                message="training stopped at a safe checkpoint; restart to resume",
            )
            return 2


def _new_session(log_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return log_root / f"{stamp}_{uuid.uuid4().hex[:8]}"


def _write_worker_snapshot(path: Path, config, session_dir: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"config": config, "session_dir": str(session_dir)},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def launch_gui(config, runner_path):
    from .training_gui import run_training_gui

    return run_training_gui(runner_path, _runtime_config(config))


def launch(config, argv=None, *, runner_path=None):
    parser = argparse.ArgumentParser(description="SDF-MPNEO vNext unified workflow")
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
        help="ignore model checkpoints for this invocation; teacher cache is still reusable",
    )
    args = parser.parse_args(argv)

    if args.worker_config:
        payload = json.loads(Path(args.worker_config).read_text(encoding="utf-8"))
        return run_training_worker(payload["config"], payload["session_dir"])

    cfg = _runtime_config(config)
    if args.fresh:
        cfg["TRAINING"]["resume"] = False
    mode = args.mode or str(cfg.get("RUN", {}).get("mode", "gui"))
    runner = Path(runner_path or sys.argv[0]).resolve()
    if mode == "gui":
        return launch_gui(cfg, runner)

    root = Path(cfg["ROOT"])
    log_root = _resolve(root, cfg["FILES"]["log_dir"])
    session = _new_session(log_root)
    session.mkdir(parents=True, exist_ok=True)
    paths = SessionPaths.from_root(session)
    write_control_command(paths.control, "run")
    _write_worker_snapshot(paths.snapshot, cfg, session)
    return run_training_worker(cfg, session)