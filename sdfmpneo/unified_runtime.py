"""Runtime for the single unified geometry-independent neural electrothermal model."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import uuid

import numpy as np
from scipy.stats import qmc

from .unified_background import FixedMultiscaleBackground
from .unified_basis import build_residual_basis
from .unified_dataset import MaxwellOperatorDataset, generate_operator_dataset
from .unified_geometry import UnifiedUWPTGeometry, sample_geometry
from .unified_maxwell import NeuralMaxwellAccelerator
from .unified_model import UnifiedNeuralElectroThermalModel
from .unified_trainer import train_maxwell_accelerator

_CACHE_FORMAT = 3


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {"real": value.real.tolist(), "imag": value.imag.tolist()}
        return value.tolist()
    if isinstance(value, np.generic):
        if np.iscomplexobj(value):
            return {"real": float(np.real(value)), "imag": float(np.imag(value))}
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _progress(message, percent, monitor=None):
    print(f"{message}……{float(percent):.0f}%", flush=True)
    if monitor is not None:
        with monitor._lock:
            monitor.data.update(progress_percent=float(percent), progress_message=str(message))


def _signature(settings):
    """Identity of reusable physical operator data, excluding NN/optimizer choices."""
    keys = (
        "BACKGROUND",
        "DEFAULT_GEOMETRY",
        "GEOMETRY_SAMPLING",
        "PHYSICS",
        "MATERIALS",
        "REGIONS",
        "THERMAL_RANK",
        "TRAINING",
    )
    payload = {k: settings[k] for k in keys}
    payload["TRAINING"] = {
        k: v
        for k, v in payload["TRAINING"].items()
        if k not in {"network", "optimizer", "device"}
    }
    payload["cache_format"] = _CACHE_FORMAT
    text = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def build_background(settings):
    r = settings["REGIONS"]
    return FixedMultiscaleBackground.from_config(
        settings["BACKGROUND"],
        frequency_hz=settings["PHYSICS"]["frequency_hz"],
        materials=settings["MATERIALS"],
        coil_materials=r["coil_materials"],
        package_materials=r["package_materials"],
        seawater_material=r["seawater_material"],
        thermal_rank=settings["THERMAL_RANK"],
        ambient_temperature=settings["PHYSICS"]["ambient_temperature"],
    )


def _sample_geometries(settings, n, rng, background):
    out = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 100 * n:
            raise ValueError(
                "geometry sampling produced too many invalid physical geometries; "
                "adjust sampling ranges or BACKGROUND bounds"
            )
        candidate = sample_geometry(
            settings["DEFAULT_GEOMETRY"], settings.get("GEOMETRY_SAMPLING"), rng
        )
        try:
            geometry = UnifiedUWPTGeometry.from_mapping(candidate)
            background.validate_geometry(geometry)
        except ValueError:
            continue
        out.append(candidate)
    return out


def _sample_states(settings, n, seed):
    lo = np.asarray(settings["TRAINING"]["state_lower"], float)
    hi = np.asarray(settings["TRAINING"]["state_upper"], float)
    if lo.shape != (settings["THERMAL_RANK"],) or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("state bounds must match THERMAL_RANK")
    return qmc.scale(qmc.LatinHypercube(len(lo), seed=seed).random(n), lo, hi)


def _cache_paths(directory):
    return (
        directory / "unified.cache.json",
        directory / "unified.em_basis.npy",
        directory / "unified.operator_dataset.npz",
    )


def _require_effective_basis(report):
    converged = bool(report.get("converged", False)) if isinstance(report, dict) else bool(report.converged)
    if converged:
        return
    residual = (
        float(report.get("maximum_anchor_relative_residual"))
        if isinstance(report, dict)
        else float(report.maximum_anchor_relative_residual)
    )
    target = (
        float(report.get("target_relative_residual"))
        if isinstance(report, dict)
        else float(report.target_relative_residual)
    )
    rank = int(report.get("basis_dimension")) if isinstance(report, dict) else int(report.basis_dimension)
    reason = (
        str(report.get("stop_reason", "unknown"))
        if isinstance(report, dict)
        else str(report.stop_reason)
    )
    raise RuntimeError(
        "Maxwell 公共空间无法达到训练所要求的初解残差："
        f"自动 rank={rank}, maximum anchor residual={residual:.3e}, "
        f"target={target:.3e}, stop={reason}。"
        "rank 已由物理 residual 自动增加；停止意味着没有新的数值独立 residual 方向，"
        "而不是需要手工调大某个 rank 参数。"
    )


def train(settings, model_path, settings_dir, monitor=None):
    from .training.monitor import TrainingStopped

    settings_dir.mkdir(parents=True, exist_ok=True)
    sig = _signature(settings)
    meta_path, basis_path, data_path = _cache_paths(settings_dir)
    checkpoint = Path(settings["FILES"]["training_checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else Path(settings["ROOT"]) / checkpoint
    try:
        _progress("构建固定多尺度背景物理空间", 0, monitor)
        bg = build_background(settings)
        _progress("构建固定多尺度背景物理空间", 8, monitor)
        print(
            f"背景空间：{bg.n_cells} cells，{bg.n_edges} Maxwell edge DOFs，"
            f"thermal rank={settings['THERMAL_RANK']}",
            flush=True,
        )

        valid_cache = False
        cache_meta = {}
        if meta_path.is_file() and basis_path.is_file() and data_path.is_file():
            try:
                cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                valid_cache = (
                    cache_meta.get("signature") == sig
                    and int(cache_meta.get("cache_format", -1)) == _CACHE_FORMAT
                )
            except (OSError, ValueError, TypeError):
                valid_cache = False

        if valid_cache:
            _progress("复用统一物理算子训练数据", 35, monitor)
            basis_report = cache_meta.get("basis_report", {})
            _require_effective_basis(basis_report)
            V = np.load(basis_path, allow_pickle=False)
            dataset = MaxwellOperatorDataset.load(data_path)
        else:
            checkpoint.unlink(missing_ok=True)
            rng = np.random.default_rng(int(settings["TRAINING"].get("seed", 17)))
            nb = int(settings["TRAINING"].get("basis_samples", 24))
            geoms = _sample_geometries(settings, nb, rng, bg)
            states = _sample_states(settings, nb, int(settings["TRAINING"].get("seed", 17)) + 1)
            _progress("构建 residual-driven Maxwell 公共空间", 10, monitor)
            V, basis_obj = build_residual_basis(
                bg,
                geoms,
                states,
                target_relative_residual=float(
                    settings["TRAINING"].get("em_basis_anchor_residual", 2e-1)
                ),
                monitor=monitor,
            )
            _require_effective_basis(basis_obj)
            np.save(basis_path, V)
            _progress("构建 residual-driven Maxwell 公共空间", 30, monitor)

            nd = int(settings["TRAINING"].get("n_operator_samples", 512))
            geoms = _sample_geometries(settings, nd, rng, bg)
            states = _sample_states(settings, nd, int(settings["TRAINING"].get("seed", 17)) + 2)
            dataset = generate_operator_dataset(
                bg,
                V,
                geoms,
                states,
                seed=int(settings["TRAINING"].get("seed", 17)),
                monitor=monitor,
            )
            dataset.save(data_path)
            basis_report = asdict(basis_obj)
            write_json(
                meta_path,
                {
                    "cache_format": _CACHE_FORMAT,
                    "signature": sig,
                    "basis_report": basis_report,
                },
            )
            _progress("生成 Maxwell residual 训练数据", 55, monitor)

        _progress("训练 Maxwell 神经初解器", 60, monitor)
        network, report = train_maxwell_accelerator(
            dataset,
            network_settings=settings["TRAINING"].get("network"),
            training_settings=settings["TRAINING"].get("optimizer"),
            device=settings["TRAINING"].get("device", "cuda"),
            monitor=monitor,
            checkpoint_path=checkpoint,
        )
        accelerator = NeuralMaxwellAccelerator(
            network,
            V,
            residual_tolerance=float(settings["PHYSICS"].get("maxwell_residual_tolerance", 1e-7)),
            max_iterations=int(settings["PHYSICS"].get("maxwell_max_iterations", 200)),
        )
        model = UnifiedNeuralElectroThermalModel(
            bg,
            accelerator,
            default_geometry=settings["DEFAULT_GEOMETRY"],
            current_offset=settings["PORTS"].get("current_offset"),
            current_matrix=settings["PORTS"].get("current_matrix"),
        )
        if monitor is not None:
            monitor.phase("saving", check=False)
        _progress("保存统一神经物理模型", 98, monitor)
        model.save(
            model_path,
            metadata={"basis_report": basis_report, "training_report": asdict(report)},
        )
        checkpoint.unlink(missing_ok=True)
        write_json(
            settings_dir / "training.report.json",
            {
                "model": str(model_path),
                "background_cells": bg.n_cells,
                "maxwell_dofs": bg.n_edges,
                "em_basis_rank": V.shape[1],
                "basis": basis_report,
                "training": report,
            },
        )
        _progress("训练完成", 100, monitor)
        print(
            f"训练完成：自动 EM rank={V.shape[1]}，best epoch={report.best_epoch}，"
            f"validation residual loss={report.best_validation_residual_loss:.6g} "
            f"(RMS={np.sqrt(report.best_validation_residual_loss):.6g})，"
            f"test residual loss={report.test_residual_loss:.6g} "
            f"(RMS={np.sqrt(report.test_residual_loss):.6g})",
            flush=True,
        )
        print(f"模型已保存：{model_path}", flush=True)
        if monitor is not None:
            monitor.finish("completed", model=str(model_path))
        return 0
    except TrainingStopped:
        if monitor is not None:
            monitor.finish(
                "stopped", checkpoint=str(checkpoint) if checkpoint.is_file() else None
            )
        print(
            f"训练已停止；神经训练检查点：{checkpoint}"
            if checkpoint.is_file()
            else "训练已停止。",
            flush=True,
        )
        return 130


def _device(requested):
    value = str(requested)
    if value.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                return "cpu"
        except ImportError:
            return "cpu"
    return value


def predict(settings, model_path, output_path, settings_dir):
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    device = _device(settings["TRAINING"].get("device", "cuda"))
    model = UnifiedNeuralElectroThermalModel.load(model_path, device=device)
    p = settings["PREDICTION"]
    geometry = p.get("geometry") or settings["DEFAULT_GEOMETRY"]
    initial = np.asarray(p["a0"], float)
    operating = np.asarray(p["operating"], float)
    results = []
    print(f"加载统一神经物理模型：{model_path}  device={device}", flush=True)
    for requested in p["times"]:
        if isinstance(requested, str) and requested.lower() == "inf":
            r = model.steady_state(
                initial_guess=initial,
                geometry=geometry,
                operating=operating,
                tolerance=float(p.get("steady_tolerance", 1e-10)),
                max_iterations=int(p.get("steady_max_iterations", 40)),
            )
            print(
                f"t=inf，Tmax={r.maximum_temperature:.6g} K，"
                f"thermal residual={r.residual_norm:.3e}，"
                f"Maxwell residual={max(r.maxwell_final_residual):.3e}",
                flush=True,
            )
            results.append({"time": "inf", "steady_state": r})
        else:
            t = float(requested)
            r = model.predict(
                t,
                initial_state=initial,
                geometry=geometry,
                operating=operating,
                max_step=float(p.get("max_step", 100)),
                method=p.get("method", "etd2_adaptive"),
                rtol=float(p.get("rtol", 1e-5)),
                atol=float(p.get("atol", 1e-8)),
                initial_step=p.get("initial_step"),
            )
            print(
                f"t={t:g}s，Tmax={r.maximum_temperature:.6g} K，steps={r.steps}，"
                f"Maxwell initial={max(r.maxwell_initial_residual):.3e} → "
                f"final={max(r.maxwell_final_residual):.3e}，"
                f"correction iterations={max(r.maxwell_correction_iterations)}",
                flush=True,
            )
            results.append({"time": t, "prediction": r})
    write_json(
        output_path,
        {"model": str(model_path), "geometry": geometry, "operating": operating, "results": results},
    )
    write_json(settings_dir / "predict.settings.json", {"model": str(model_path), "prediction": p})
    print(f"推理结果已保存：{output_path}", flush=True)
    return 0


def _worker_from_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    wrapper = payload["settings"]
    settings = wrapper["parameters"]
    return execute_training(
        settings,
        Path(wrapper["model_path"]),
        Path(wrapper["settings_dir"]),
        Path(payload["session_dir"]),
    )


def launch(settings, argv=None):
    parser = argparse.ArgumentParser(description="统一几何 residual-corrected 神经电热求解器")
    parser.add_argument("--mode", choices=("train", "predict"), default=settings.get("MODE", "train"))
    parser.add_argument("--model")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gui", action="store_true")
    group.add_argument("--headless", action="store_true")
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config:
        return _worker_from_file(args.worker_config)

    root = Path(settings["ROOT"])
    model_path = Path(args.model or settings["FILES"]["model"])
    model_path = model_path if model_path.is_absolute() else root / model_path
    settings_dir = Path(settings["FILES"]["settings_dir"])
    settings_dir = settings_dir if settings_dir.is_absolute() else root / settings_dir
    settings_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "predict":
        out = Path(settings["FILES"]["predictions"])
        out = out if out.is_absolute() else root / out
        return predict(settings, model_path, out, settings_dir)

    if not args.headless and (args.gui or settings["MONITOR"].get("enabled", True)):
        from .training.qt_monitor import launch_window

        log_root = Path(settings["MONITOR"]["log_dir"])
        log_root = log_root if log_root.is_absolute() else root / log_root
        worker_settings = jsonable(
            {
                "root": settings["ROOT"],
                "model_path": str(model_path),
                "settings_dir": str(settings_dir),
                "parameters": settings,
            }
        )
        return launch_window(root / "run.py", worker_settings, log_root, settings["MONITOR"])
    return execute_training(settings, model_path, settings_dir)


def execute_training(settings, model_path, settings_dir, session_dir=None):
    from .training.monitor import TrainingMonitor

    if session_dir is None:
        log_root = Path(settings["MONITOR"]["log_dir"])
        log_root = (
            log_root
            if log_root.is_absolute()
            else Path(settings["ROOT"]) / log_root
        )
        session_dir = log_root / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(session_dir / "settings.json", settings)
    with TrainingMonitor(
        session_dir / "metrics.jsonl",
        session_dir / "control.json",
        interval=float(settings["MONITOR"].get("log_interval_s", 1.0)),
    ) as monitor:
        return train(settings, Path(model_path), Path(settings_dir), monitor)


__all__ = [
    "build_background",
    "execute_training",
    "jsonable",
    "launch",
    "predict",
    "train",
    "write_json",
]
