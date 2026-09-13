"""Runtime for the single full-background neural-FGMRES electrothermal model."""
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
from .unified_dataset import MaxwellResidualDataset, generate_residual_dataset
from .unified_geometry import UnifiedUWPTGeometry, sample_geometry
from .unified_maxwell import NeuralMaxwellAccelerator
from .unified_model import UnifiedNeuralElectroThermalModel
from .unified_thermal import build_thermal_basis
from .unified_trainer import train_maxwell_accelerator

_CACHE_FORMAT = 6


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
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _progress(message, percent, monitor=None):
    print(f"{message}……{float(percent):.0f}%", flush=True)
    if monitor is not None:
        with monitor._lock:
            monitor.data.update(progress_percent=float(percent), progress_message=str(message))


def _signature(settings):
    keys = ("BACKGROUND", "DEFAULT_GEOMETRY", "GEOMETRY_SAMPLING", "PHYSICS",
            "MATERIALS", "REGIONS", "TRAINING")
    payload = {k: settings[k] for k in keys}
    payload["TRAINING"] = {k: v for k, v in payload["TRAINING"].items()
                           if k not in {"network", "optimizer", "device"}}
    payload["cache_format"] = _CACHE_FORMAT
    text = json.dumps(jsonable(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def build_background(settings):
    regions = settings["REGIONS"]
    return FixedMultiscaleBackground.from_config(
        settings["BACKGROUND"],
        frequency_hz=settings["PHYSICS"]["frequency_hz"],
        materials=settings["MATERIALS"],
        coil_materials=regions["coil_materials"],
        package_materials=regions["package_materials"],
        seawater_material=regions["seawater_material"],
        ambient_temperature=settings["PHYSICS"]["ambient_temperature"],
    )


def _sample_geometries(settings, n, rng, background):
    out = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 100 * n:
            raise ValueError("geometry sampling produced too many invalid physical geometries; "
                             "adjust sampling ranges or BACKGROUND bounds")
        candidate = sample_geometry(settings["DEFAULT_GEOMETRY"], settings.get("GEOMETRY_SAMPLING"), rng)
        try:
            background.validate_geometry(UnifiedUWPTGeometry.from_mapping(candidate))
        except ValueError:
            continue
        out.append(candidate)
    return out


def _temperature_sensitive_materials(background):
    return [name for name, material in background.materials.items()
            if float(material.get("electrical_conductivity", 0.0)) > 0.0
            and float(material.get("resistivity_temperature_coefficient", 0.0)) != 0.0]


def _sample_em_states(settings, n, seed, background):
    names = _temperature_sensitive_materials(background)
    if not names:
        return [None] * n
    configured = settings["TRAINING"].get("em_temperature_rise_bounds", [0.0, 80.0])
    if isinstance(configured, dict):
        bounds = np.asarray([configured.get(name, [0.0, 80.0]) for name in names], float)
    else:
        pair = np.asarray(configured, float)
        if pair.shape != (2,):
            raise ValueError("em_temperature_rise_bounds must be [lower, upper] or a material mapping")
        bounds = np.tile(pair, (len(names), 1))
    if bounds.shape != (len(names), 2) or np.any(~np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("invalid EM material temperature-rise bounds")
    unit = qmc.LatinHypercube(len(names), seed=seed).random(n)
    values = qmc.scale(unit, bounds[:, 0], bounds[:, 1])
    return [{name: float(row[j]) for j, name in enumerate(names)} for row in values]


def _cache_paths(directory):
    return (directory / "unified.cache.json",
            directory / "unified.thermal_basis.npy",
            directory / "unified.residual_dataset.npz")


def _require_effective_thermal_basis(report):
    converged = bool(report.get("converged", False)) if isinstance(report, dict) else bool(report.converged)
    if converged:
        return
    get = report.get if isinstance(report, dict) else lambda name, default=None: getattr(report, name, default)
    residual = float(get("maximum_anchor_relative_residual"))
    target = float(get("target_relative_residual"))
    rank = int(get("basis_dimension"))
    reason = str(get("stop_reason", "unknown"))
    raise RuntimeError(f"thermal 公共空间未达到训练要求：自动 rank={rank}, "
                       f"maximum anchor residual={residual:.3e}, target={target:.3e}, stop={reason}。")


def train(settings, model_path, settings_dir, monitor=None):
    from .training.monitor import TrainingStopped

    settings_dir.mkdir(parents=True, exist_ok=True)
    sig = _signature(settings)
    meta_path, thermal_path, data_path = _cache_paths(settings_dir)
    checkpoint = Path(settings["FILES"]["training_checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else Path(settings["ROOT"]) / checkpoint
    try:
        _progress("构建固定多尺度背景物理空间", 0, monitor)
        bg = build_background(settings)
        _progress("构建固定多尺度背景物理空间", 8, monitor)
        print(f"背景空间：{bg.n_cells} cells，{bg.n_edges} Maxwell edge DOFs，"
              "thermal rank=自动计算；Maxwell 不做全局降阶", flush=True)

        valid_cache = False
        cache_meta = {}
        if meta_path.is_file() and thermal_path.is_file() and data_path.is_file():
            try:
                cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                valid_cache = (cache_meta.get("signature") == sig
                               and int(cache_meta.get("cache_format", -1)) == _CACHE_FORMAT)
            except (OSError, ValueError, TypeError):
                valid_cache = False

        if valid_cache:
            _progress("复用 thermal 公共空间与 residual 采样", 35, monitor)
            thermal_report = cache_meta.get("thermal_basis_report", {})
            _require_effective_thermal_basis(thermal_report)
            bg.set_thermal_basis(np.load(thermal_path, allow_pickle=False))
            dataset = MaxwellResidualDataset.load(data_path)
            print(f"复用公共空间：thermal rank={bg.thermal_rank}；Maxwell full edge-space={bg.n_edges} DOFs",
                  flush=True)
        else:
            checkpoint.unlink(missing_ok=True)
            rng = np.random.default_rng(int(settings["TRAINING"].get("seed", 17)))
            n_basis = int(settings["TRAINING"].get("basis_samples", 24))
            basis_geometries = _sample_geometries(settings, n_basis, rng, bg)

            _progress("构建 residual-driven thermal 公共空间", 9, monitor)
            thermal_basis, thermal_obj = build_thermal_basis(
                bg,
                basis_geometries,
                target_relative_residual=float(settings["TRAINING"].get("thermal_basis_anchor_residual", 5e-2)),
                monitor=monitor,
            )
            _require_effective_thermal_basis(thermal_obj)
            np.save(thermal_path, thermal_basis)
            thermal_report = asdict(thermal_obj)
            _progress("构建 residual-driven thermal 公共空间", 28, monitor)

            n_data = int(settings["TRAINING"].get("n_operator_samples", 96))
            data_geometries = _sample_geometries(settings, n_data, rng, bg)
            data_states = _sample_em_states(settings, n_data,
                                            int(settings["TRAINING"].get("seed", 17)) + 2, bg)
            dataset = generate_residual_dataset(
                bg,
                data_geometries,
                data_states,
                seed=int(settings["TRAINING"].get("seed", 17)),
                residual_steps=int(settings["TRAINING"].get("residual_training_steps", 3)),
                monitor=monitor,
            )
            dataset.save(data_path)
            write_json(meta_path, {
                "cache_format": _CACHE_FORMAT,
                "signature": sig,
                "thermal_basis_report": thermal_report,
                "maxwell_representation": "full_sparse_edge_space",
            })
            _progress("生成 solution-label-free Maxwell residual 采样", 38, monitor)

        _progress("训练 full-edge Maxwell residual corrector", 40, monitor)
        network, report = train_maxwell_accelerator(
            bg,
            dataset,
            network_settings=settings["TRAINING"].get("network"),
            training_settings=settings["TRAINING"].get("optimizer"),
            device=settings["TRAINING"].get("device", "cuda"),
            monitor=monitor,
            checkpoint_path=checkpoint,
        )
        accelerator = NeuralMaxwellAccelerator(
            network,
            residual_tolerance=float(settings["PHYSICS"].get("maxwell_residual_tolerance", 1e-7)),
            max_iterations=int(settings["PHYSICS"].get("maxwell_max_iterations", 200)),
            restart=int(settings["PHYSICS"].get("maxwell_restart", 40)),
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
        model.save(model_path, metadata={
            "thermal_basis_report": thermal_report,
            "maxwell_representation": "full_sparse_edge_space",
            "training_report": asdict(report),
        })
        checkpoint.unlink(missing_ok=True)
        write_json(settings_dir / "training.report.json", {
            "model": str(model_path),
            "background_cells": bg.n_cells,
            "maxwell_dofs": bg.n_edges,
            "maxwell_representation": "full_sparse_edge_space",
            "thermal_basis_rank": bg.thermal_rank,
            "thermal_basis": thermal_report,
            "training": report,
        })
        _progress("训练完成", 100, monitor)
        print(f"训练完成：thermal rank={bg.thermal_rank}，Maxwell={bg.n_edges} full edge DOFs（无 Maxwell rank），"
              f"best epoch={report.best_epoch}，validation residual loss="
              f"{report.best_validation_residual_loss:.6g} (RMS={np.sqrt(report.best_validation_residual_loss):.6g})，"
              f"test residual loss={report.test_residual_loss:.6g} (RMS={np.sqrt(report.test_residual_loss):.6g})",
              flush=True)
        print(f"模型已保存：{model_path}", flush=True)
        if monitor is not None:
            monitor.finish("completed", model=str(model_path))
        return 0
    except TrainingStopped:
        if monitor is not None:
            monitor.finish("stopped", checkpoint=str(checkpoint) if checkpoint.is_file() else None)
        print(f"训练已停止；神经训练检查点：{checkpoint}" if checkpoint.is_file() else "训练已停止。",
              flush=True)
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


def _initial_state(model, prediction):
    value = prediction.get("initial_temperature_rise", 0.0)
    if value is None or (isinstance(value, str) and value.lower() == "ambient"):
        return np.zeros(model.thermal_rank)
    array = np.asarray(value, float)
    if array.ndim == 0 and float(array) == 0.0:
        return np.zeros(model.thermal_rank)
    return model.background.project_temperature_rise(array)


def predict(settings, model_path, output_path, settings_dir):
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}")
    device = _device(settings["TRAINING"].get("device", "cuda"))
    model = UnifiedNeuralElectroThermalModel.load(model_path, device=device)
    p = settings["PREDICTION"]
    geometry = p.get("geometry") or settings["DEFAULT_GEOMETRY"]
    initial = _initial_state(model, p)
    operating = np.asarray(p["operating"], float)
    results = []
    print(f"加载统一神经物理模型：{model_path}  device={device}  thermal rank={model.thermal_rank}  "
          f"Maxwell={model.background.n_edges} full edge DOFs", flush=True)
    for requested in p["times"]:
        if isinstance(requested, str) and requested.lower() == "inf":
            result = model.steady_state(
                initial_guess=initial,
                geometry=geometry,
                operating=operating,
                tolerance=float(p.get("steady_tolerance", 1e-10)),
                max_iterations=int(p.get("steady_max_iterations", 40)),
            )
            print(f"t=inf，Tmax={result.maximum_temperature:.6g} K，thermal residual={result.residual_norm:.3e}，"
                  f"Maxwell residual={max(result.maxwell_final_residual):.3e}", flush=True)
            results.append({"time": "inf", "steady_state": result})
        else:
            t = float(requested)
            result = model.predict(
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
            print(f"t={t:g}s，Tmax={result.maximum_temperature:.6g} K，steps={result.steps}，"
                  f"Maxwell initial={max(result.maxwell_initial_residual):.3e} → "
                  f"final={max(result.maxwell_final_residual):.3e}，"
                  f"FGMRES iterations={max(result.maxwell_correction_iterations)}，"
                  f"restarts={max(result.maxwell_restarts)}", flush=True)
            results.append({"time": t, "prediction": result})
    write_json(output_path, {"model": str(model_path), "geometry": geometry,
                             "operating": operating, "results": results})
    write_json(settings_dir / "predict.settings.json", {"model": str(model_path), "prediction": p})
    print(f"推理结果已保存：{output_path}", flush=True)
    return 0


def _worker_from_file(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    wrapper = payload["settings"]
    return execute_training(wrapper["parameters"], Path(wrapper["model_path"]),
                            Path(wrapper["settings_dir"]), Path(payload["session_dir"]))


def launch(settings, argv=None):
    parser = argparse.ArgumentParser(description="统一几何 full-space neural-FGMRES 神经电热求解器")
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
        output = Path(settings["FILES"]["predictions"])
        output = output if output.is_absolute() else root / output
        return predict(settings, model_path, output, settings_dir)
    if not args.headless and (args.gui or settings["MONITOR"].get("enabled", True)):
        from .training.qt_monitor import launch_window
        log_root = Path(settings["MONITOR"]["log_dir"])
        log_root = log_root if log_root.is_absolute() else root / log_root
        worker_settings = jsonable({
            "root": settings["ROOT"], "model_path": str(model_path),
            "settings_dir": str(settings_dir), "parameters": settings,
        })
        return launch_window(root / "run.py", worker_settings, log_root, settings["MONITOR"])
    return execute_training(settings, model_path, settings_dir)


def execute_training(settings, model_path, settings_dir, session_dir=None):
    from .training.monitor import TrainingMonitor
    if session_dir is None:
        log_root = Path(settings["MONITOR"]["log_dir"])
        log_root = log_root if log_root.is_absolute() else Path(settings["ROOT"]) / log_root
        session_dir = log_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(session_dir / "settings.json", settings)
    with TrainingMonitor(session_dir / "metrics.jsonl", session_dir / "control.json",
                         interval=float(settings["MONITOR"].get("log_interval_s", 1.0))) as monitor:
        return train(settings, Path(model_path), Path(settings_dir), monitor)


__all__ = ["build_background", "execute_training", "jsonable", "launch", "predict", "train", "write_json"]
