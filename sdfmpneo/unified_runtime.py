"""Runtime for the geometry-to-tensor electrothermal production path."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import uuid

import numpy as np

from .unified_background import FixedMultiscaleBackground
from .unified_geometry import UnifiedUWPTGeometry, sample_geometry
from .unified_model import UnifiedNeuralElectroThermalModel
from .unified_tensor_surrogate import (
    TensorDataset,
    encode_geometry,
    pack_tensors,
    solve_truth_tensors,
)
from .unified_tensor_training import train_matrix_tensor_surrogate
from .unified_thermal import build_thermal_basis

_CACHE_FORMAT = 7


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
    keys = (
        "BACKGROUND",
        "DEFAULT_GEOMETRY",
        "GEOMETRY_SAMPLING",
        "PHYSICS",
        "MATERIALS",
        "REGIONS",
        "TRAINING",
    )
    payload = {k: settings[k] for k in keys}
    payload["TRAINING"] = {
        k: v for k, v in payload["TRAINING"].items()
        if k not in {"network", "optimizer", "device"}
    }
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
    while len(out) < int(n):
        attempts += 1
        if attempts > 100 * max(1, int(n)):
            raise ValueError(
                "geometry sampling produced too many invalid physical geometries; "
                "adjust sampling ranges or BACKGROUND bounds"
            )
        candidate = sample_geometry(
            settings["DEFAULT_GEOMETRY"], settings.get("GEOMETRY_SAMPLING"), rng
        )
        try:
            background.validate_geometry(UnifiedUWPTGeometry.from_mapping(candidate))
            encode_geometry(candidate)
        except ValueError:
            continue
        out.append(candidate)
    return out


def _cache_paths(directory):
    return (
        directory / "unified.cache.json",
        directory / "unified.thermal_basis.npy",
        directory / "unified.tensor_dataset.npz",
    )


def _require_effective_thermal_basis(report):
    get = report.get if isinstance(report, dict) else lambda name, default=None: getattr(report, name, default)
    if bool(get("converged", False)):
        return
    error = get("maximum_validation_relative_energy_error", None)
    if error is None or float(error) == 0.0:
        error = get("maximum_anchor_relative_energy_error", get("maximum_anchor_relative_residual", float("nan")))
    target = get("target_relative_error", get("target_relative_residual", float("nan")))
    rank = int(get("basis_dimension", 0))
    reason = str(get("stop_reason", "unknown"))
    raise RuntimeError(
        f"thermal 公共空间未达到训练要求：自动 rank={rank}, "
        f"energy error={float(error):.3e}, target={float(target):.3e}, stop={reason}。"
    )


def _split_labels(n, seed):
    if int(n) < 5:
        raise ValueError("至少需要 5 个 tensor geometry 样本")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n))
    n_test = max(1, int(round(0.1 * n)))
    n_val = max(1, int(round(0.1 * n)))
    if n - n_test - n_val < 3:
        n_test = n_val = 1
    split = np.full(int(n), "train", dtype="U16")
    split[order[:n_test]] = "test"
    split[order[n_test:n_test + n_val]] = "validation"
    return split


def _generate_tensor_dataset(background, geometries, *, seed, monitor=None):
    inputs = []
    outputs = []
    audit_rows = []
    geometries = list(geometries)
    for index, geometry in enumerate(geometries):
        if monitor is not None:
            monitor.checkpoint()
        z, d, modal, audit = solve_truth_tensors(background, geometry)
        inputs.append(encode_geometry(geometry))
        outputs.append(pack_tensors(z, d, modal))
        audit_rows.append(audit)
        print(
            f"生成 Z_field / D_vol / H_j truth……{100.0 * (index + 1) / len(geometries):5.1f}%  "
            f"({index + 1}/{len(geometries)})",
            flush=True,
        )
    inputs = np.asarray(inputs, float)
    outputs = np.asarray(outputs, float)
    audit = {
        "maximum_reciprocity_relative_error": max(
            row["reciprocity_relative_error"] for row in audit_rows
        ),
        "minimum_d_vol_eigenvalue": min(
            row["minimum_d_vol_eigenvalue"] for row in audit_rows
        ),
        "minimum_implied_outward_eigenvalue": min(
            row["minimum_implied_outward_eigenvalue"] for row in audit_rows
        ),
        "maximum_loewner_violation": max(
            row["maximum_loewner_violation"] for row in audit_rows
        ),
        "independent_outward_power_available": 0.0,
    }
    return TensorDataset(
        inputs=inputs,
        outputs=outputs,
        split=_split_labels(len(geometries), seed),
        audit=audit,
        n_ports=len(background.coil_materials),
        thermal_rank=background.thermal_rank,
    )


def _physics_gate(dataset):
    audit = dict(dataset.audit)
    reciprocity_ok = audit["maximum_reciprocity_relative_error"] <= 1e-8
    d_ok = audit["minimum_d_vol_eigenvalue"] >= -1e-9
    loewner_ok = audit["maximum_loewner_violation"] <= 1e-8
    # Gate 0 is intentionally explicit: the current fixed background still uses
    # a finite PEC truncation and therefore has no independent open-boundary
    # Poynting flux certificate.  Do not label the final model fully certified.
    return {
        "reciprocity_ok": bool(reciprocity_ok),
        "volume_passivity_ok": bool(d_ok),
        "modal_loewner_ok": bool(loewner_ok),
        "reaction_impedance_convention": "negative_source_reaction",
        "open_boundary_verified": False,
        "independent_outward_power_verified": False,
        "boundary_model": "finite_pec_truncation_provisional",
        "certified": False,
        "status": "provisional_until_open_boundary_gate",
        "audit": audit,
    }


def train(settings, model_path, settings_dir, monitor=None):
    from .training.monitor import TrainingStopped

    settings_dir.mkdir(parents=True, exist_ok=True)
    sig = _signature(settings)
    meta_path, thermal_path, data_path = _cache_paths(settings_dir)
    checkpoint = Path(settings["FILES"]["training_checkpoint"])
    checkpoint = checkpoint if checkpoint.is_absolute() else Path(settings["ROOT"]) / checkpoint
    try:
        _progress("构建固定背景物理空间", 0, monitor)
        bg = build_background(settings)
        _progress("构建固定背景物理空间", 6, monitor)
        print(
            f"背景空间：{bg.n_cells} cells，{bg.n_edges} Maxwell edge DOFs；"
            "Maxwell 仅用于离线 truth，不进入在线网络/迭代求解。",
            flush=True,
        )

        valid_cache = False
        cache_meta = {}
        if meta_path.is_file() and thermal_path.is_file() and data_path.is_file():
            try:
                cache_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                valid_cache = (
                    cache_meta.get("signature") == sig
                    and int(cache_meta.get("cache_format", -1)) == _CACHE_FORMAT
                )
            except (OSError, ValueError, TypeError):
                valid_cache = False

        if valid_cache:
            _progress("复用 thermal basis 与 tensor truth 数据", 35, monitor)
            thermal_report = cache_meta.get("thermal_basis_report", {})
            _require_effective_thermal_basis(thermal_report)
            bg.set_thermal_basis(np.load(thermal_path, allow_pickle=False))
            dataset = TensorDataset.load(data_path)
            if dataset.thermal_rank != bg.thermal_rank:
                raise RuntimeError("cached tensor dataset thermal rank 与 basis 不一致")
        else:
            checkpoint.unlink(missing_ok=True)
            rng = np.random.default_rng(int(settings["TRAINING"].get("seed", 17)))
            n_basis = int(settings["TRAINING"].get("basis_samples", 20))
            n_basis_val = int(settings["TRAINING"].get("basis_validation_samples", 6))
            basis_geometries = _sample_geometries(settings, n_basis, rng, bg)
            validation_geometries = _sample_geometries(settings, n_basis_val, rng, bg)

            _progress("构建 transient-aware thermal 公共空间", 8, monitor)
            thermal_basis, thermal_obj = build_thermal_basis(
                bg,
                basis_geometries,
                validation_geometries=validation_geometries,
                target_relative_error=float(
                    settings["TRAINING"].get("thermal_basis_energy_tolerance", 5e-2)
                ),
                time_scales=settings["TRAINING"].get(
                    "thermal_time_scales", [1e-3, 1.0, 1000.0]
                ),
                maximum_rank=settings["TRAINING"].get("thermal_basis_max_rank"),
                monitor=monitor,
            )
            _require_effective_thermal_basis(thermal_obj)
            np.save(thermal_path, thermal_basis)
            thermal_report = asdict(thermal_obj)
            _progress("构建 transient-aware thermal 公共空间", 28, monitor)

            n_tensor = int(settings["TRAINING"].get("n_tensor_samples", 96))
            tensor_geometries = _sample_geometries(settings, n_tensor, rng, bg)
            dataset = _generate_tensor_dataset(
                bg,
                tensor_geometries,
                seed=int(settings["TRAINING"].get("seed", 17)),
                monitor=monitor,
            )
            dataset.save(data_path)
            write_json(
                meta_path,
                {
                    "cache_format": _CACHE_FORMAT,
                    "signature": sig,
                    "thermal_basis_report": thermal_report,
                    "em_representation": "geometry_to_port_and_joule_tensors",
                },
            )
            _progress("生成几何 tensor truth 数据", 52, monitor)

        gate = _physics_gate(dataset)
        print(
            "Physics Gate 0：reaction sign / reciprocity / volume passivity / modal bounds 已检查；"
            "open-boundary 独立 Poynting 证据当前仍缺失，因此模型标记为 provisional。",
            flush=True,
        )

        phi = np.asarray(bg.thermal_basis, float)
        phi_min = np.min(phi, axis=0)
        phi_max = np.max(phi, axis=0)
        _progress("训练 geometry→tensor MLP", 55, monitor)
        surrogate, report = train_matrix_tensor_surrogate(
            dataset,
            phi_min,
            phi_max,
            network_settings=settings["TRAINING"].get("network"),
            training_settings=settings["TRAINING"].get("optimizer"),
            device=settings["TRAINING"].get("device", "cuda"),
            monitor=monitor,
            checkpoint_path=checkpoint,
        )
        model = UnifiedNeuralElectroThermalModel(
            bg,
            surrogate,
            default_geometry=settings["DEFAULT_GEOMETRY"],
            current_offset=settings["PORTS"].get("current_offset"),
            current_matrix=settings["PORTS"].get("current_matrix"),
        )
        _progress("保存统一 tensor-ROM 模型", 98, monitor)
        model.save(
            model_path,
            metadata={
                "thermal_basis_report": thermal_report,
                "physics_gate": gate,
                "training_report": asdict(report),
            },
        )
        checkpoint.unlink(missing_ok=True)
        write_json(
            settings_dir / "training.report.json",
            {
                "model": str(model_path),
                "background_cells": bg.n_cells,
                "offline_maxwell_dofs": bg.n_edges,
                "online_em_representation": "Z_field + D_vol + modal H_j",
                "thermal_basis_rank": bg.thermal_rank,
                "thermal_basis": thermal_report,
                "physics_gate": gate,
                "training": report,
            },
        )
        _progress("训练完成", 100, monitor)
        print(
            f"训练完成：thermal rank={bg.thermal_rank}，在线 Maxwell solve=0，"
            f"best epoch={report.best_epoch}，validation matrix loss={report.best_validation_loss:.6g}，"
            f"test relative tensor error={report.test_relative_tensor_error:.6g}。",
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
            f"训练已停止；tensor MLP 检查点：{checkpoint}"
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
    operating = p.get("drive", p.get("operating", [1.0] + [0.0] * (model.current_dimension - 1)))
    results = []
    tensors = model.tensors(geometry)
    print(
        f"加载 geometry-tensor electrothermal ROM：{model_path}  device={device}  "
        f"thermal rank={model.thermal_rank}  tensor projection correction={tensors.projection_correction:.3e}",
        flush=True,
    )
    for requested in p["times"]:
        if isinstance(requested, str) and requested.lower() == "inf":
            result = model.steady_state(
                initial_guess=initial,
                geometry=geometry,
                operating=operating,
                tolerance=float(p.get("steady_tolerance", 1e-10)),
                max_iterations=int(p.get("steady_max_iterations", 40)),
            )
            print(
                f"t=inf，Tmax={result.maximum_temperature:.6g} K，"
                f"thermal residual={result.residual_norm:.3e}，"
                f"Pvol={result.volume_power:.6g} W，Pwire={result.wire_power:.6g} W",
                flush=True,
            )
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
            print(
                f"t={t:g}s，Tmax={result.maximum_temperature:.6g} K，steps={result.steps}，"
                f"Pvol={result.volume_power:.6g} W，Pwire={result.wire_power:.6g} W",
                flush=True,
            )
            results.append({"time": t, "prediction": result})
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
    return execute_training(
        wrapper["parameters"],
        Path(wrapper["model_path"]),
        Path(wrapper["settings_dir"]),
        Path(payload["session_dir"]),
    )


def launch(settings, argv=None):
    parser = argparse.ArgumentParser(
        description="统一几何 geometry→tensor 电磁-热 ROM（在线无 Maxwell/FGMRES）"
    )
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
        log_root = log_root if log_root.is_absolute() else Path(settings["ROOT"]) / log_root
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
    "build_background", "execute_training", "jsonable", "launch", "predict", "train", "write_json"
]
