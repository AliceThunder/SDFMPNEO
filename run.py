"""UWPT 一键训练/推理。

只修改本文件顶部配置，然后运行：

    python run.py --mode train
    python run.py --mode predict

训练默认使用原有 PyQt 非阻塞窗口，支持启动、暂停、恢复和停止；
``--headless`` 可关闭 GUI。run.py 会自动生成底层物理配置，不需要手写 JSON。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import threading
import time

import numpy as np


# ==================== 1. 运行模式 ====================
MODE = "train"                  # "train" / "predict"


# ==================== 2. 输入输出路径 ====================
FILES = {
    "model": "results/uwpt/model.npz",
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",
    "resume_model": None,
    "training_checkpoint": "results/uwpt/model.training.pt",
}


# ==================== 3. UWPT 几何与网格（长度单位 m） ====================
MESH = {
    "generate": True,
    "path": "results/uwpt/uwpt.msh",
    "geometry_tolerance": 0.0005,
    "mesh_size": 0.01,
}

TRANSMITTER = {
    "shape": "circle", "turns": 0.5, "outer_half_size": 0.015,
    "pitch": 0.002, "conductor_width": 0.001, "conductor_thickness": 0.001,
    "corner_radius": 0.006, "translation": [0.0, 0.0, 0.0],
    "angles": [0.0, 0.0, 0.0],
}

RECEIVER = {
    "shape": "circle", "turns": 0.5, "outer_half_size": 0.015,
    "pitch": 0.002, "conductor_width": 0.001, "conductor_thickness": 0.001,
    "corner_radius": 0.006, "translation": [0.0, 0.0, 0.01],
    "angles": [0.0, 0.0, 0.0],
}

ENVIRONMENT = {
    "package_half_extent": [0.019, 0.019, 0.003],
    "seawater_padding": 0.006,
}

PHYSICAL_TAGS = {
    "tx_copper": 101, "rx_copper": 102,
    "tx_package": 201, "rx_package": 202, "seawater": 301,
    "tx_terminal_start": 1001, "tx_terminal_end": 1002,
    "rx_terminal_start": 1003, "rx_terminal_end": 1004,
    "outer_boundary": 2001,
}

GEOMETRY_FAMILY = {
    "enabled": True,
    "parameters": {
        "tx_planar_scale": {"bounds": [0.97, 1.03]},
        "rx_planar_scale": {"bounds": [0.97, 1.03]},
        "tx_thickness_scale": {"bounds": [0.95, 1.05]},
        "rx_thickness_scale": {"bounds": [0.95, 1.05]},
        "rx_offset_x": {"bounds": [-0.0002, 0.0002]},
        "rx_offset_y": {"bounds": [-0.0002, 0.0002]},
        "rx_gap": {"bounds": [0.0098, 0.0102]},
        "tx_package_scale": {"bounds": [0.98, 1.02]},
        "rx_package_scale": {"bounds": [0.98, 1.02]},
        "seawater_radius": {"relative": [0.98, 1.02]},
    },
    "em_anchor_count": 4,
    "cache_size": 128,
}


# ==================== 4. 材料与电磁参数（SI） ====================
PHYSICS = {
    "frequency_hz": 100000.0,
    "ambient_temperature": 293.15,
    "constitutive_relative_error": 1e-8,
    "em_energy_error": 1e-6,
}

MATERIALS = {
    "101": {
        "name": "tx_copper", "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15, "relative_permeability": 1.0,
        "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6,
    },
    "102": {
        "name": "rx_copper", "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15, "relative_permeability": 1.0,
        "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6,
    },
    "201": {
        "name": "tx_package", "electrical_conductivity": 0.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15, "relative_permeability": 1.0,
        "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6,
    },
    "202": {
        "name": "rx_package", "electrical_conductivity": 0.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15, "relative_permeability": 1.0,
        "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6,
    },
    "301": {
        "name": "seawater", "electrical_conductivity": 5.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15, "relative_permeability": 1.0,
        "thermal_conductivity": 0.6, "volumetric_heat_capacity": 4.1e6,
    },
}

PORTS = {
    "terminal_pairs": [[1001, 1002], [1003, 1004]],
    "port_names": ["tx", "rx"],
    "current_offset": None,
    "current_matrix": None,
}

# None 时由 run.py 自动使用神经训练 state box 的 lower / center / upper。
EM_CANDIDATE_STATES = None


# ==================== 5. 热空间截断 ====================
THERMAL_RANK = 2
THERMAL_TRUNCATION = {
    "initial_temperature_deviation_free": None,
    "source_dual_bound": None,
    "requested_state_tolerance": None,
    "prefer_partial_thermal_spectrum": True,
}


# ==================== 6. 神经 ROM 训练配置 ====================
TRAINING = {
    # 初始条件允许范围，只描述 a(0)。
    "initial_lower": [-0.1, -0.1],
    "initial_upper": [0.1, 0.1],

    # 神经网络真正学习的热状态范围，必须覆盖加热后的轨迹。
    # 与 initial_* 分离，避免长时间预测因为初值盒过小而被错误阻断。
    "state_lower": [-2.0, -2.0],
    "state_upper": [2.0, 2.0],

    "operating_lower": [0.0, 0.0],
    "operating_upper": [10.0, 10.0],

    "n_snapshots": 2048,
    "seed": 17,
    "snapshot_workers": 4,

    "pod_rank": None,
    "pod_relative_tail_tolerance": 1e-4,

    # 默认优先 CUDA；CUDA 不可用时 trainer 会明确提示并退回 CPU。
    "device": "cuda",

    "network": {
        "width": 256,
        "blocks": 4,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 500,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "heat_loss_weight": 0.25,
        "patience": 50,
        "seed": 17,
        "dtype": "float32",
        "mixed_precision": True,
        "validation_interval": 5,
        "preload_to_device": True,
        "enable_tf32": True,
    },
}


# ==================== 7. 推理配置 ====================
PREDICTION = {
    "a0": [0.0, 0.0],
    "operating": [5.0, 0.0],
    "times": [0.0, 0.001, 1.0, 1000.0, 100000.0, "inf"],
    "geometry": None,
    "method": "etd2_adaptive",
    "max_step": 100.0,
    "initial_step": 0.001,
    "rtol": 1e-5,
    "atol": 1e-8,
    "steady_tolerance": 1e-10,
    "steady_max_iterations": 40,

    # 默认允许旧模型长时间查询继续计算，但一旦离开训练 state box 会打印醒目警告。
    # 若希望严格阻止任何外推，改为 False。
    "allow_extrapolation": True,
}


# ==================== 8. PyQt 非阻塞训练窗口 ====================
MONITOR = {
    "enabled": True,
    "auto_start": False,
    "log_dir": "results/uwpt/logs",
    "log_interval_s": 1.0,
    "refresh_ms": 300,
    "max_plot_points": 4000,
    "compute_threads": 1,
    "assembly_progress_interval_s": 5.0,
}


# ==================== 运行实现（通常无需修改） ====================
ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
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
    value = float(np.clip(percent, 0.0, 100.0))
    print(f"{message}……{value:.0f}%", flush=True)
    if monitor is not None:
        try:
            with monitor._lock:
                monitor.data["progress_percent"] = value
                monitor.data["progress_message"] = str(message)
        except Exception:
            pass


@contextmanager
def assembly_progress(monitor=None):
    interval = float(MONITOR.get("assembly_progress_interval_s", 5.0))
    stop = threading.Event()
    started = time.monotonic()
    phase_progress = {
        "assembly": ("组装物理模型并构建参考电磁空间", 12),
        "geometry_em_basis": ("构建跨几何共享电磁降阶空间", 22),
        "geometry_seed": ("构造跨几何物理初始空间", 27),
    }

    def current():
        phase = "assembly"
        if monitor is not None:
            try:
                phase = monitor.data.get("phase", "assembly")
            except Exception:
                pass
        return phase_progress.get(phase, (str(phase), 12))

    def reporter():
        last = None
        while not stop.wait(max(0.2, interval)):
            label, percent = current()
            elapsed = time.monotonic() - started
            text = (label, percent)
            if text != last or elapsed >= interval:
                print(f"{label}……{percent}%  已耗时 {elapsed:.1f}s", flush=True)
                last = text

    thread = threading.Thread(target=reporter, name="assembly-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval))


def generate_mesh(path):
    from sdfmpneo.spatial import RigidPose, SpiralCoilGeometry, UnderwaterWPTGeometry
    from sdfmpneo.spatial.gmsh_pipeline import UWPTPhysicalTags, mesh_underwater_wpt_geometry

    def coil(settings):
        parameters = dict(settings)
        pose = RigidPose(np.asarray(parameters.pop("translation")), *parameters.pop("angles"))
        if parameters["shape"] == "circle":
            parameters.pop("corner_radius")
        return SpiralCoilGeometry(**parameters, pose=pose)

    geometry = UnderwaterWPTGeometry(
        coil(TRANSMITTER), coil(RECEIVER),
        np.asarray(ENVIRONMENT["package_half_extent"]), ENVIRONMENT["seawater_padding"],
    )
    result = mesh_underwater_wpt_geometry(
        geometry, path,
        geometry_tolerance=MESH["geometry_tolerance"],
        mesh_size=MESH["mesh_size"],
        physical_tags=UWPTPhysicalTags(**PHYSICAL_TAGS),
    )
    mesh = result.tagged_mesh
    if not np.array_equal(
        mesh.mesh.boundary_nodes(), mesh.boundary_nodes(PHYSICAL_TAGS["outer_boundary"])
    ):
        raise RuntimeError("材料界面网格不共形")
    print(f"网格已生成：{mesh.mesh.n_nodes} 节点，{mesh.mesh.n_tetrahedra} 四面体", flush=True)


def _state_lower():
    return TRAINING.get("state_lower", TRAINING["initial_lower"])


def _state_upper():
    return TRAINING.get("state_upper", TRAINING["initial_upper"])


def _legacy_physical_training_config():
    return {
        "initial_lower": TRAINING["initial_lower"],
        "initial_upper": TRAINING["initial_upper"],
        "operating_lower": TRAINING["operating_lower"],
        "operating_upper": TRAINING["operating_upper"],
        "time_horizon": 1.0,
        "residual_tolerance": 1e-5,
        "sample_count": 4,
        "validation_count": 4,
        "max_nodes": 4,
        "max_degree": 2,
    }


def _physical_config(mesh_path):
    physical = {
        **PHYSICS, **PORTS,
        "mesh": str(mesh_path),
        "materials": MATERIALS,
        "thermal_rank": THERMAL_RANK,
        "thermal_truncation": THERMAL_TRUNCATION,
        "training": _legacy_physical_training_config(),
    }
    if GEOMETRY_FAMILY["enabled"]:
        physical["geometry_family"] = {
            **GEOMETRY_FAMILY,
            "transmitter": TRANSMITTER,
            "receiver": RECEIVER,
            "physical_tags": PHYSICAL_TAGS,
        }
    if EM_CANDIDATE_STATES is not None:
        physical["em_candidate_states"] = EM_CANDIDATE_STATES
    else:
        lo = np.asarray(_state_lower(), dtype=float)
        hi = np.asarray(_state_upper(), dtype=float)
        physical["em_candidate_states"] = [lo.tolist(), (0.5 * (lo + hi)).tolist(), hi.tolist()]
    return physical


def _build_physical_model(config_path, monitor=None):
    if GEOMETRY_FAMILY["enabled"]:
        from sdfmpneo.geometry_research import geometry_model_from_config
        return geometry_model_from_config(config_path, monitor=monitor)[0]
    from sdfmpneo.research import model_from_config
    return model_from_config(config_path)[0]


def _dataset_candidates(settings_dir):
    return (
        settings_dir / "quadratic_joule_dataset.npz",
        settings_dir / "quadratic_joule_dataset.store",
    )


def _existing_dataset_path(settings_dir):
    existing = [path for path in _dataset_candidates(settings_dir) if path.exists()]
    if not existing:
        return None
    if len(existing) != 1:
        raise ValueError("同时存在 NPZ 和 store 数据集，请删除过期的一个")
    return existing[0]


def _dataset_path(dataset, settings_dir):
    directory = getattr(dataset, "directory", None)
    return Path(directory) if directory is not None else settings_dir / "quadratic_joule_dataset.npz"


def _continuation_requested(settings_dir, training_checkpoint):
    return (
        training_checkpoint.is_file()
        or (settings_dir / "quadratic_joule.partial.npz").is_file()
        or _existing_dataset_path(settings_dir) is not None
    )


def _pipeline_common(settings_dir):
    return dict(
        state_lower=np.asarray(_state_lower(), dtype=float),
        state_upper=np.asarray(_state_upper(), dtype=float),
        operating_lower=np.asarray(TRAINING["operating_lower"], dtype=float),
        operating_upper=np.asarray(TRAINING["operating_upper"], dtype=float),
        n_snapshots=int(TRAINING["n_snapshots"]),
        work_directory=settings_dir,
        seed=int(TRAINING.get("seed", 0)),
        pod_rank=TRAINING.get("pod_rank"),
        pod_relative_tail_tolerance=float(TRAINING.get("pod_relative_tail_tolerance", 1e-4)),
        network_config=TRAINING.get("network"),
        training_config=TRAINING.get("optimizer"),
        device=TRAINING.get("device", "cuda"),
        save_model=False,
        snapshot_workers=int(TRAINING.get("snapshot_workers", 1)),
    )


def train(model_path, settings_dir, monitor=None):
    from sdfmpneo.electrothermal_tensor.dataset import QuadraticJouleDataset
    from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM
    from sdfmpneo.electrothermal_tensor.neural_control import NeuralTrainingRuntime
    from sdfmpneo.electrothermal_tensor.pipeline import (
        build_fixed_neural_rom,
        build_geometry_neural_rom,
        retrain_neural_rom,
    )
    from sdfmpneo.training.monitor import TrainingStopped

    settings_dir.mkdir(parents=True, exist_ok=True)
    training_checkpoint = resolve_path(FILES["training_checkpoint"])
    snapshot_checkpoint = settings_dir / "quadratic_joule.partial.npz"
    resume_value = FILES.get("resume_model")
    resume_path = None if resume_value is None else resolve_path(resume_value)
    runtime = None

    try:
        _progress("准备训练", 0, monitor)
        if resume_path is not None:
            if not resume_path.is_file():
                raise FileNotFoundError(f"续训模型不存在：{resume_path}")
            dataset_path = _existing_dataset_path(settings_dir)
            if dataset_path is None:
                raise FileNotFoundError(
                    "resume_model 续训需要原冻结数据集 quadratic_joule_dataset.npz/.store"
                )
            if monitor is not None:
                monitor.phase("loading")
            _progress("加载已有神经 ROM 与冻结数据", 55, monitor)
            template = StructurePreservingNeuralElectroThermalROM.load(resume_path, device="cpu")
            dataset = QuadraticJouleDataset.load(dataset_path)
            runtime = NeuralTrainingRuntime(
                monitor,
                training_checkpoint,
                initial_network_state=template.surrogate.network.state_dict(),
                initial_pod=template.surrogate.pod,
                total_snapshots=int(dataset.n_samples),
            )
            with runtime.installed():
                result = retrain_neural_rom(
                    dataset,
                    template,
                    work_directory=settings_dir,
                    pod_rank=TRAINING.get("pod_rank"),
                    pod_relative_tail_tolerance=float(TRAINING.get("pod_relative_tail_tolerance", 1e-4)),
                    network_config=TRAINING.get("network"),
                    training_config=TRAINING.get("optimizer"),
                    device=TRAINING.get("device", "cuda"),
                    save_model=False,
                )
            physical = None
        else:
            mesh_path = resolve_path(MESH["path"])
            continuing = _continuation_requested(settings_dir, training_checkpoint)
            if MESH["generate"] and not (continuing and mesh_path.is_file()):
                if monitor is not None:
                    monitor.phase("mesh")
                _progress("生成 UWPT 网格", 3, monitor)
                generate_mesh(mesh_path)
                _progress("生成 UWPT 网格", 8, monitor)
            elif not mesh_path.is_file():
                raise FileNotFoundError(f"网格不存在：{mesh_path}；可设置 MESH['generate']=True")
            elif continuing:
                print("检测到训练断点/冻结数据，复用现有网格。", flush=True)

            config_path = settings_dir / "model.config.json"
            physical = _physical_config(mesh_path)
            write_json(config_path, physical)

            if monitor is not None:
                monitor.phase("assembly")
            _progress("组装物理模型并构建电磁降阶空间", 10, monitor)
            with assembly_progress(monitor):
                physical_model = _build_physical_model(config_path, monitor=monitor)
            _progress("组装物理模型并构建电磁降阶空间", 30, monitor)

            runtime = NeuralTrainingRuntime(
                monitor,
                training_checkpoint,
                total_snapshots=int(TRAINING["n_snapshots"]),
                snapshot_checkpoint_path=snapshot_checkpoint,
            )
            common = _pipeline_common(settings_dir)
            _progress("生成或复用 Joule tensor 标签", 31, monitor)
            with runtime.installed():
                if hasattr(physical_model, "geometry_names"):
                    result = build_geometry_neural_rom(
                        physical_model,
                        thermal_cache_size=int(GEOMETRY_FAMILY.get("cache_size", 128)),
                        **common,
                    )
                else:
                    result = build_fixed_neural_rom(physical_model, **common)
            physical = physical

        _progress("保存神经 ROM", 99, monitor)
        if monitor is not None:
            monitor.phase("saving", check=False)
        result.model.save(
            model_path,
            metadata={
                "dataset_hash": result.dataset.manifest().dataset_hash,
                "pod_rank": result.pod.rank,
                "training_report": result.training_report.__dict__,
            },
        )
        runtime.clear_checkpoint()

        report = {
            "model": str(model_path),
            "dataset": str(_dataset_path(result.dataset, settings_dir)),
            "pod_rank": result.pod.rank,
            "pod_energy_fraction": result.pod.energy_fraction(),
            "training": result.training_report,
            "resumed_from": None if resume_path is None else str(resume_path),
        }
        write_json(settings_dir / "training.report.json", report)
        write_json(settings_dir / "train.settings.json", {
            "case": "uwpt", "mode": "train", "model": str(model_path),
            "training": TRAINING,
            "resume_model": None if resume_path is None else str(resume_path),
            "physics": physical,
        })
        _progress("训练完成", 100, monitor)
        print(f"POD rank={result.pod.rank}，能量比例={result.pod.energy_fraction():.8g}")
        print(
            f"NN 最佳 epoch={result.training_report.best_epoch}，"
            f"validation loss={result.training_report.best_validation_loss:.6g}"
        )
        print(f"模型已保存：{model_path}")
        if monitor is not None:
            monitor.finish("completed", model=str(model_path), checkpoint=None)
        return 0

    except TrainingStopped:
        checkpoint = training_checkpoint if training_checkpoint.is_file() else snapshot_checkpoint
        write_json(settings_dir / "training.stopped.json", {
            "status": "stopped", "checkpoint": str(checkpoint),
            "resume_model": None if resume_path is None else str(resume_path),
        })
        print(f"训练已停止；续训状态：{checkpoint}", flush=True)
        if monitor is not None:
            monitor.finish("stopped", checkpoint=str(checkpoint))
        return 130


def _normalized_geometry(model, configured):
    dimension = model.surrogate.geometry_dimension
    if dimension == 0:
        return np.empty(0)
    family = model.thermal_operators
    if configured is None:
        return np.zeros(dimension)
    if isinstance(configured, dict):
        physical = family.geometry_reference.copy()
        for index, name in enumerate(family.parameter_names):
            if name in configured:
                physical[index] = float(configured[name])
        return 2.0 * (physical - 0.5 * (family.lower + family.upper)) / (
            family.upper - family.lower
        )
    value = np.asarray(configured, dtype=float).reshape(-1)
    if value.shape != (dimension,):
        raise ValueError(f"PREDICTION['geometry'] 需要 {dimension} 个归一化几何坐标")
    return value


def _runtime_device():
    requested = str(TRAINING.get("device") or "cuda")
    if requested.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA 不可用，推理自动退回 CPU。", flush=True)
                return "cpu"
        except ImportError:
            return "cpu"
    return requested


def _temperature_summary(model, state):
    """Return physically readable temperature output when the saved operator has a basis."""
    basis = getattr(model.thermal_operators, "thermal_basis", None)
    if basis is None:
        return None
    a = np.asarray(state, dtype=float).reshape(-1)
    phi = np.asarray(basis, dtype=float)
    if phi.ndim != 2 or phi.shape[1] != a.size:
        return None
    deviation = phi @ a
    # Dirichlet boundary nodes remain at ambient, so include zero rise in extrema.
    maximum_rise = float(max(0.0, np.max(deviation)))
    minimum_rise = float(min(0.0, np.min(deviation)))
    ambient = float(PHYSICS["ambient_temperature"])
    return {
        "maximum_temperature_rise": maximum_rise,
        "minimum_temperature_rise": minimum_rise,
        "maximum_temperature": ambient + maximum_rise,
        "minimum_temperature": ambient + minimum_rise,
        "maximum_temperature_celsius": ambient + maximum_rise - 273.15,
    }


def _state_extrapolation(model, state):
    domain = getattr(model, "training_domain", {}) or {}
    lo = domain.get("state_lower")
    hi = domain.get("state_upper")
    if lo is None or hi is None:
        return False
    value = np.asarray(state, dtype=float).reshape(-1)
    lo = np.asarray(lo, dtype=float).reshape(-1)
    hi = np.asarray(hi, dtype=float).reshape(-1)
    return value.shape == lo.shape and bool(np.any(value < lo) or np.any(value > hi))


def _print_prediction(model, time_label, state, *, steps=None, residual=None, converged=None):
    summary = _temperature_summary(model, state)
    if summary is not None:
        text = (
            f"t={time_label}，Tmax={summary['maximum_temperature']:.6g} K "
            f"({summary['maximum_temperature_celsius']:.4g} °C)，"
            f"最大温升={summary['maximum_temperature_rise']:.6g} K"
        )
    else:
        coords = np.asarray(state, dtype=float)
        text = f"t={time_label}，热降阶坐标 a={np.array2string(coords, precision=5)}"
    if steps is not None:
        text += f"，步数={steps}"
    if residual is not None:
        text += f"，稳态残差={residual:.6g}"
    if converged is not None:
        text += f"，converged={converged}"
    print(text, flush=True)
    return summary


def predict(model_path, output_path, settings_dir):
    from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM

    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}；请先运行 python run.py --mode train")
    device = _runtime_device()
    model = StructurePreservingNeuralElectroThermalROM.load(model_path, device=device)
    initial = np.asarray(PREDICTION["a0"], dtype=float)
    operating = np.asarray(PREDICTION["operating"], dtype=float)
    geometry = _normalized_geometry(model, PREDICTION.get("geometry"))
    allow_extrapolation = bool(PREDICTION.get("allow_extrapolation", True))
    if not PREDICTION["times"]:
        raise ValueError("PREDICTION['times'] 至少需要一个时间点")

    print(f"加载神经 ROM 推理：{model_path}  device={device}", flush=True)
    results = []
    warned = False
    for requested in PREDICTION["times"]:
        if isinstance(requested, str) and requested.strip().lower() == "inf":
            result = model.steady_state(
                initial_guess=initial,
                geometry=geometry,
                operating=operating,
                tolerance=float(PREDICTION.get("steady_tolerance", 1e-10)),
                max_iterations=int(PREDICTION.get("steady_max_iterations", 40)),
                allow_extrapolation=allow_extrapolation,
            )
            summary = _print_prediction(
                model, "inf", result.state,
                residual=result.residual_norm, converged=result.converged,
            )
            results.append({"time": "inf", "steady_state": result, "temperature": summary})
            state = result.state
        else:
            t = float(requested)
            result = model.predict(
                t,
                initial_state=initial,
                geometry=geometry,
                operating=operating,
                max_step=float(PREDICTION.get("max_step", 100.0)),
                method=str(PREDICTION.get("method", "etd2_adaptive")),
                allow_extrapolation=allow_extrapolation,
                rtol=float(PREDICTION.get("rtol", 1e-5)),
                atol=float(PREDICTION.get("atol", 1e-8)),
                initial_step=PREDICTION.get("initial_step"),
            )
            summary = _print_prediction(model, f"{t:g} s", result.state, steps=result.steps)
            results.append({"prediction": result, "temperature": summary})
            state = result.state
        if allow_extrapolation and not warned and _state_extrapolation(model, state):
            warned = True
            print(
                "[警告] 轨迹已经离开该模型保存的 thermal-state 训练范围；当前结果属于神经网络外推。\n"
                "       旧模型仍可继续查询，但若要保证长时间精度，请扩大 "
                "TRAINING['state_lower'/'state_upper'] 后重新训练。",
                flush=True,
            )

    write_json(output_path, {
        "model": str(model_path),
        "device": device,
        "initial_state": initial,
        "geometry_normalized": geometry,
        "operating": operating,
        "method": PREDICTION.get("method", "etd2_adaptive"),
        "results": results,
    })
    write_json(settings_dir / "predict.settings.json", {
        "case": "uwpt", "mode": "predict", "model": str(model_path),
        "predictions": str(output_path), "prediction": PREDICTION,
    })
    print(f"推理结果已保存：{output_path}")
    return 0


CONFIG_NAMES = (
    "FILES", "MESH", "TRANSMITTER", "RECEIVER", "ENVIRONMENT", "PHYSICAL_TAGS",
    "GEOMETRY_FAMILY", "PHYSICS", "MATERIALS", "PORTS", "EM_CANDIDATE_STATES",
    "THERMAL_RANK", "THERMAL_TRUNCATION", "TRAINING", "PREDICTION", "MONITOR",
)


def configuration_snapshot(model_path):
    return {
        "root": str(ROOT),
        "model_path": str(model_path),
        "parameters": {name: globals()[name] for name in CONFIG_NAMES},
    }


def execute_training(model_path, settings_dir, session_dir=None):
    from datetime import datetime
    import uuid
    from sdfmpneo.training.monitor import TrainingMonitor

    if session_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        session_dir = resolve_path(MONITOR["log_dir"]) / stamp
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    write_json(session_dir / "settings.json", configuration_snapshot(model_path))
    print(f"训练日志：{session_dir}", flush=True)
    with TrainingMonitor(
        session_dir / "metrics.jsonl",
        session_dir / "control.json",
        interval=float(MONITOR["log_interval_s"]),
    ) as monitor:
        return train(model_path, settings_dir, monitor)


def training_worker(snapshot_path):
    global ROOT
    payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    settings = payload["settings"]
    ROOT = Path(settings["root"])
    for name in CONFIG_NAMES:
        globals()[name] = settings["parameters"][name]
    return execute_training(
        Path(settings["model_path"]),
        resolve_path(FILES["settings_dir"]),
        payload["session_dir"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="UWPT 结构保持神经电热 ROM；所有参数集中在 run.py 顶部。"
    )
    parser.add_argument("--mode", choices=("train", "predict"), default=MODE)
    parser.add_argument("--model", help="覆盖 FILES['model']")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="打开 PyQt 非阻塞训练窗口")
    display.add_argument("--headless", action="store_true", help="不打开 GUI")
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_config:
        return training_worker(args.worker_config)

    model_path = resolve_path(args.model or FILES["model"])
    settings_dir = resolve_path(FILES["settings_dir"])
    settings_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        if not args.headless and (args.gui or MONITOR["enabled"]):
            try:
                from sdfmpneo.training.qt_monitor import launch_window
            except ImportError as exc:
                raise SystemExit(
                    '请安装 GUI 依赖：python -m pip install -e ".[cad,gui,neural]"；'
                    '或使用 --headless。'
                ) from exc
            return launch_window(
                __file__, configuration_snapshot(model_path),
                resolve_path(MONITOR["log_dir"]), MONITOR,
            )
        return execute_training(model_path, settings_dir)

    return predict(model_path, resolve_path(FILES["predictions"]), settings_dir)


if __name__ == "__main__":
    raise SystemExit(main())
