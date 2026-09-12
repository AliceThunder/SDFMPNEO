"""UWPT 一键训练/推理。

只修改本文件顶部配置，然后运行：

    python run.py --mode train
    python run.py --mode predict

训练默认使用原有 PyQt 非阻塞窗口，支持启动、暂停、恢复和停止；
``--headless`` 可关闭 GUI。run.py 会自动生成底层物理配置，不需要手写 JSON。

首次使用：

    python -m pip install -e '.[cad,gui,neural,dev]'
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np


# ==================== 1. 运行模式 ====================
MODE = "train"                  # "train" / "predict"


# ==================== 2. 输入输出路径 ====================
FILES = {
    "model": "results/uwpt/model.npz",
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",

    # None：新训练；填写已有神经 ROM 时，从该模型权重 + 原 POD 继续训练。
    # 续训会复用 settings_dir 中已冻结的 quadratic_joule_dataset，不重新生成 EM 标签。
    "resume_model": None,

    # GUI“停止”期间自动写入。再次启动 train 时若文件存在，会自动续上
    # POD、网络权重和 AdamW 动量；训练正常完成后自动删除。
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
    "shape": "circle",
    "turns": 0.5,
    "outer_half_size": 0.015,
    "pitch": 0.002,
    "conductor_width": 0.001,
    "conductor_thickness": 0.001,
    "corner_radius": 0.006,
    "translation": [0.0, 0.0, 0.0],
    "angles": [0.0, 0.0, 0.0],
}

RECEIVER = {
    "shape": "circle",
    "turns": 0.5,
    "outer_half_size": 0.015,
    "pitch": 0.002,
    "conductor_width": 0.001,
    "conductor_thickness": 0.001,
    "corner_radius": 0.006,
    "translation": [0.0, 0.0, 0.01],
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


# 一次训练覆盖整个连续几何参数盒。
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

# EM 降阶空间使用的热状态锚点。None 时自动使用状态盒上下界和中点。
EM_CANDIDATE_STATES = [
    [-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [0.1, 0.1], [0.0, 0.0],
]


# ==================== 5. 热空间截断 ====================
THERMAL_RANK = 2
THERMAL_TRUNCATION = {
    "initial_temperature_deviation_free": None,
    "source_dual_bound": None,
    "requested_state_tolerance": None,
    "prefer_partial_thermal_spectrum": True,
}


# ==================== 6. 神经 ROM 训练配置 ====================
# 为兼容旧 run.py，仍使用 initial_lower/upper 这个名字。
# 在新模型中它表示 NN 学习的“热状态 a 的有效范围”，应覆盖实际轨迹，不只是初值。
TRAINING = {
    "initial_lower": [-0.1, -0.1],
    "initial_upper": [0.1, 0.1],
    "operating_lower": [0.0, 0.0],
    "operating_upper": [10.0, 10.0],

    "n_snapshots": 2048,
    "seed": 17,
    "snapshot_workers": 1,

    "pod_rank": None,
    "pod_relative_tail_tolerance": 1e-4,

    # None=自动选择 CUDA/CPU；也可显式写 "cuda" 或 "cpu"。
    "device": None,

    "network": {
        "width": 256,
        "blocks": 4,
        "activation": "silu",
    },
    "optimizer": {
        # 对 resume_model / training checkpoint，epochs 表示本次追加的最大 epoch 数。
        "epochs": 500,
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "heat_loss_weight": 0.25,
        "patience": 50,
        "seed": 17,
        "dtype": "float32",
        "mixed_precision": False,
    },
}


# ==================== 7. 推理配置 ====================
PREDICTION = {
    "a0": [0.0, 0.0],
    "operating": [5.0, 0.0],
    "times": [0.0, 0.001, 1.0, 1000.0, 100000.0, "inf"],
    "geometry": None,
    "method": "etd2_adaptive",          # etd2_adaptive / etd2 / imex
    "max_step": 100.0,
    "initial_step": 0.001,
    "rtol": 1e-5,
    "atol": 1e-8,
    "steady_tolerance": 1e-10,
    "steady_max_iterations": 40,
    "allow_extrapolation": False,
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


def _legacy_physical_training_config():
    """旧物理构建器仍需要该 dataclass；这里只用于构造 EM/thermal ROM。"""
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
        state_lower=np.asarray(TRAINING["initial_lower"], dtype=float),
        state_upper=np.asarray(TRAINING["initial_upper"], dtype=float),
        operating_lower=np.asarray(TRAINING["operating_lower"], dtype=float),
        operating_upper=np.asarray(TRAINING["operating_upper"], dtype=float),
        n_snapshots=int(TRAINING["n_snapshots"]),
        work_directory=settings_dir,
        seed=int(TRAINING.get("seed", 0)),
        pod_rank=TRAINING.get("pod_rank"),
        pod_relative_tail_tolerance=float(TRAINING.get("pod_relative_tail_tolerance", 1e-4)),
        network_config=TRAINING.get("network"),
        training_config=TRAINING.get("optimizer"),
        device=TRAINING.get("device"),
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
    resume_value = FILES.get("resume_model")
    resume_path = None if resume_value is None else resolve_path(resume_value)
    runtime = None

    try:
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
            print(f"加载已有神经 ROM 继续训练：{resume_path}", flush=True)
            template = StructurePreservingNeuralElectroThermalROM.load(resume_path, device="cpu")
            dataset = QuadraticJouleDataset.load(dataset_path)
            runtime = NeuralTrainingRuntime(
                monitor,
                training_checkpoint,
                initial_network_state=template.surrogate.network.state_dict(),
                initial_pod=template.surrogate.pod,
            )
            with runtime.installed():
                result = retrain_neural_rom(
                    dataset,
                    template,
                    work_directory=settings_dir,
                    pod_rank=TRAINING.get("pod_rank"),
                    pod_relative_tail_tolerance=float(
                        TRAINING.get("pod_relative_tail_tolerance", 1e-4)
                    ),
                    network_config=TRAINING.get("network"),
                    training_config=TRAINING.get("optimizer"),
                    device=TRAINING.get("device"),
                    save_model=False,
                )
            physical = None
        else:
            mesh_path = resolve_path(MESH["path"])
            continuing = _continuation_requested(settings_dir, training_checkpoint)
            if MESH["generate"] and not (continuing and mesh_path.is_file()):
                if monitor is not None:
                    monitor.phase("mesh")
                print("生成线圈、封装和海水网格……", flush=True)
                generate_mesh(mesh_path)
            elif not mesh_path.is_file():
                raise FileNotFoundError(
                    f"网格不存在：{mesh_path}；可设置 MESH['generate']=True"
                )
            elif continuing:
                print("检测到训练断点/冻结数据，复用现有网格。", flush=True)

            config_path = settings_dir / "model.config.json"
            physical = _physical_config(mesh_path)
            write_json(config_path, physical)

            if monitor is not None:
                monitor.phase("assembly")
            print("组装物理模型并构建电磁降阶空间……", flush=True)
            physical_model = _build_physical_model(config_path, monitor=monitor)
            runtime = NeuralTrainingRuntime(monitor, training_checkpoint)
            common = _pipeline_common(settings_dir)
            print(
                f"生成/复用 Joule tensor 标签：{common['n_snapshots']} 个 (a,g) 样本……",
                flush=True,
            )
            with runtime.installed():
                if hasattr(physical_model, "geometry_names"):
                    result = build_geometry_neural_rom(
                        physical_model,
                        thermal_cache_size=int(GEOMETRY_FAMILY.get("cache_size", 128)),
                        **common,
                    )
                else:
                    result = build_fixed_neural_rom(physical_model, **common)

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
            "case": "uwpt",
            "mode": "train",
            "model": str(model_path),
            "training": TRAINING,
            "resume_model": None if resume_path is None else str(resume_path),
            "physics": physical,
        })
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
        checkpoint = training_checkpoint if training_checkpoint.is_file() else settings_dir / "quadratic_joule.partial.npz"
        write_json(settings_dir / "training.stopped.json", {
            "status": "stopped",
            "checkpoint": str(checkpoint),
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


def predict(model_path, output_path, settings_dir):
    from sdfmpneo.electrothermal_tensor.model import StructurePreservingNeuralElectroThermalROM

    if not model_path.is_file():
        raise FileNotFoundError(
            f"模型不存在：{model_path}；请先运行 python run.py --mode train"
        )
    device = TRAINING.get("device") or "cpu"
    model = StructurePreservingNeuralElectroThermalROM.load(model_path, device=str(device))
    initial = np.asarray(PREDICTION["a0"], dtype=float)
    operating = np.asarray(PREDICTION["operating"], dtype=float)
    geometry = _normalized_geometry(model, PREDICTION.get("geometry"))
    allow_extrapolation = bool(PREDICTION.get("allow_extrapolation", False))
    if not PREDICTION["times"]:
        raise ValueError("PREDICTION['times'] 至少需要一个时间点")

    print(f"加载神经 ROM 推理：{model_path}", flush=True)
    results = []
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
            results.append({"time": "inf", "steady_state": result})
            print(
                f"t=inf，||a||={np.linalg.norm(result.state):.8g}，"
                f"残差={result.residual_norm:.6g}，converged={result.converged}"
            )
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
            results.append(result)
            print(f"t={t:g} s，||a||={np.linalg.norm(result.state):.8g}，步数={result.steps}")

    write_json(output_path, {
        "model": str(model_path),
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
                __file__,
                configuration_snapshot(model_path),
                resolve_path(MONITOR["log_dir"]),
                MONITOR,
            )
        return execute_training(model_path, settings_dir)

    return predict(model_path, resolve_path(FILES["predictions"]), settings_dir)


if __name__ == "__main__":
    raise SystemExit(main())
