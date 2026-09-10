"""一键训练/推理：修改下方配置后直接运行 python run.py。

推荐：
    python run.py --mode train
    python run.py --mode predict

首次使用：python -m pip install -e '.[cad,gui]'
训练默认打开 PyQt 窗口；无界面运行使用 --headless。
所有相对路径均相对于本文件，支持从 IDE 或其他工作目录启动。
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import threading
import time


# ==================== 1. 运行模式 ====================
MODE = "train"                  # "train" 训练；"predict" 推理


# ==================== 2. 输入输出路径 ====================
FILES = {
    "model": "results/uwpt/model.npz",
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",
    "resume_model": None,       # 继续旧模型时显式填写 .npz；None 表示 fresh training
}


# ==================== 3. UWPT 几何与网格（m） ====================
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


# ==================== 4. 材料与电磁参数 ====================
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
# 自动热秩下不手写定长热坐标候选；EM ROM 会从最终训练坐标域生成候选状态。
EM_CANDIDATE_STATES = None


# ==================== 5. 自动热空间截断 ====================
# None = 自动选热秩。显式整数仅用于收敛研究/复现实验。
THERMAL_RANK = None
THERMAL_TRUNCATION = {
    "mode": "automatic_physics_envelope",
    # 主要用户精度旋钮：舍弃模态稳态响应包络相对总包络的允许比例。
    "relative_tolerance": 1e-3,
    "absolute_tolerance": 0.0,
    # 训练初始模态坐标域；最终选到 r 阶后自动扩展成 [-bound,+bound]^r。
    "initial_coordinate_bound": 0.1,
    # 自动探测从低阶开始逐步扩大；若 32 阶仍不能确认尾部衰减，则退回全热空间。
    "probe_start_rank": 4,
    "max_probe_rank": 32,
    "source_bound_safety_factor": 2.0,
    "boundary_fraction": 0.25,
    "temperature_probe_axes": 4,
    # 若有严格的全域热源 M^-1 对偶范数界，可同时提供以下三项；程序会改走
    # 原有 theorem-level thermal tail certificate，而不是有限锚点包络判据。
    "initial_temperature_deviation_free": None,
    "source_dual_bound": None,
    "requested_state_tolerance": None,
    "prefer_partial_thermal_spectrum": True,
}


# ==================== 6. 无解标签残差训练 ====================
TRAINING = {
    # 空列表表示按自动热秩和 initial_coordinate_bound 自动展开，不再写死 2 个 a0。
    "initial_lower": [], "initial_upper": [],
    "operating_lower": [0.0, 0.0], "operating_upper": [10.0, 10.0],
    "time_horizon": 100000.0, "residual_tolerance": 1e-5,
    "time_sampling": "mixed_log", "time_min": 1e-6, "include_steady_state": True,
    "sample_count": 64, "validation_count": 64,
    # 以下仅为旧 v1/v2 非空 DAG checkpoint 的兼容预算；fresh fixed network 不使用
    # candidate search，但 ResearchTrainingConfig 仍保留这些字段用于旧模型继续训练。
    "max_nodes": 256, "max_degree": 3,
    "max_parent_responses": 1, "max_realization_dimension": 64,
}


# ==================== 7. 推理 ====================
PREDICTION = {
    # None = 按保存模型的最终热秩自动创建全零初态；也可显式给最终 r 维 a0。
    "a0": None,
    "operating": [5.0, 0.0],
    "times": [0.0, 0.001, 1.0, 1000.0, 100000.0, 1000000.0, "inf"],
    "geometry": None,
    "allow_time_extrapolation": True,
    "initial_temperature_file": None,
    "state_only": False,
    "allow_extrapolation": False,
}


# ==================== 8. 训练窗口与日志 ====================
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


def write_json(path, value):
    from sdfmpneo.__main__ import jsonable
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def resolve_training_rank(config, rank):
    """Resolve an empty user a0 box after the saved/automatic thermal rank is known."""
    rank = int(rank)
    if len(config.initial_lower) == len(config.initial_upper) == 0:
        bound = float(THERMAL_TRUNCATION.get("initial_coordinate_bound", 0.1))
        return replace(config, initial_lower=(-bound,) * rank, initial_upper=(bound,) * rank)
    if len(config.initial_lower) != rank or len(config.initial_upper) != rank:
        raise ValueError(
            f"训练初态维数与最终 thermal rank={rank} 不一致；自动模式请将 initial_lower/upper 留空")
    return config


def thermal_rank_owner(model):
    return model.reference if hasattr(model, "reference") else model


def thermal_rank_summary(model):
    owner = thermal_rank_owner(model)
    rank = int(owner.core.thermal_model.rank)
    report = getattr(owner, "thermal_rank_report", None)
    if report is None:
        report = {
            "method": "saved_or_explicit_basis",
            "selected_rank": rank,
            "certified_continuous_domain": bool(owner.core.thermal_tail_certificate is not None),
        }
    return rank, report


@contextmanager
def assembly_progress(monitor=None):
    interval = float(MONITOR.get("assembly_progress_interval_s", 5.0))
    if not 0 < interval < float("inf"):
        raise ValueError("MONITOR['assembly_progress_interval_s'] 必须为有限正数")
    stop = threading.Event()
    started = time.monotonic()
    labels = {
        "assembly": "组装物理模型与参考电磁空间",
        "thermal_rank_selection": "自动选择热空间阶数",
        "geometry_em_basis": "构建跨几何共享电磁空间",
        "geometry_seed": "构造跨几何物理初始网络",
    }

    def current_label():
        if monitor is None:
            return labels["assembly"]
        try:
            phase = monitor.data.get("phase", "assembly")
        except Exception:
            phase = "assembly"
        return labels.get(phase, phase or labels["assembly"])

    def reporter():
        previous = None
        next_heartbeat = started
        while not stop.wait(0.2):
            now = time.monotonic()
            label = current_label()
            if label != previous:
                print(f"[组装进度] {label}", flush=True)
                previous = label
            if now >= next_heartbeat:
                print(f"[组装进度] {label} · 已耗时 {now-started:.1f} s", flush=True)
                next_heartbeat = now + interval

    thread = threading.Thread(target=reporter, name="assembly-progress", daemon=True)
    thread.start()
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        stop.set()
        thread.join(timeout=max(1.0, interval))
        elapsed = time.monotonic() - started
        print(f"[组装进度] {'完成' if succeeded else '中断'} · 总耗时 {elapsed:.1f} s", flush=True)


def print_training_sample_ranges(model, config):
    print("\n训练样本参数范围：", flush=True)
    if hasattr(model, "geometry_names"):
        print("  几何参数 G（物理值；网络内部归一化到 [-1, 1]）：", flush=True)
        for name, lower, upper, reference in zip(
                model.geometry_names, model.lower, model.upper, model.geometry_reference):
            print(f"    {name}: [{float(lower):.8g}, {float(upper):.8g}]"
                  f"  参考值={float(reference):.8g}", flush=True)
    else:
        print("  几何参数 G：固定几何", flush=True)
    print("  初始热坐标 a0：", flush=True)
    for index, (lower, upper) in enumerate(zip(config.initial_lower, config.initial_upper)):
        print(f"    a0[{index}]: [{float(lower):.8g}, {float(upper):.8g}]", flush=True)
    print("  工况参数 U：", flush=True)
    for index, (lower, upper) in enumerate(zip(config.operating_lower, config.operating_upper)):
        print(f"    U[{index}]: [{float(lower):.8g}, {float(upper):.8g}]", flush=True)
    sampling = getattr(config, "time_sampling", "linear")
    detail = f"采样={sampling}"
    if getattr(config, "time_min", None) is not None:
        detail += f"，time_min={float(config.time_min):.8g} s"
    print(f"  有限时间 t: [0, {float(config.time_horizon):.8g}] s；{detail}", flush=True)
    print(f"  稳态残差点: {'包含' if getattr(config, 'include_steady_state', False) else '不包含'}", flush=True)
    print(f"  配点数量: 训练={int(config.sample_count)}，独立检查={int(config.validation_count)}", flush=True)
    print(f"  残差目标: {float(config.residual_tolerance):.8g}\n", flush=True)


def generate_mesh(path):
    import numpy as np
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
        np.asarray(ENVIRONMENT["package_half_extent"]), ENVIRONMENT["seawater_padding"])
    result = mesh_underwater_wpt_geometry(
        geometry, path, geometry_tolerance=MESH["geometry_tolerance"],
        mesh_size=MESH["mesh_size"], physical_tags=UWPTPhysicalTags(**PHYSICAL_TAGS))
    mesh = result.tagged_mesh
    if not np.array_equal(mesh.mesh.boundary_nodes(), mesh.boundary_nodes(PHYSICAL_TAGS["outer_boundary"])):
        raise RuntimeError("材料界面网格不共形")
    print(f"网格已生成：{mesh.mesh.n_nodes} 节点，{mesh.mesh.n_tetrahedra} 四面体", flush=True)


def train(model_path, settings_dir, monitor=None):
    from sdfmpneo import ResearchElectroThermalModel, ResearchTrainingConfig
    from sdfmpneo.research import model_from_config

    config = ResearchTrainingConfig(**TRAINING)
    resume = FILES["resume_model"]
    settings = {"case": "uwpt", "mode": "train", "model": str(model_path),
                "training": TRAINING, "resume_model": None}
    if resume is not None:
        if monitor is not None:
            monitor.phase("loading")
        resume_path = resolve_path(resume)
        settings["resume_model"] = str(resume_path)
        print(f"加载模型继续训练：{resume_path}", flush=True)
        model = ResearchElectroThermalModel.load(resume_path)
        config = resolve_training_rank(config, model.thermal_model.rank)
    else:
        mesh_path = resolve_path(MESH["path"])
        physical = {
            **PHYSICS, **PORTS, "mesh": str(mesh_path), "materials": MATERIALS,
            "thermal_rank": THERMAL_RANK, "thermal_truncation": THERMAL_TRUNCATION,
            "training": TRAINING,
        }
        if GEOMETRY_FAMILY["enabled"]:
            physical["geometry_family"] = {
                **GEOMETRY_FAMILY, "transmitter": TRANSMITTER,
                "receiver": RECEIVER, "physical_tags": PHYSICAL_TAGS}
        if EM_CANDIDATE_STATES is not None:
            physical["em_candidate_states"] = EM_CANDIDATE_STATES
        settings.update(physics=physical, mesh=MESH, transmitter=TRANSMITTER,
                        receiver=RECEIVER, environment=ENVIRONMENT, physical_tags=PHYSICAL_TAGS)
        if MESH["generate"]:
            if monitor is not None:
                monitor.phase("mesh")
            print("生成线圈、封装和海水网格……", flush=True)
            generate_mesh(mesh_path)
        elif not mesh_path.is_file():
            raise FileNotFoundError(f"网格不存在：{mesh_path}；可设置 MESH['generate']=True")
        config_path = settings_dir / "model.config.json"
        write_json(config_path, physical)
        print("自动选择热秩、组装物理模型并构建电磁降阶空间……", flush=True)
        if monitor is not None:
            monitor.phase("assembly")
        with assembly_progress(monitor):
            if GEOMETRY_FAMILY["enabled"]:
                from sdfmpneo.geometry_research import geometry_model_from_config
                model, config = geometry_model_from_config(config_path, monitor=monitor)
            else:
                model, config = model_from_config(config_path, monitor=monitor)
        if GEOMETRY_FAMILY["enabled"]:
            write_json(settings_dir / "geometry.domain.json", {
                "names": model.geometry_names, "reference": model.geometry_reference,
                "lower": model.lower, "upper": model.upper,
                "mesh_certificate": model.certificate, "em_basis": model.em_basis_report})
            print(f"共享几何代理：{len(model.geometry_names)} 个几何输入，EM 基维数={model.reference.em.n_reduced}", flush=True)

    rank, rank_report = thermal_rank_summary(model)
    config = resolve_training_rank(config, rank)
    settings["selected_thermal_rank"] = rank
    settings["training_resolved"] = config
    write_json(settings_dir / "thermal.rank.json", rank_report)
    print(f"最终热空间阶数：rank={rank}；选择方法={rank_report.get('method', 'unknown')}", flush=True)
    if not rank_report.get("certified_continuous_domain", False):
        print("热秩说明：当前自动结果是有限物理锚点收敛判据，不冒充连续域严格证书。", flush=True)

    write_json(settings_dir / "train.settings.json", settings)
    print_training_sample_ranges(model, config)
    print("开始无解标签残差训练（fresh fixed network 不进行 candidate search）……", flush=True)
    from sdfmpneo.training.monitor import TrainingStopped
    try:
        report = model.train(config, monitor=monitor, progress=lambda n, r, m: print(
            f"有效响应通道={n}  RMS残差={r:.6g}  最大残差={m:.6g}", flush=True))
    except TrainingStopped:
        checkpoint = model_path.with_name(model_path.stem + ".stopped" + model_path.suffix)
        if model.graph is not None:
            model.save(checkpoint)
            print(f"训练已停止，有效模型已保存：{checkpoint}", flush=True)
        else:
            checkpoint = None
        stopped = {"status": "stopped", "checkpoint": None if checkpoint is None else str(checkpoint),
                   "numerical_tolerance_met": False}
        write_json(settings_dir / "training.stopped.json", stopped)
        if monitor is not None:
            monitor.finish("stopped", **stopped)
        return 130

    if monitor is not None:
        monitor.phase("saving", check=False)
    model.save(model_path)
    write_json(settings_dir / "training.report.json", report)
    if hasattr(model.graph, "structure_summary"):
        structure = model.graph.structure_summary(0.0)
        write_json(settings_dir / "network.structure.json", structure)
        print("最终网络有效结构：" + json.dumps(structure, ensure_ascii=False), flush=True)
    print(f"训练状态：{report.status}；独立检查最大残差={report.maximum_validation_residual:.6g}")
    print(f"模型已保存：{model_path}")
    if not report.numerical_tolerance_met:
        print("本次尚未达到残差目标；已保存当前模型和报告，退出码为 2。")
    if monitor is not None:
        monitor.finish("completed" if report.numerical_tolerance_met else report.status,
                       model=str(model_path), numerical_tolerance_met=report.numerical_tolerance_met)
    return 0 if report.numerical_tolerance_met else 2


def predict(model_path, output_path, settings_dir):
    import numpy as np
    from sdfmpneo import ResearchElectroThermalModel

    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}；请先运行 python run.py --mode train")
    model = ResearchElectroThermalModel.load(model_path)
    parameters = dict(PREDICTION)
    initial = parameters["a0"]
    if initial is None:
        initial = np.zeros(model.graph.n_modes, dtype=float)
    geometry_args = {}
    if hasattr(model, "geometry_names"):
        g = parameters.get("geometry")
        if g is None:
            g = dict(zip(model.geometry_names, (model.lower + model.upper) / 2))
        parameters["geometry"] = g
        geometry_args = {"geometry": g}
    temperature_file = parameters["initial_temperature_file"]
    if temperature_file is not None:
        temperature_path = resolve_path(temperature_file)
        initial = model.project_initial_temperature(
            np.load(temperature_path, allow_pickle=False), **geometry_args)
        parameters["initial_temperature_file"] = str(temperature_path)
    if not parameters["times"]:
        raise ValueError("推理 times 至少需要一个时间点")
    print(f"加载模型推理：{model_path}；thermal rank={model.graph.n_modes}", flush=True)
    results = [model.predict(
        t, a0=initial, operating=parameters["operating"],
        diagnostics=not parameters["state_only"],
        allow_extrapolation=parameters["allow_extrapolation"],
        allow_time_extrapolation=parameters.get("allow_time_extrapolation", True),
        **geometry_args) for t in parameters["times"]]
    write_json(output_path, results)
    write_json(settings_dir / "predict.settings.json", {
        "case": "uwpt", "mode": "predict", "model": str(model_path),
        "predictions": str(output_path), "prediction": parameters, "effective_a0": initial})
    for result in results:
        if isinstance(result, dict):
            time_value, maximum = result["time"], result["maximum_temperature"]
        else:
            time_value, maximum = result.time, result.maximum_temperature
        display_time = "inf" if time_value == "inf" else f"{float(time_value):g}"
        print(f"t={display_time} s，最高温度={maximum:.8g} K")
    print(f"推理结果已保存：{output_path}")
    return 0


CONFIG_NAMES = (
    "FILES", "MESH", "TRANSMITTER", "RECEIVER", "ENVIRONMENT", "PHYSICAL_TAGS",
    "PHYSICS", "MATERIALS", "PORTS", "EM_CANDIDATE_STATES", "THERMAL_RANK",
    "THERMAL_TRUNCATION", "TRAINING", "PREDICTION", "MONITOR", "GEOMETRY_FAMILY")


def configuration_snapshot(model_path):
    return {"root": str(ROOT), "model_path": str(model_path),
            "parameters": {name: globals()[name] for name in CONFIG_NAMES}}


def execute_training(model_path, settings_dir, session_dir=None):
    from datetime import datetime
    import uuid
    from sdfmpneo.training.monitor import TrainingMonitor, TrainingStopped
    if session_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        session_dir = resolve_path(MONITOR["log_dir"]) / stamp
    session_dir = Path(session_dir)
    write_json(session_dir / "settings.json", configuration_snapshot(model_path))
    print(f"训练日志：{session_dir}", flush=True)
    with TrainingMonitor(session_dir / "metrics.jsonl", session_dir / "control.json",
                         interval=MONITOR["log_interval_s"]) as monitor:
        try:
            return train(model_path, settings_dir, monitor)
        except TrainingStopped:
            print("已停止：物理模型构建尚未完成，暂无可保存的训练网络。", flush=True)
            monitor.finish("stopped", checkpoint=None, message="模型构建阶段停止，暂无网络检查点")
            return 130


def training_worker(snapshot_path):
    global ROOT
    payload = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    settings = payload["settings"]
    ROOT = Path(settings["root"])
    for name in CONFIG_NAMES:
        globals()[name] = settings["parameters"][name]
    return execute_training(Path(settings["model_path"]), resolve_path(FILES["settings_dir"]),
                            payload["session_dir"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="一键运行电磁–热代理；用户主要指定物理域和误差目标。")
    parser.add_argument("--mode", choices=("train", "predict"), default=MODE,
                        help="train=训练，predict=推理；覆盖顶部 MODE")
    parser.add_argument("--model", help="覆盖 FILES['model']，指定保存/加载的模型路径")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true", help="打开 PyQt 训练窗口")
    display.add_argument("--headless", action="store_true", help="仅后台训练及日志，不打开窗口")
    parser.add_argument("--worker-config", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_config:
        return training_worker(args.worker_config)
    model_path = resolve_path(args.model or FILES["model"])
    settings_dir = resolve_path(FILES["settings_dir"])
    if args.mode == "train":
        if not args.headless and (args.gui or MONITOR["enabled"]):
            try:
                from sdfmpneo.training.qt_monitor import launch_window
            except ImportError as exc:
                raise SystemExit('请安装图形依赖：python -m pip install -e ".[cad,gui]"；或使用 --headless。') from exc
            return launch_window(__file__, configuration_snapshot(model_path),
                                 resolve_path(MONITOR["log_dir"]), MONITOR)
        return execute_training(model_path, settings_dir)
    return predict(model_path, resolve_path(FILES["predictions"]), settings_dir)


if __name__ == "__main__":
    raise SystemExit(main())
