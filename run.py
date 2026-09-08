"""一键训练/推理：修改下方配置后直接运行 python run.py。

也可临时覆盖模式：
    python run.py --mode train
    python run.py --mode predict

首次使用安装依赖：python -m pip install -e '.[cad,gui]'
训练默认打开 PyQt 窗口；无界面运行使用 --headless。
所有相对路径均相对于本文件，支持从 IDE 或其他工作目录启动。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# ==================== 1. 运行模式（直接使用 UWPT 算例） ====================
MODE = "train"                  # "train" 训练；"predict" 推理


# ==================== 2. 输入输出路径 ====================
# 可填写相对于 run.py 的路径，也可直接写绝对路径。
FILES = {
    "model": "results/uwpt/model.npz",          # 训练保存 / 推理加载
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",           # 实际配置和训练报告所在目录
    "resume_model": None,                       # 可选：已有 .npz，继续训练
}
# 继续训练使用文件内保存的物理模型/空间基/网络，以及下方当前 TRAINING；
# 此时不会重新生成网格，几何/材料等构建参数不参与本次继续训练。


# ==================== 3. UWPT 几何与网格（长度单位 m） ====================
MESH = {
    "generate": True,                   # True：每次新训练按配置生成；False：导入已有网格
    "path": "results/uwpt/uwpt.msh",     # Gmsh 2.2 ASCII 四面体网格
    "geometry_tolerance": 0.0005,        # 中心线折线弦误差
    "mesh_size": 0.01,
}
# 发射/接收线圈可独立配置。角度为弧度，依次绕 x/y/z 旋转。
TRANSMITTER = {
    "shape": "circle",                  # "circle" / "rounded_square"
    "turns": 0.5,
    "outer_half_size": 0.015,
    "pitch": 0.002,
    "conductor_width": 0.001,
    "conductor_thickness": 0.001,
    "corner_radius": 0.006,             # 仅 rounded_square 使用
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


# 几何参数族：一次训练覆盖整个连续参数盒；推理输入保存模型内的这些参数。
# 形状、匝数、材料拓扑固定。planar_scale 同比例改变外径、匝距和导体宽度；
# thickness_scale 独立改变厚度；package_scale 仅改变封装外表面尺寸。
# 位移/间距单位 m，seawater_radius 是实际网格外球半径（包含几何容差）。
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
    "em_anchor_count": 4,       # 额外 Halton 几何锚点；另含中心、上下角点及各轴端点
    "cache_size": 128,          # 缓存几何物理算子，减少残差训练中的重复组装
}


# ==================== 4. UWPT 材料与电磁参数（SI 单位） ====================
PHYSICS = {
    "frequency_hz": 100000.0,
    "ambient_temperature": 293.15,      # K，固定环境/边界参考温度
    "constitutive_relative_error": 1e-8,
    "em_energy_error": 1e-6,            # 单位端口源的 EM 降阶误差目标
}
# 键是网格的体物理标签；修改 PHYSICAL_TAGS 后也应对应修改这里及 PORTS。
# 电导率 S/m；电阻率温度系数 1/K；热导率 W/(m K)；体积热容量 J/(m³ K)。
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
    "current_offset": None,             # I = I0 + C U；None 表示零偏置
    "current_matrix": None,             # None 表示单位阵，即 I=U
}
# 电流采用峰值相量 A；复系数写为字符串，例如 [["1", "0"], ["0", "1j"]]。
EM_CANDIDATE_STATES = [
    [-0.1, -0.1], [-0.1, 0.1], [0.1, -0.1], [0.1, 0.1], [0.0, 0.0],
]  # 热坐标维数须等于保留热秩；None 表示由训练域上下界及中点生成


# ==================== 5. UWPT 热空间截断 ====================
THERMAL_RANK = 2                     # 数值截断阶数；None 表示全空间或下方证书选秩
THERMAL_TRUNCATION = {
    "initial_temperature_deviation_free": None,
    "source_dual_bound": None,
    "requested_state_tolerance": None,
    "prefer_partial_thermal_spectrum": True,
}
# 常规运行仅修改 THERMAL_RANK。若使用原有证书选秩，将其设为 None，
# 同时填入以上前三项：自由节点初温偏差数组、热源对偶范数界和状态误差目标。


# ==================== 6. 训练配置 ====================
# initial_* 是质量正交热模态坐标，不是摄氏温度；长度等于热秩。
# operating_* 对应实工况 U；time_horizon 单位 s。
# residual_tolerance 是模态方程残差目标，不是温度误差目标。
TRAINING = {
    "initial_lower": [-0.1, -0.1], "initial_upper": [0.1, 0.1],
    "operating_lower": [0.0, 0.0], "operating_upper": [10.0, 10.0],
    "time_horizon": 100000.0, "residual_tolerance": 1e-5,
    "time_sampling": "mixed_log", "time_min": 1e-6, "include_steady_state": True,
    "sample_count": 64, "validation_count": 64,
    "max_nodes": 48, "max_degree": 3,
    "max_parent_responses": 1, "max_realization_dimension": 64,
}
# validation_count 仅为无标签物理残差的独立输入检查点数量，不做瞬态参考积分。


# ==================== 7. 推理配置 ====================
PREDICTION = {
    "a0": [0.0, 0.0], "operating": [5.0, 0.0],
    "times": [0.0, 0.001, 1.0, 1000.0, 100000.0, 1000000.0, "inf"],
    "geometry": None,                   # None：保存几何域的中心；或填写完整参数字典
    "allow_time_extrapolation": True,   # 时间可超出训练窗；"inf" 查询解析稳态极限
    "initial_temperature_file": None,
    "state_only": False, "allow_extrapolation": False,
}
# initial_temperature_file：可选的一维 .npy 全节点开尔文温度；设置后投影得到 a0。
# state_only=True：仅解析网络与温度重构；False：另输出阻抗、电感和材料损耗。
# allow_extrapolation 仅放开初态/电流域；几何必须处于保存的有效网格参数域。
# 任意非负有限时间及 "inf" 均可查询；域外时间精度不由有限残差检查保证。
# 推理始终使用 .npz 内保存的物理参数，不使用本文件的几何/材料构建配置。


# ==================== 8. PyQt 实时训练窗口与日志 ====================
MONITOR = {
    "enabled": True,                    # 训练打开窗口；推理不受影响
    "auto_start": False,                # False：窗口打开后点击“启动”
    "log_dir": "results/uwpt/logs",      # 每次任务独立子目录，保留全部 JSONL/文本日志
    "log_interval_s": 1.0,              # 训练侧周期写日志，接受更新时也立即记录
    "refresh_ms": 300,                  # 日志读取线程轮询周期
    "max_plot_points": 4000,            # 窗口最多保留的曲线点数，日志不截断
    "compute_threads": 1,               # 后台进程 BLAS/OpenMP 线程数
}
# 窗口主线程只绘图；QThread 读日志；独立进程训练，其日志线程周期写盘。
# 暂停/停止在当前不可拆分运算结束后的检查点生效。暂停不退出进程，恢复继续原任务。
# 停止后保存 model.stopped.npz（若已有有效网络），不会覆盖上次正常完成的模型。


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
        np.asarray(ENVIRONMENT["package_half_extent"]), ENVIRONMENT["seawater_padding"],
    )
    result = mesh_underwater_wpt_geometry(
        geometry, path, geometry_tolerance=MESH["geometry_tolerance"],
        mesh_size=MESH["mesh_size"], physical_tags=UWPTPhysicalTags(**PHYSICAL_TAGS),
    )
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
    else:
        mesh_path = resolve_path(MESH["path"])
        physical = {
            **PHYSICS, **PORTS, "mesh": str(mesh_path), "materials": MATERIALS,
            "thermal_rank": THERMAL_RANK, "thermal_truncation": THERMAL_TRUNCATION,
            "training": TRAINING,
        }
        if GEOMETRY_FAMILY["enabled"]:
            physical["geometry_family"] = {**GEOMETRY_FAMILY, "transmitter": TRANSMITTER,
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
        # 由本文件自动生成底层入口所需配置，无需另行维护 JSON。
        config_path = settings_dir / "model.config.json"
        write_json(config_path, physical)
        print("组装物理模型并构建电磁降阶空间……", flush=True)
        if monitor is not None:
            monitor.phase("assembly")
        if GEOMETRY_FAMILY["enabled"]:
            from sdfmpneo.geometry_research import geometry_model_from_config
            model, config = geometry_model_from_config(config_path, monitor=monitor)
            write_json(settings_dir / "geometry.domain.json", {
                "names": model.geometry_names, "reference": model.geometry_reference,
                "lower": model.lower, "upper": model.upper,
                "mesh_certificate": model.certificate, "em_basis": model.em_basis_report,
            })
            print(f"共享几何代理：{len(model.geometry_names)} 个几何输入，EM 基维数={model.reference.em.n_reduced}", flush=True)
        else:
            model, config = model_from_config(config_path)
    write_json(settings_dir / "train.settings.json", settings)
    print("开始无解标签残差训练……", flush=True)
    from sdfmpneo.training.monitor import TrainingStopped
    try:
        report = model.train(config, monitor=monitor, progress=lambda n, r, m: print(
            f"响应节点={n}  RMS残差={r:.6g}  最大残差={m:.6g}", flush=True))
    except TrainingStopped:
        checkpoint = model_path.with_name(model_path.stem+".stopped"+model_path.suffix)
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
    # Saving is allowed to finish even if stop arrives after training completed.
    if monitor is not None:
        monitor.phase("saving", check=False)
    model.save(model_path)
    write_json(settings_dir / "training.report.json", report)
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
    geometry_args = {}
    if hasattr(model, "geometry_names"):
        g = parameters.get("geometry")
        if g is None:
            g = dict(zip(model.geometry_names, (model.lower+model.upper)/2))
        parameters["geometry"] = g
        geometry_args = {"geometry": g}
    temperature_file = parameters["initial_temperature_file"]
    if temperature_file is not None:
        temperature_path = resolve_path(temperature_file)
        initial = model.project_initial_temperature(np.load(temperature_path, allow_pickle=False), **geometry_args)
        parameters["initial_temperature_file"] = str(temperature_path)
    if not parameters["times"]:
        raise ValueError("推理 times 至少需要一个时间点")
    print(f"加载模型推理：{model_path}", flush=True)
    results = [model.predict(
        t, a0=initial, operating=parameters["operating"],
        diagnostics=not parameters["state_only"], allow_extrapolation=parameters["allow_extrapolation"],
        allow_time_extrapolation=parameters.get("allow_time_extrapolation", True), **geometry_args,
    ) for t in parameters["times"]]
    write_json(output_path, results)
    write_json(settings_dir / "predict.settings.json", {
        "case": "uwpt", "mode": "predict", "model": str(model_path),
        "predictions": str(output_path), "prediction": parameters, "effective_a0": initial,
    })
    for result in results:
        if isinstance(result, dict):
            time, maximum = result["time"], result["maximum_temperature"]
        else:
            time, maximum = result.time, result.maximum_temperature
        print(f"t={float(time):g} s，最高温度={maximum:.8g} K")
    print(f"推理结果已保存：{output_path}")
    return 0


CONFIG_NAMES = ("FILES", "MESH", "TRANSMITTER", "RECEIVER", "ENVIRONMENT", "PHYSICAL_TAGS",
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
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8]
        session_dir = resolve_path(MONITOR["log_dir"])/stamp
    session_dir = Path(session_dir)
    write_json(session_dir/"settings.json", configuration_snapshot(model_path))
    print(f"训练日志：{session_dir}", flush=True)
    with TrainingMonitor(session_dir/"metrics.jsonl", session_dir/"control.json",
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
    parser = argparse.ArgumentParser(description="一键运行电磁–热代理；参数集中在 run.py 顶部。")
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
    if args.mode not in ("train", "predict"):
        parser.error("MODE 应为 train/predict")
    model_path = resolve_path(args.model or FILES["model"])
    settings_dir = resolve_path(FILES["settings_dir"])
    if args.mode == "train":
        if not args.headless and (args.gui or MONITOR["enabled"]):
            try:
                from sdfmpneo.training.qt_monitor import launch_window
            except ImportError as exc:
                raise SystemExit('请安装图形依赖：python -m pip install -e ".[cad,gui]"；'
                                 '或使用 --headless 仅训练并记录日志。') from exc
            return launch_window(__file__, configuration_snapshot(model_path),
                                 resolve_path(MONITOR["log_dir"]), MONITOR)
        return execute_training(model_path, settings_dir)
    return predict(model_path, resolve_path(FILES["predictions"]), settings_dir)


if __name__ == "__main__":
    raise SystemExit(main())
