"""一键训练/推理：修改下方配置后直接运行 python run.py。

训练: python run.py --mode train
无界面: python run.py --mode train --headless
推理: python run.py --mode predict
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

MODE = "train"
FILES = {
    "model": "results/uwpt/model.npz",
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",
    "resume_model": None,
}
MESH = {
    "generate": True,
    "path": "results/uwpt/uwpt.msh",
    "geometry_tolerance": 0.0005,
    "mesh_size": 0.01,
}
TRANSMITTER = {
    "shape": "circle", "turns": 0.5, "outer_half_size": 0.015,
    "pitch": 0.002, "conductor_width": 0.001, "conductor_thickness": 0.001,
    "corner_radius": 0.006, "translation": [0.0, 0.0, 0.0], "angles": [0.0, 0.0, 0.0],
}
RECEIVER = {
    "shape": "circle", "turns": 0.5, "outer_half_size": 0.015,
    "pitch": 0.002, "conductor_width": 0.001, "conductor_thickness": 0.001,
    "corner_radius": 0.006, "translation": [0.0, 0.0, 0.01], "angles": [0.0, 0.0, 0.0],
}
ENVIRONMENT = {"package_half_extent": [0.019, 0.019, 0.003], "seawater_padding": 0.006}
PHYSICAL_TAGS = {
    "tx_copper": 101, "rx_copper": 102, "tx_package": 201, "rx_package": 202,
    "seawater": 301, "tx_terminal_start": 1001, "tx_terminal_end": 1002,
    "rx_terminal_start": 1003, "rx_terminal_end": 1004, "outer_boundary": 2001,
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
PHYSICS = {
    "frequency_hz": 100000.0,
    "ambient_temperature": 293.15,
    "constitutive_relative_error": 1e-8,
    "em_energy_error": 1e-6,
}
MATERIALS = {
    "101": {"name": "tx_copper", "electrical_conductivity": 5.8e7, "resistivity_temperature_coefficient": 0.00393, "reference_temperature": 293.15, "relative_permeability": 1.0, "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6},
    "102": {"name": "rx_copper", "electrical_conductivity": 5.8e7, "resistivity_temperature_coefficient": 0.00393, "reference_temperature": 293.15, "relative_permeability": 1.0, "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6},
    "201": {"name": "tx_package", "electrical_conductivity": 0.0, "resistivity_temperature_coefficient": 0.0, "reference_temperature": 293.15, "relative_permeability": 1.0, "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6},
    "202": {"name": "rx_package", "electrical_conductivity": 0.0, "resistivity_temperature_coefficient": 0.0, "reference_temperature": 293.15, "relative_permeability": 1.0, "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6},
    "301": {"name": "seawater", "electrical_conductivity": 5.0, "resistivity_temperature_coefficient": 0.0, "reference_temperature": 293.15, "relative_permeability": 1.0, "thermal_conductivity": 0.6, "volumetric_heat_capacity": 4.1e6},
}
PORTS = {
    "terminal_pairs": [[1001, 1002], [1003, 1004]],
    "port_names": ["tx", "rx"],
    "current_offset": None,
    "current_matrix": None,
}
EM_CANDIDATE_STATES = None

THERMAL_RANK = None
THERMAL_TRUNCATION = {
    "mode": "automatic_physics_envelope",
    "relative_tolerance": 1e-3,
    "absolute_tolerance": 0.0,
    "initial_coordinate_bound": 0.1,
    "probe_start_rank": 4,
    "source_bound_safety_factor": 2.0,
    "restart_state_safety_factor": 1.5,
    "boundary_fraction": 0.25,
    "temperature_probe_axes": 4,
    "initial_temperature_deviation_free": None,
    "source_dual_bound": None,
    "requested_state_tolerance": None,
    "prefer_partial_thermal_spectrum": True,
}

MAX_RESPONSE_TIME = 100.0
TRAINING = {
    "initial_lower": [],
    "initial_upper": [],
    "operating_lower": [0.0, 0.0],
    "operating_upper": [10.0, 10.0],
    "max_response_time": MAX_RESPONSE_TIME,
    "residual_tolerance": 1e-5,
    "time_sampling": "mixed_log",
    "time_min": 1e-6,
    "sample_count": 64,
    "validation_count": 64,
    "semigroup_sample_count": 8,
    "semigroup_validation_count": 8,
    # Stage 0: fit the t=0 physical source before finite-time residual training.
    "source_prefit_count": 64,
    # Restart seeds live in a low-dimensional ellipsoid; the full modal bounds
    # remain safety limits instead of a 198-D Cartesian training box.
    "initial_training_rank": 16,
    # Multilayer analytic response funnel. None keeps rank-aware defaults:
    # for rank>=96 this is depth=3, widths r -> 64 -> 32.
    "max_network_depth": 3,
    "max_channels_per_mode": 1,
    "network_layer_widths": None,
    "network_hidden_ranks": None,
    "network_cross_ranks": None,
    "network_state_ranks": None,
    "max_linear_rank": None,
    "max_quadratic_rank": None,
    "max_square_rank": None,
    "max_hidden_rank": None,
    "max_cross_rank": None,
    "max_state_rank": None,
    "state_feature_term_budget": 24,
    # Inexact layer Jacobians propose directions; exact physics residuals decide
    # every backtracking step and damping contraction.
    "jacobian_point_budget": 12,
    "semigroup_jacobian_point_budget": 4,
    "max_iterations": 36,
    "max_iterations_per_layer": 12,
    "max_damping_retries": 4,
    "max_backtracks": 6,
    "backtrack_factor": 0.5,
    "max_validation_epochs": 5,
    # Gates stay frozen at one during training. Shrink/prune is post-convergence.
    "gate_shrink": 0.0,
    "prune_relative_budget": 0.10,
    "prune_rounds": 3,
}
PREDICTION = {
    "a0": None,
    "operating": [5.0, 0.0],
    "times": [0.0, 1.0, 100.0, 350.0, "inf"],
    "geometry": None,
    "initial_temperature_file": None,
    "state_only": False,
    "allow_extrapolation": False,
}
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

ROOT = Path(__file__).resolve().parent


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    from sdfmpneo.__main__ import jsonable
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_training_rank(config, rank, saved_config=None):
    rank = int(rank)
    if len(config.initial_lower) == len(config.initial_upper) == 0:
        if saved_config is not None and len(saved_config.initial_lower) == rank:
            return replace(
                config,
                initial_lower=tuple(saved_config.initial_lower),
                initial_upper=tuple(saved_config.initial_upper),
            )
        bound = float(THERMAL_TRUNCATION.get("initial_coordinate_bound", 0.1))
        return replace(config, initial_lower=(-bound,) * rank, initial_upper=(bound,) * rank)
    if len(config.initial_lower) != rank or len(config.initial_upper) != rank:
        raise ValueError(
            f"训练初态维数与 thermal rank={rank} 不一致；自动模式请将 initial_lower/upper 留空"
        )
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
    return rank, dict(report)


def add_horizon_rank_diagnostic(model, report):
    from sdfmpneo.training.thermal_horizon import diagnostic_from_rank_report

    owner = thermal_rank_owner(model)
    updated = dict(report)
    updated["finite_horizon_diagnostic"] = diagnostic_from_rank_report(
        owner.core, updated, horizons=(1.0, 10.0, 30.0, 100.0)
    )
    return updated


@contextmanager
def assembly_progress(monitor=None):
    interval = float(MONITOR.get("assembly_progress_interval_s", 5.0))
    if not 0 < interval < float("inf"):
        raise ValueError("assembly_progress_interval_s 必须为有限正数")
    stop = threading.Event()
    started = time.monotonic()
    labels = {
        "assembly": "组装物理模型与参考电磁空间",
        "thermal_rank_selection": "自动选择热空间阶数",
        "thermal_rank_cache_hit": "命中自动热秩缓存",
        "geometry_em_basis": "构建跨几何共享电磁空间",
    }

    def current_label():
        if monitor is None:
            return labels["assembly"]
        phase = monitor.data.get("phase", "assembly")
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
    import numpy as np

    print("\n训练样本参数范围：", flush=True)
    if hasattr(model, "geometry_names"):
        print("  几何参数 G（网络内部归一化到 [-1,1]）：", flush=True)
        for name, lower, upper, reference in zip(
            model.geometry_names, model.lower, model.upper, model.geometry_reference
        ):
            print(f"    {name}: [{float(lower):.8g}, {float(upper):.8g}]  参考值={float(reference):.8g}", flush=True)
    else:
        print("  几何参数 G：固定几何", flush=True)
    lower = np.asarray(config.initial_lower, float)
    upper = np.asarray(config.initial_upper, float)
    active = config.initial_active_indices()
    print(
        f"  restart 热坐标安全边界：{len(lower)} 维；训练种子只参数化 {len(active)} 个慢/大幅模态，"
        f"索引={active.tolist()}",
        flush=True,
    )
    if len(lower):
        print(
            f"    全模态下界范围=[{float(np.min(lower)):.6g},{float(np.max(lower)):.6g}]；"
            f"上界范围=[{float(np.min(upper)):.6g},{float(np.max(upper)):.6g}]",
            flush=True,
        )
    print("  工况参数 U：", flush=True)
    for i, (lo, hi) in enumerate(zip(config.operating_lower, config.operating_upper)):
        print(f"    U[{i}]: [{lo:.8g}, {hi:.8g}]", flush=True)
    print(
        f"  单段时间: [0,{config.max_response_time:.8g}] s；采样={config.time_sampling}",
        flush=True,
    )
    print(
        f"  source prefit={config.source_prefit_count}；physics 训练/验证={config.sample_count}/{config.validation_count}；"
        f"restart 训练/验证={config.semigroup_sample_count}/{config.semigroup_validation_count}；"
        f"残差目标={config.residual_tolerance:.8g}",
        flush=True,
    )
    print(
        f"  响应网络：depth={config.max_network_depth or 'auto'}；"
        f"layer widths={config.network_layer_widths or 'rank-aware auto (高秩默认 r→64→32)'}；"
        f"每层最大迭代={config.max_iterations_per_layer}；backtracks={config.max_backtracks}\n",
        flush=True,
    )


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
        np.asarray(ENVIRONMENT["package_half_extent"]),
        ENVIRONMENT["seawater_padding"],
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


def train(model_path, settings_dir, monitor=None):
    from sdfmpneo import ResearchElectroThermalModel, ResearchTrainingConfig
    from sdfmpneo.research import model_from_config

    config = ResearchTrainingConfig(**TRAINING)
    resume = FILES["resume_model"]
    settings = {
        "case": "uwpt", "mode": "train", "model": str(model_path),
        "training": TRAINING, "resume_model": None,
    }
    if resume is not None:
        if monitor is not None:
            monitor.phase("loading")
        resume_path = resolve_path(resume)
        settings["resume_model"] = str(resume_path)
        print(f"加载 segmented fixed-network 模型继续训练：{resume_path}", flush=True)
        model = ResearchElectroThermalModel.load(resume_path)
        rank, _ = thermal_rank_summary(model)
        config = resolve_training_rank(config, rank, getattr(model, "training_config", None))
        if model.network.max_response_time != config.max_response_time:
            raise ValueError("继续训练时 MAX_RESPONSE_TIME 必须与已保存网络一致")
        if model.network.depth != int(config.max_network_depth or model.network.depth):
            print(
                "注意：resume checkpoint 保留其原有响应网络深度；要使用新的三层漏斗结构请将 FILES['resume_model']=None。",
                flush=True,
            )
    else:
        mesh_path = resolve_path(MESH["path"])
        physical = {
            **PHYSICS, **PORTS,
            "mesh": str(mesh_path),
            "materials": MATERIALS,
            "thermal_rank": THERMAL_RANK,
            "thermal_truncation": THERMAL_TRUNCATION,
            "training": TRAINING,
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
        settings.update(
            physics=physical, mesh=MESH, transmitter=TRANSMITTER,
            receiver=RECEIVER, environment=ENVIRONMENT, physical_tags=PHYSICAL_TAGS,
        )
        if MESH["generate"]:
            if monitor is not None:
                monitor.phase("mesh")
            print("生成线圈、封装和海水网格……", flush=True)
            generate_mesh(mesh_path)
        elif not mesh_path.is_file():
            raise FileNotFoundError(f"网格不存在：{mesh_path}")
        config_path = settings_dir / "model.config.json"
        write_json(config_path, physical)
        if monitor is not None:
            monitor.phase("assembly")
        print("自动热秩、物理组装与电磁降阶……", flush=True)
        with assembly_progress(monitor):
            if GEOMETRY_FAMILY["enabled"]:
                from sdfmpneo.geometry_research import geometry_model_from_config
                model, config = geometry_model_from_config(config_path, monitor=monitor)
            else:
                model, config = model_from_config(config_path, monitor=monitor)
        if GEOMETRY_FAMILY["enabled"]:
            write_json(
                settings_dir / "geometry.domain.json",
                {
                    "names": model.geometry_names,
                    "reference": model.geometry_reference,
                    "lower": model.lower,
                    "upper": model.upper,
                    "mesh_certificate": model.certificate,
                    "em_basis": model.em_basis_report,
                },
            )
    rank, rank_report = thermal_rank_summary(model)
    rank_report = add_horizon_rank_diagnostic(model, rank_report)
    config = resolve_training_rank(config, rank, getattr(model, "training_config", None))
    settings["selected_thermal_rank"] = rank
    settings["training_resolved"] = config
    write_json(settings_dir / "thermal.rank.json", rank_report)
    write_json(settings_dir / "train.settings.json", settings)
    print(f"最终热空间阶数：rank={rank}；选择方法={rank_report.get('method','unknown')}", flush=True)
    horizon_diag = rank_report.get("finite_horizon_diagnostic", {})
    if horizon_diag.get("available"):
        ranks = horizon_diag["horizons"]
        print(
            "有限时间热秩诊断（仅诊断，生产选秩仍采用稳态包络）："
            + "，".join(f"{name}→{value['rank']}" for name, value in ranks.items()),
            flush=True,
        )
    print_training_sample_ranges(model, config)
    print("开始多层 finite-horizon analytic-response 残差训练……", flush=True)
    from sdfmpneo.training.monitor import TrainingStopped
    try:
        report = model.train(
            config,
            monitor=monitor,
            progress=lambda n, r, m: print(
                f"accepted revision={n}  RMS联合残差={r:.6g}  最大联合残差={m:.6g}", flush=True
            ),
        )
    except TrainingStopped:
        checkpoint = model_path.with_name(model_path.stem + ".stopped" + model_path.suffix)
        if model.network is not None:
            model.save(checkpoint)
            print(f"训练已停止，当前网络已保存：{checkpoint}", flush=True)
        else:
            checkpoint = None
        stopped = {
            "status": "stopped",
            "checkpoint": None if checkpoint is None else str(checkpoint),
            "numerical_tolerance_met": False,
        }
        write_json(settings_dir / "training.stopped.json", stopped)
        if monitor is not None:
            monitor.finish("stopped", **stopped)
        return 130
    if monitor is not None:
        monitor.phase("saving", check=False)
    model.save(model_path)
    write_json(settings_dir / "training.report.json", report)
    structure = model.network.structure_summary(0.0)
    write_json(settings_dir / "network.structure.json", structure)
    print("最终网络有效结构：" + json.dumps(structure, ensure_ascii=False), flush=True)
    if report.source_prefit_rms_residual is not None:
        print(
            f"source prefit：RMS={report.source_prefit_rms_residual:.6g}；"
            f"max={report.source_prefit_max_residual:.6g}",
            flush=True,
        )
    if report.validation_performed:
        print(
            f"训练状态：{report.status}；训练 physics max={report.maximum_training_physics_residual:.6g}；"
            f"独立验证 physics max={report.maximum_validation_physics_residual:.6g}；"
            f"restart-rate max={report.maximum_validation_semigroup_rate_defect:.6g}；"
            f"实际训练响应深度={report.trained_response_depth}",
            flush=True,
        )
    else:
        print(
            f"训练状态：{report.status}；训练 physics max={report.maximum_training_physics_residual:.6g}；"
            "尚未达到进入独立验证的条件，因此未执行独立 validation；"
            f"实际训练响应深度={report.trained_response_depth}",
            flush=True,
        )
    print(f"模型已保存：{model_path}")
    if monitor is not None:
        monitor.finish(
            "completed" if report.numerical_tolerance_met else report.status,
            model=str(model_path),
            numerical_tolerance_met=report.numerical_tolerance_met,
            validation_performed=report.validation_performed,
            trained_response_depth=report.trained_response_depth,
        )
    return 0 if report.numerical_tolerance_met else 2


def predict(model_path, output_path, settings_dir):
    import numpy as np
    from sdfmpneo import ResearchElectroThermalModel

    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在：{model_path}；请先训练")
    model = ResearchElectroThermalModel.load(model_path)
    parameters = dict(PREDICTION)
    initial = parameters["a0"]
    if initial is None:
        initial = np.zeros(model.network.n_modes, dtype=float)
    geometry_args = {}
    if hasattr(model, "geometry_names"):
        g = parameters.get("geometry")
        if g is None:
            g = dict(zip(model.geometry_names, (model.lower + model.upper) / 2))
        parameters["geometry"] = g
        geometry_args = {"geometry": g}
    if parameters["initial_temperature_file"] is not None:
        temperature_path = resolve_path(parameters["initial_temperature_file"])
        initial = model.project_initial_temperature(
            np.load(temperature_path, allow_pickle=False), **geometry_args
        )
        parameters["initial_temperature_file"] = str(temperature_path)
    if not parameters["times"]:
        raise ValueError("推理 times 至少需要一个时间点")
    print(
        f"加载模型推理：{model_path}；thermal rank={model.network.n_modes}；"
        f"response depth={model.network.depth}；单段上限={model.network.max_response_time:g} s",
        flush=True,
    )
    results = [
        model.predict(
            t,
            a0=initial,
            operating=parameters["operating"],
            diagnostics=not parameters["state_only"],
            allow_extrapolation=parameters["allow_extrapolation"],
            **geometry_args,
        )
        for t in parameters["times"]
    ]
    write_json(output_path, results)
    write_json(
        settings_dir / "predict.settings.json",
        {
            "case": "uwpt", "mode": "predict", "model": str(model_path),
            "predictions": str(output_path), "prediction": parameters,
            "effective_a0": initial,
        },
    )
    for result in results:
        if result.get("steady_state"):
            label = "steady"
        else:
            label = f"{float(result['time']):g} s"
        print(
            f"t={label}，最高温度={result['maximum_temperature']:.8g} K，"
            f"segments={result.get('segment_count', 0)}",
            flush=True,
        )
    print(f"推理结果已保存：{output_path}")
    return 0


CONFIG_NAMES = (
    "FILES", "MESH", "TRANSMITTER", "RECEIVER", "ENVIRONMENT", "PHYSICAL_TAGS",
    "PHYSICS", "MATERIALS", "PORTS", "EM_CANDIDATE_STATES", "THERMAL_RANK",
    "THERMAL_TRUNCATION", "MAX_RESPONSE_TIME", "TRAINING", "PREDICTION", "MONITOR",
    "GEOMETRY_FAMILY",
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
    from sdfmpneo.training.monitor import TrainingMonitor, TrainingStopped

    if session_dir is None:
        session_dir = resolve_path(MONITOR["log_dir"]) / (
            datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        )
    session_dir = Path(session_dir)
    write_json(session_dir / "settings.json", configuration_snapshot(model_path))
    print(f"训练日志：{session_dir}", flush=True)
    with TrainingMonitor(
        session_dir / "metrics.jsonl",
        session_dir / "control.json",
        interval=MONITOR["log_interval_s"],
    ) as monitor:
        try:
            return train(model_path, settings_dir, monitor)
        except TrainingStopped:
            monitor.finish("stopped", checkpoint=None, message="模型构建阶段停止，暂无网络检查点")
            return 130


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
    parser = argparse.ArgumentParser(description="SDF-MPNEO multilayer segmented analytic response network")
    parser.add_argument("--mode", choices=("train", "predict"), default=MODE)
    parser.add_argument("--model")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--gui", action="store_true")
    display.add_argument("--headless", action="store_true")
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
                raise SystemExit(
                    '请安装图形依赖：python -m pip install -e ".[cad,gui]"；或使用 --headless。'
                ) from exc
            return launch_window(
                __file__, configuration_snapshot(model_path),
                resolve_path(MONITOR["log_dir"]), MONITOR,
            )
        return execute_training(model_path, settings_dir)
    return predict(model_path, resolve_path(FILES["predictions"]), settings_dir)


if __name__ == "__main__":
    raise SystemExit(main())
