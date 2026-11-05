"""SDF-MPNEO vNext 一键入口。

正常使用只需要修改本文件的 CONFIG，然后运行：

    python run.py

默认打开非阻塞 PyQt 训练界面。也可以：

    python run.py --mode train     # 无界面训练（历史 vNext）
    python run.py --mode bfzi      # BFZI-MQS 海水、圆截面铜损与电热示例
    python run.py --mode bfzi --green-ball # 完整均匀球Green电磁热示例
    python run.py --mode bfzi-snapshot --out bfzi_response.npz # 双端口低秩保存加载推理
    python run.py --mode bfzi-generate --samples 12 --out bfzi_geometry_labels.npz # 跨几何样本
    python run.py --mode bfzi-train --dataset bfzi_geometry_labels.npz --model bfzi_neural.pt # 实验网络训练
    python run.py --mode bfzi-infer --model bfzi_neural.pt # 推理
    python run.py --mode bfzi-eval --dataset bfzi_geometry_labels.npz --model bfzi_neural.pt # 独立几何误差验证
    python run.py --mode bfzi-p0-source --with-tests # P0：锁定物理源，解析直线 + 高阶角段 C/B 与线圈离散收敛（自动上传）
    python run.py --mode bfzi-p1-coulomb --with-tests # P1：原核 1/r 磁能解析真值（无散度球电流）
    python run.py --mode bfzi-p1-full-sea --with-tests # P1-C：仅运行真实双封装全海水 curl L 分轴收敛（独立自动日志）
    python run.py --mode bfzi-g1-foundations --with-tests # G1：裁剪局部Green积分+层级curl构造+多极外域解析对照（自动上传）
    python run.py --mode bfzi-g1-foundations --g1-hierarchy # 可选昂贵的5→11→17层级J/q试验（尚非精度证书）
    python run.py --mode bfzi-g1-foundations --g1-source-panels # 可选昂贵的15万节点源坐标切向实验（默认跳过）
    python run.py --mode bfzi-convergence  # BFZI 海水数值收敛分轴诊断
    python run.py --mode bfzi-convergence --basis-green  # Green L/软化单轴诊断
    python run.py --mode bfzi-convergence --basis-green-refined # 高精度R/K固定，独立Green L/软化研究
    python run.py --mode bfzi-convergence --basis-green-cross   # 双套独立磁节点交叉Green对照
    python run.py --mode bfzi-convergence --basis-green-gaussian # 高斯Green A/B和宽度敏感性诊断
    python run.py --mode bfzi-convergence --basis-green-ball # 均匀实体球解析Coulomb Green对照
    python run.py --mode bfzi-convergence --basis-ball # 均匀球Green嵌套模态检查
    python run.py --mode bfzi-convergence --basis-ball-check-l # 159模态磁积分A/B独立验证
    python run.py --mode bfzi-convergence --basis-ball-face # 159→167 完整端口源引导面电流基
    python run.py --mode bfzi-convergence --basis-ball-face --basis-ball-check-l # 167模态磁积分A/B
    python run.py --mode bfzi-convergence --basis-ball-edge # 167→199 近边缘方向扩展；昂贵
    python run.py --mode bfzi-convergence --basis-ball-edge --basis-ball-check-l # 199模态磁积分A/B；更昂贵
    python run.py --mode bfzi-convergence --basis-ball-aniso # 199→207 旋转各向异性面法向电流响应
    python run.py --mode bfzi-convergence --basis-ball-aniso --basis-ball-check-rk # 207模态独立法向 R/K 积分对照
    python run.py --mode bfzi-convergence --basis-ball-targeted # 默认精简终端输出、保存并推送最新日志
    python run.py --mode bfzi-convergence --basis-ball-targeted --with-tests # 同时运行 BFZI 测试并保存结果
    python run.py --mode bfzi-convergence --basis-ball-targeted --basis-ball-check-far --with-tests # 两种独立远场角度R/K对照，自动上传
    python run.py --mode bfzi-convergence --basis-ball-targeted --basis-ball-check-far-refined --with-tests # 当前三阶精细基准再比较4/5阶，径向独立检验
    python run.py --mode bfzi-convergence --basis-ball-targeted --verbose-log # 终端也显示原始详细输出
    python run.py --mode bfzi-convergence --basis-ball-targeted --no-publish-log # 只写本地，不提交 Git
    python run.py --mode bfzi-convergence --basis-ball-aniso --basis-ball-ablate # 207模态父空间误差上界与重要方向筛选
    python run.py --mode bfzi-convergence --basis-domain-ball # 均匀球Green外域检查
    python run.py --mode bfzi-convergence --basis-domain-ball --basis-ball-targeted --with-tests # 已测：229模态外域0.52→0.70→0.85m均FAIL
    python run.py --mode bfzi-convergence --basis-domain-ball --basis-ball-targeted --basis-domain-extend --with-tests # 已测：0.85→1.05→1.30m均FAIL
    python run.py --mode bfzi-convergence --basis-domain-ball --basis-ball-targeted --basis-domain-far --with-tests # 新：1.05→1.30→1.60m及R/K与Green L敏感性拆分
    python run.py --mode bfzi-convergence --basis-domain # 外域半径收敛
    python run.py --mode bfzi-convergence --basis-local  # R/K 分轴与组合加密
    python run.py --mode bfzi-convergence --basis-angular  # 球域角向分片 Gauss 积分
    python run.py --mode bfzi-convergence --basis-radial   # 真实几何径向分片及两种 K 弱式对照
    python run.py --mode bfzi-convergence --basis-kparts   # 球域与近壁分区的 K 弱式积分误差
    python run.py --fresh          # 忽略模型 checkpoint，但继续复用 teacher 缓存

所有面向用户的训练、缓存、采样、teacher、CUDA/CPU、GUI、checkpoint
和输出路径配置均集中在本文件；数值实现模块不固化具体实验参数。
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


CONFIG = {
    # ------------------------------------------------------------------
    # 总运行模式
    # ------------------------------------------------------------------
    "ROOT": str(ROOT),
    "RUN": {
        # gui: PyQt 控制台；train: 当前终端直接训练。
        "mode": "gui",
    },

    # ------------------------------------------------------------------
    # 文件与目录
    # ------------------------------------------------------------------
    "FILES": {
        "log_dir": "results/vnext/logs",
        "checkpoint_dir": "results/vnext/checkpoints",
        "port_checkpoint": "tensor_port.training.pt",
        "spatial_checkpoint": "tensor_spatial.training.pt",
        "port_artifact": "results/vnext/tensor_port.pt",
        "spatial_artifact": "results/vnext/tensor_spatial.pt",
        "bundle": "results/vnext/tensor_bundle",
        "summary": "results/vnext/training_summary.json",
        "overwrite_bundle": True,
    },

    # ------------------------------------------------------------------
    # Teacher 数据缓存
    # ------------------------------------------------------------------
    "CACHE": {
        "root": "results/vnext/teacher_cache",
        # reuse: 命中即复用，只补缺失样本；
        # refresh: 删除当前参数空间对应缓存并重新生成；
        # readonly: 只允许读取，缺样本直接报错。
        "policy": "reuse",
        "verify_checksums": True,
    },

    # ------------------------------------------------------------------
    # 数据生成调度
    # count 不属于缓存身份：从 64 改到 128 时只补 64 个样本。
    # workers/native_threads_per_worker/maximum_teacher_attempts 也不属于缓存身份。
    # ------------------------------------------------------------------
    "DATA": {
        "count": 4096,
        "seed": 37,
        "workers": 8,
        "native_threads_per_worker": 1,
        # 单个 cache index 若遇到明确的 REFERENCE 数值诊断拒绝，则沿同一
        # deterministic RNG 流重采样完整场景；未知异常仍立即抛出。
        "maximum_teacher_attempts": 8,
        # 每完成一小批 teacher 后检查暂停/停止命令。
        "generation_chunk_size": 8,
    },

    # ------------------------------------------------------------------
    # 参数空间：MVP conductor + package + tensor-electric 扩展。
    # 这些参数属于 teacher cache 身份，修改后自动进入新的缓存目录。
    # ------------------------------------------------------------------
    "SAMPLER": {
        "base": {
            "conductor": {
                "outer_radius_range": [0.02, 0.05],
                "aspect_ratio_range": [0.7, 1.3],
                "turns_range": [0.6, 1.6],
                "pitch_range": [8.0e-4, 3.0e-3],
                "exponent_range": [2.0, 5.0],
                "width_range": [5.0e-4, 2.5e-3],
                "thickness_range": [3.0e-4, 1.5e-3],
                "separation_range": [0.012, 0.06],
                "conductivity_range": [3.0e7, 6.0e7],
                "frequency_range": [2.0e4, 2.0e5],
            },
            "package_margin_range": [1.35, 2.0],
            "package_half_z_range": [0.004, 0.012],
            "package_center_offset_fraction_range": [0.0, 0.35],
            "package_exponent_xy_range": [2.0, 5.0],
            "package_exponent_z_range": [2.0, 5.0],
            # 默认覆盖单/双 material region；多 package 由同一图网络处理。
            "package_count_range": [1, 2],
            # 严格嵌套与自由 inclusion 都进入训练域；partial overlap 仍拒绝。
            "nested_package_probability": 0.25,
            "graded_package_probability": 0.0,
            "nested_package_scale_range": [1.15, 1.45],
            "free_inclusion_probability": 0.20,
            "free_inclusion_center_radius_fraction_range": [0.65, 1.8],
            "free_inclusion_half_extent_fraction_range": [0.12, 0.45],
            # Tensor FAST 当前训练域使用 AC；精确 DC 留给 REFERENCE/CERTIFIED。
            "dc_probability": 0.0,
            "dc_conductive_probability": 0.0,
            "relative_permittivity_range": [1.5, 6.0],
            # 磁导率保持各向同性，但不再固定为真空值，让磁性修正进入训练。
            "package_relative_permeability_range": [0.8, 1.5],
            "dielectric_conductivity_range": [1.0e-7, 5.0e-3],
            "lossless_probability": 0.20,
            "debye_package_probability": 0.0,
            "multi_debye_package_probability": 0.0,
            "package_debye_epsilon_infinite_range": [1.5, 6.0],
            "package_debye_delta_epsilon_range": [0.5, 20.0],
            "package_debye_relaxation_time_range": [1.0e-8, 1.0e-4],
            "background_relative_permittivity_range": [1.0, 1.0],
            "background_conductivity_range": [1.0e-7, 5.0e-3],
            "lossy_background_probability": 0.25,
            "debye_background_probability": 0.0,
            "multi_debye_background_probability": 0.0,
            "multi_debye_poles_range": [2, 4],
            "background_debye_epsilon_infinite_range": [1.0, 6.0],
            "background_debye_delta_epsilon_range": [0.5, 30.0],
            "background_debye_relaxation_time_range": [1.0e-8, 1.0e-4],
        },
        "tensor_package_probability": 1.0,
        "tensor_background_probability": 0.25,
        "tensor_relative_permittivity_range": [1.5, 10.0],
        "tensor_conductivity_range": [1.0e-7, 5.0e-3],
        "tensor_lossless_probability": 0.20,
        # 单次 hybrid package rejection budget 用尽后，从同一 RNG 流继续
        # 采样完整场景；这是合法参数空间中的 rejection，不是训练错误。
        "maximum_scene_attempts": 64,
    },

    # ------------------------------------------------------------------
    # MQS correctness teacher。修改这些参数会生成新的数据缓存。
    # ------------------------------------------------------------------
    "TEACHER": {
        "segments_per_turn": 20,
        "min_segments": 16,
        "section_degree": 1,
        "radial_order": 4,
        "angular_order": 24,
        "line_order": 2,
        "self_softening_factor": 0.45,
        "section_basis_family": "adaptive",
        "skin_enrichment_threshold": 2.0,
        "skin_boundary_layers": 2,
        "skin_angular_order": 1,
        "skin_lambda_cap": 24.0,
    },

    # ------------------------------------------------------------------
    # Teacher truth 输出和积分阶数。修改会生成新的缓存。
    # port + spatial 共用一次 coupled EM solve，但端口能量积分与空间训练点
    # 分辨率独立：提高 port truth 精度不会自动把 spatial 数据量放大数倍。
    # ------------------------------------------------------------------
    "TRUTH": {
        "baseline_segments": 96,
        "surface_vertical_order": 16,
        "surface_azimuthal_order": 32,
        "magnetic_volume_axial_order": 8,
        "magnetic_volume_radial_order": 6,
        "magnetic_volume_azimuthal_order": 24,
        "maximum_raw_magnetic_reciprocity_defect": 0.08,
        "include_spatial": True,

        # Spatial 网络训练点分辨率：只决定保存多少局部 Joule-field 样本。
        "package_volume_axial_order": 6,
        "package_volume_radial_order": 4,
        "package_volume_azimuthal_order": 16,
        "background_radial_order": 10,
        "background_angular_order": 32,

        # Canonical port energy truth：与 REFERENCE/CERTIFIED 默认积分阶数一致。
        "energy_volume_axial_order": 8,
        "energy_volume_radial_order": 6,
        "energy_volume_azimuthal_order": 24,
        "energy_background_radial_order": 12,
        "energy_background_angular_order": 48,

        # raw closure 仅保留作诊断；energy-v3 最终闭合由共享 congruence 保证。
        "maximum_raw_spatial_closure_error": 0.35,
        # 兼容旧配置；energy-v3 不再依据 raw closure 自适应加密 truth。
        "maximum_spatial_quadrature_refinements": 0,
    },

    # ------------------------------------------------------------------
    # 通用运行设备。训练项 device="inherit" 时继承这里。
    # auto: CUDA -> MPS -> CPU。
    # ------------------------------------------------------------------
    "RUNTIME": {
        "device": "auto",
        # 传给 GUI 启动的 worker 进程；按机器需要增删。
        "environment": {
            "PYTHONUNBUFFERED": "1",
        },
    },

    # ------------------------------------------------------------------
    # 模型 checkpoint 总策略。
    # resume=True 会自动识别兼容 checkpoint；参数空间、样本数或关键训练
    # 配置改变时不会错误续接旧 checkpoint。
    # ------------------------------------------------------------------
    "TRAINING": {
        "resume": True,
    },

    # ------------------------------------------------------------------
    # Tensor FAST port 网络
    # ------------------------------------------------------------------
    "PORT_TRAINING": {
        "device": "inherit",
        # auto: CUDA/MPS=float32，CPU=float64；也可显式 float32/float64。
        "precision": "auto",
        "seed": 17,
        "epochs": 200,
        "batch_size": 32,
        "validation_fraction": 0.15,
        "patience": 30,
        "min_improvement": 1.0e-5,
        "channel_loss_weight": 1.0,
        # 1 最安全，写盘更多；可调大以减少 checkpoint I/O。
        "checkpoint_every_batches": 1,
        "geometry_domain": None,
        "model": {
            "hidden_dim": 64,
            "factor_rank": 4,
            "depth": 2,
            # 以下维度由当前 tensor feature schema 决定，通常不要覆盖；
            # 如未来 schema 扩展，可在实现支持后从此处显式配置。
        },
        "optimizer": {
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "gradient_clip_norm": 10.0,
        },
    },

    # ------------------------------------------------------------------
    # Tensor continuous spatial loss 网络
    # ------------------------------------------------------------------
    "SPATIAL_TRAINING": {
        "enabled": True,
        "device": "inherit",
        "seed": 47,
        "epochs": 140,
        "batch_size": 8,
        "validation_fraction": 0.15,
        "patience": 30,
        "validation_interval": 1,
        "min_improvement": 1.0e-5,
        "end_to_end_validation": True,
        "checkpoint_every_batches": 1,
        "model": {
            "field_hidden_dim": 128,
            "factor_rank": 4,
            "depth": 3,
        },
        "optimizer": {
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-6,
            "gradient_clip_norm": 10.0,
        },
        # FAST spatial 归一化/背景积分配置，也随 artifact 保存。
        "normalization": {
            "conductor_longitudinal_points": 12,
            "conductor_radial_order": 3,
            "conductor_angular_order": 16,
            "package_axial_order": 6,
            "package_radial_order": 4,
            "package_azimuthal_order": 16,
            "background_segments_per_turn": 16,
            "background_radial_order": 12,
            "background_angular_order": 48,
        },
    },

    # ------------------------------------------------------------------
    # 训练控制协议。暂停/停止仅在安全数值边界生效。
    # ------------------------------------------------------------------
    "CONTROL": {
        "heartbeat_interval_s": 0.5,
        "fsync_metrics": False,
    },

    # ------------------------------------------------------------------
    # PyQt + pyqtgraph 训练界面
    # GUI 不执行数值训练；训练运行在独立 QProcess，metrics 由 QThread 读取。
    # ------------------------------------------------------------------
    "GUI": {
        "title": "SDF-MPNEO vNext 训练控制台",
        "auto_start": False,
        "refresh_ms": 250,
        "max_plot_points": 4000,
        "console_blocks": 1000,
        "console_height": 190,
        "width": 1240,
        "height": 860,
    },
}


def main(argv=None):
    # The BFZI research path shares the repository entry point but deliberately
    # does not feed BFZI geometry into the legacy SuperellipseSpiral trainer.
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if "--mode" in args:
        index = args.index("--mode")
        if index + 1 < len(args):
            mode = args[index + 1]
            if mode == "bfzi":
                from examples.bfzi_mqs_demo import main as bfzi_main
                bfzi_main(ball_reference="--green-ball" in args)
                return 0
            if mode == "bfzi-snapshot":
                from examples.bfzi_snapshot_demo import main as bfzi_main
                bfzi_main(args[index + 2:])
                return 0
            if mode == "bfzi-generate":
                from examples.bfzi_dataset_generate import main as generate_main
                generate_main(args[index + 2:])
                return 0
            if mode == "bfzi-train":
                from examples.bfzi_neural_demo import train as neural_train
                neural_train(args[index + 2:])
                return 0
            if mode == "bfzi-infer":
                from examples.bfzi_neural_demo import infer as neural_infer
                neural_infer(args[index + 2:])
                return 0
            if mode == "bfzi-eval":
                from examples.bfzi_neural_demo import evaluate as neural_evaluate
                neural_evaluate(args[index + 2:])
                return 0
            if mode == "bfzi-p0-source":
                from examples.bfzi_p0_source import main as p0_source_main
                from examples.bfzi_run_record import run_and_publish
                return run_and_publish(
                    p0_source_main,
                    flags=args,
                    verbose="--verbose-log" in args,
                    publish="--no-publish-log" not in args,
                    with_tests="--with-tests" in args,
                    log_subdir="p0",
                )
            if mode == "bfzi-p1-full-sea":
                from examples.bfzi_p1_full_sea_taylor import main as p1_full_sea_main
                from examples.bfzi_run_record import run_and_publish
                return run_and_publish(
                    p1_full_sea_main,
                    flags=args,
                    verbose="--verbose-log" in args,
                    publish="--no-publish-log" not in args,
                    with_tests="--with-tests" in args,
                    log_subdir="p1_full",
                )
            if mode == "bfzi-p1-coulomb":
                from examples.bfzi_p1_coulomb import main as p1_coulomb_main
                from examples.bfzi_run_record import run_and_publish
                return run_and_publish(
                    p1_coulomb_main,
                    flags=args,
                    verbose="--verbose-log" in args,
                    publish="--no-publish-log" not in args,
                    with_tests="--with-tests" in args,
                    log_subdir="p1",
                )
            if mode == "bfzi-g1-foundations":
                from examples.bfzi_g1_foundations import main as g1_main
                from examples.bfzi_run_record import run_and_publish
                return run_and_publish(
                    lambda: g1_main(
                        hierarchy="--g1-hierarchy" in args,
                        source_panels="--g1-source-panels" in args,
                    ),
                    flags=args,
                    verbose="--verbose-log" in args,
                    publish="--no-publish-log" not in args,
                    with_tests="--with-tests" in args,
                )
            if mode == "bfzi-convergence":
                from examples.bfzi_convergence import main as bfzi_main
                from examples.bfzi_run_record import run_and_publish
                return run_and_publish(lambda: bfzi_main(
                    full="--full" in args,
                    basis_only="--basis" in args,
                    basis_verify="--basis-verify" in args,
                    basis_green="--basis-green" in args,
                    basis_green_refined="--basis-green-refined" in args,
                    basis_green_cross="--basis-green-cross" in args,
                    basis_green_gaussian="--basis-green-gaussian" in args,
                    basis_green_ball="--basis-green-ball" in args,
                    basis_ball="--basis-ball" in args,
                    basis_ball_check_l="--basis-ball-check-l" in args,
                    basis_ball_check_rk="--basis-ball-check-rk" in args,
                    basis_ball_check_far="--basis-ball-check-far" in args,
                    basis_ball_check_far_refined="--basis-ball-check-far-refined" in args,
                    basis_ball_ablate="--basis-ball-ablate" in args,
                    basis_ball_face="--basis-ball-face" in args,
                    basis_ball_edge="--basis-ball-edge" in args,
                    basis_ball_aniso="--basis-ball-aniso" in args,
                    basis_ball_targeted="--basis-ball-targeted" in args,
                    basis_domain_ball="--basis-domain-ball" in args,
                    basis_domain_extend="--basis-domain-extend" in args,
                    basis_domain_far="--basis-domain-far" in args,
                    basis_domain="--basis-domain" in args,
                    basis_local="--basis-local" in args,
                    basis_angular="--basis-angular" in args,
                    basis_geometry="--basis-geometry" in args,
                    basis_radial="--basis-radial" in args,
                    basis_kparts="--basis-kparts" in args,
                    basis_wallpanels="--basis-wallpanels" in args,
                ), flags=args,
                    verbose="--verbose-log" in args,
                    publish="--no-publish-log" not in args,
                    with_tests="--with-tests" in args,
                )

    from sdfmpneo_vnext.workflow import launch

    return launch(CONFIG, args, runner_path=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
