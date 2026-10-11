"""SDF-MPNEO vNext Generation-2 一键入口。

正常使用只需要修改本文件的 CONFIG，然后运行：

    python run.py

默认打开非阻塞 PyQt 训练界面。也可以：

    python run.py --mode train     # 无界面训练
    python run.py --fresh          # 忽略 Gen2 模型 checkpoint，继续复用 teacher 缓存

Generation-2 与旧模型使用不同 checkpoint/artifact/bundle 路径；energy-v3 teacher
缓存保持不变并直接复用。
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


CONFIG = {
    "ROOT": str(ROOT),
    "RUN": {
        "mode": "gui",
    },

    "FILES": {
        "log_dir": "results/vnext/logs",
        "checkpoint_dir": "results/vnext/checkpoints",
        "port_checkpoint": "generation2_port.training.pt",
        "spatial_checkpoint": "generation2_spatial.training.pt",
        "port_artifact": "results/vnext/generation2_port.pt",
        "spatial_artifact": "results/vnext/generation2_spatial.pt",
        "bundle": "results/vnext/generation2_tensor_bundle",
        "summary": "results/vnext/generation2_training_summary.json",
        "overwrite_bundle": True,
    },

    # energy-v3 teacher 缓存身份与 Gen1 保持一致；神经网络换代不会重算 teacher。
    "CACHE": {
        "root": "results/vnext/teacher_cache",
        "policy": "reuse",
        "verify_checksums": True,
    },

    "DATA": {
        "count": 4096,
        "seed": 37,
        "workers": 8,
        "native_threads_per_worker": 1,
        "maximum_teacher_attempts": 8,
        "generation_chunk_size": 8,
    },

    # 第一次 Gen2 实验故意保持与 Gen1 相同的 teacher 分布，隔离架构收益。
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
            "package_count_range": [1, 2],
            "nested_package_probability": 0.25,
            "graded_package_probability": 0.0,
            "nested_package_scale_range": [1.15, 1.45],
            "free_inclusion_probability": 0.20,
            "free_inclusion_center_radius_fraction_range": [0.65, 1.8],
            "free_inclusion_half_extent_fraction_range": [0.12, 0.45],
            "dc_probability": 0.0,
            "dc_conductive_probability": 0.0,
            "relative_permittivity_range": [1.5, 6.0],
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
        "maximum_scene_attempts": 64,
    },

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

    "TRUTH": {
        "baseline_segments": 96,
        "surface_vertical_order": 16,
        "surface_azimuthal_order": 32,
        "magnetic_volume_axial_order": 8,
        "magnetic_volume_radial_order": 6,
        "magnetic_volume_azimuthal_order": 24,
        "maximum_raw_magnetic_reciprocity_defect": 0.08,
        "include_spatial": True,
        "package_volume_axial_order": 6,
        "package_volume_radial_order": 4,
        "package_volume_azimuthal_order": 16,
        "background_radial_order": 10,
        "background_angular_order": 32,
        "energy_volume_axial_order": 8,
        "energy_volume_radial_order": 6,
        "energy_volume_azimuthal_order": 24,
        "energy_background_radial_order": 12,
        "energy_background_angular_order": 48,
        "maximum_raw_spatial_closure_error": 0.35,
        "maximum_spatial_quadrature_refinements": 0,
    },

    "RUNTIME": {
        "device": "auto",
        "environment": {
            "PYTHONUNBUFFERED": "1",
        },
    },

    # 一个全局不可变 partition 同时服务 Port 与 Spatial；TEST 从不选 checkpoint。
    "TRAINING": {
        "generation": 2,
        "resume": True,
        "split_seed": 2027,
        "validation_fraction": 0.10,
        "test_fraction": 0.10,
    },

    "PORT_TRAINING": {
        "device": "inherit",
        "precision": "auto",
        "seed": 17,
        "epochs": 200,
        "batch_size": 32,
        "patience": 20,
        "min_improvement": 1.0e-5,
        "resistance_weight": 1.0,
        "reactance_weight": 1.0,
        # Spatial 直接依赖 dissipation channels，因此其泛化误差是一等目标。
        "channel_loss_weight": 2.0,
        "checkpoint_every_batches": 1,
        "geometry_domain": None,
        "model": {
            "hidden_dim": 64,
            "factor_rank": 4,
            "depth": 2,
            "interaction_rounds": 3,
            "resistance_log_limit": 4.0,
        },
        "optimizer": {
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-5,
            "gradient_clip_norm": 10.0,
        },
    },

    "SPATIAL_TRAINING": {
        "enabled": True,
        "device": "inherit",
        "seed": 47,
        "epochs": 140,
        "batch_size": 8,
        "patience": 20,
        "validation_interval": 1,
        # E2E 是独立诊断；checkpoint 只按 canonical validation shape 选择。
        "end_to_end_validation_interval": 1,
        "min_improvement": 1.0e-5,
        "checkpoint_every_batches": 1,
        "model": {
            "context_hidden_dim": 64,
            "context_rounds": 1,
            "context_depth": 1,
            "field_hidden_dim": 128,
            "factor_rank": 4,
            "depth": 3,
        },
        "optimizer": {
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-6,
            "gradient_clip_norm": 10.0,
        },
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

    "CONTROL": {
        "heartbeat_interval_s": 0.5,
        "fsync_metrics": False,
    },

    "GUI": {
        "title": "SDF-MPNEO vNext Generation-2 训练控制台",
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
    from sdfmpneo_vnext.generation2_workflow import launch

    return launch(CONFIG, argv, runner_path=__file__)


if __name__ == "__main__":
    raise SystemExit(main())
