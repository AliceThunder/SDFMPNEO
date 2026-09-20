"""统一 geometry→EM tensor + geometry-aware thermal ROM 生产入口。

只修改本文件顶部配置：

    python run.py --mode train
    python run.py --mode predict

理论主链：

    geometry
      -> deterministic Phi(g), Mr(g), Kr(g)
      -> MLP: Z_field(g), D_vol(g), H_j(g)
      -> explicit current/circuit + wire resistance
      -> true reduced thermal ODE
      -> temperature

Maxwell 只在离线 truth 生成时求解；在线推理没有 neural Maxwell solver、Krylov/FGMRES
或 full-field correction。离线 truth 使用 finite-cross-section stranded source、Silver--Mueller
开放边界，并用 canonical local fine-minus-coarse defect 修正全局粗网格未解析的端口 self response。
训练流程先做 pre-basis truth preflight，再构造 geometry-aware thermal ROM/tensor truth，训练后还必须
通过 completely-held-out current/circuit end-to-end Go/No-Go 才会保存模型。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODE = "train"
ROOT = Path(__file__).resolve().parent

FILES = {
    "model": "results/uwpt/model.geometry_thermal.npz",
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",
    "training_checkpoint": "results/uwpt/model.tensor_training.pt",
}

BACKGROUND = {
    # Open-domain convergence at ±0.27 m was not sufficient for the production
    # geometry box: the independent ±0.39 m reference changed D_vol by ~11%
    # and mutual Z by ~22%. Keep the same 12-mm resolved core, but move the
    # artificial boundary to ±0.51 m. Far seawater may stretch to 80 mm; the
    # independent mesh Gate still checks this coarsening.
    "bounds": [[-0.51, 0.51], [-0.51, 0.51], [-0.51, 0.51]],
    "core_center": [0.0, 0.0, 0.02],
    "core_half_extent": [0.09, 0.09, 0.09],
    "fine_step": 0.012,
    "growth": 1.5,
    "max_step": 0.08,
    # Large open-domain matrices use Maxwell-aware shifted-ILU + LGMRES.
    # The shift belongs only to the preconditioner. If the main Krylov solve
    # stalls around 1e-6--1e-7, true-residual defect correction reuses the same
    # ILU until the original physical matrix reaches the 1e-9 certificate.
    "linear_solver": {
        "relative_residual_tolerance": 1e-9,
        "direct_max_dofs": 60000,
        "iterative_maxiter": 40,
        "iterative_inner_m": 30,
        "iterative_defect_steps": 3,
        "iterative_defect_maxiter": 16,
        "iterative_defect_inner_m": 20,
        "iterative_defect_start_residual": 5e-6,
        "ilu_drop_tolerance": 5e-3,
        "ilu_fill_factor": 4.0,
        "ilu_strong_drop_tolerance": 1e-3,
        "ilu_strong_fill_factor": 8.0,
        "ilu_shift_factor": 3e-2,
        "ilu_strong_shift_factor": 1e-1,
    },
    # The global grid is intentionally kept coarse enough for many-geometry truth.
    # Only the unresolved diagonal self response receives a small canonical local
    # fine-minus-coarse defect. Translation/rotation are removed in that local
    # solve, while global mutual/far-field coupling remains from the full domain.
    "self_correction": {
        "enabled": True,
        "samples": 1,
        "fine_step": 0.003,
        "validation_fine_step": 0.00225,
        "core_padding": 0.006,
        "boundary_padding": 0.04,
        "growth": 1.5,
        "max_step": 0.02,
        "relative_tolerance": 1e-1,
        "joule_identity_tolerance": 1e-10,
        "linear_relative_residual_tolerance": 1e-9,
        "linear_direct_max_dofs": 60000,
        # Never send the 118k/254k local problems back to a fill-heavy full LU.
        "linear_direct_fallback_max_dofs": 60000,
        "linear_iterative_maxiter": 40,
        "linear_iterative_inner_m": 30,
        "linear_iterative_defect_steps": 3,
        "linear_iterative_defect_maxiter": 16,
        "linear_iterative_defect_inner_m": 20,
        "linear_iterative_defect_start_residual": 5e-6,
        "linear_ilu_drop_tolerance": 5e-3,
        "linear_ilu_fill_factor": 4.0,
        "linear_ilu_strong_drop_tolerance": 1e-3,
        "linear_ilu_strong_fill_factor": 8.0,
        "linear_ilu_shift_factor": 3e-2,
        "linear_ilu_strong_shift_factor": 1e-1,
        "parallel_ports": 2,
        "linear_result_cache_size": 64,
    },
    # Production is ±0.51 m; the independent reference is ±0.63 m. The 5%
    # convergence requirement is unchanged. D_out itself is diagnostic in
    # conductive seawater; its change is normalized by the terminal-dissipation
    # scale, while Z/D_vol/current-space power remain hard Gate quantities.
    "open_boundary_check": {
        "samples": 1,
        "padding": 0.12,
        "relative_tolerance": 5e-2,
    },
    "formulation_check": {
        "samples": 1,
        "relative_tolerance": 2e-2,
    },
    "mesh_check": {
        "samples": 1,
        "refinement_factor": 0.75,
        "relative_tolerance": 1e-1,
        "source_path_relative_tolerance": 1e-10,
    },
    "geometry_continuity_check": {
        "samples": 1,
        "translation_step": 1e-4,
        "angle_step": 1e-3,
        "relative_change_limit": 2e-1,
    },
}

DEFAULT_GEOMETRY = {
    "transmitter": {
        "shape": "circle", "turns": 1.5, "outer_half_size": 0.025,
        "pitch": 0.002, "conductor_width": 0.0015, "conductor_thickness": 0.001,
        "corner_radius": 0.012, "translation": [0.0, 0.0, 0.0], "angles": [0.0, 0.0, 0.0],
    },
    "receiver": {
        "shape": "circle", "turns": 1.5, "outer_half_size": 0.025,
        "pitch": 0.002, "conductor_width": 0.0015, "conductor_thickness": 0.001,
        "corner_radius": 0.012, "translation": [0.0, 0.0, 0.035], "angles": [0.0, 0.0, 0.0],
    },
    "package_half_extent": [0.035, 0.035, 0.005],
}

GEOMETRY_SAMPLING = {
    "transmitter": {
        "shape": {"choices": ["circle", "rounded_square"]},
        "turns": {"bounds": [0.5, 3.0]},
        "outer_half_size": {"bounds": [0.012, 0.045]},
        "pitch": {"bounds": [0.001, 0.004]},
        "conductor_width": {"bounds": [0.0006, 0.003]},
        "conductor_thickness": {"bounds": [0.0004, 0.0025]},
        "corner_radius": {"bounds": [0.006, 0.035]},
        "translation": {"bounds": [[-0.025, 0.025], [-0.025, 0.025], [-0.02, 0.02]]},
        "angles": {"bounds": [[-0.5, 0.5], [-0.5, 0.5], [-np.pi, np.pi]]},
    },
    "receiver": {
        "shape": {"choices": ["circle", "rounded_square"]},
        "turns": {"bounds": [0.5, 3.0]},
        "outer_half_size": {"bounds": [0.012, 0.045]},
        "pitch": {"bounds": [0.001, 0.004]},
        "conductor_width": {"bounds": [0.0006, 0.003]},
        "conductor_thickness": {"bounds": [0.0004, 0.0025]},
        "corner_radius": {"bounds": [0.006, 0.035]},
        "translation": {"bounds": [[-0.06, 0.06], [-0.06, 0.06], [0.005, 0.09]]},
        "angles": {"bounds": [[-1.0, 1.0], [-1.0, 1.0], [-np.pi, np.pi]]},
    },
    "package_half_extent": {"bounds": [[0.02, 0.06], [0.02, 0.06], [0.003, 0.012]]},
}

PHYSICS = {
    "frequency_hz": 100000.0,
    "ambient_temperature": 293.15,
}

MATERIALS = {
    "tx_copper": {"electrical_conductivity": 5.8e7, "resistivity_temperature_coefficient": 0.00393,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 1.0,
                   "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6},
    "rx_copper": {"electrical_conductivity": 5.8e7, "resistivity_temperature_coefficient": 0.00393,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 1.0,
                   "thermal_conductivity": 400.0, "volumetric_heat_capacity": 3.45e6},
    "tx_package": {"electrical_conductivity": 0.0, "resistivity_temperature_coefficient": 0.0,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 3.0,
                   "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6},
    "rx_package": {"electrical_conductivity": 0.0, "resistivity_temperature_coefficient": 0.0,
                   "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 3.0,
                   "thermal_conductivity": 0.2, "volumetric_heat_capacity": 1.5e6},
    "seawater": {"electrical_conductivity": 5.0, "resistivity_temperature_coefficient": 0.0,
                 "reference_temperature": 293.15, "relative_permeability": 1.0, "relative_permittivity": 80.0,
                 "thermal_conductivity": 0.6, "volumetric_heat_capacity": 4.1e6},
}

REGIONS = {
    "coil_materials": ["tx_copper", "rx_copper"],
    "package_materials": ["tx_package", "rx_package"],
    "seawater_material": "seawater",
}

PORTS = {"current_offset": None, "current_matrix": None}

TRAINING = {
    "seed": 17,
    "thermal_basis_schema": "geometry_aware_scaled_local_atlas_v9",
    "basis_samples": 8,
    "basis_design_pool_multiplier": 16,
    "basis_validation_samples": 6,
    "thermal_basis_energy_tolerance": 5e-2,
    "thermal_time_scales": [0.1, 1.0, 10.0],
    "thermal_trajectory_times": [0.1, 1.0, 10.0, 100.0],
    "thermal_basis_max_rank": None,
    "thermal_basis_conditioning_limit": 1e10,
    "thermal_component_target_multiplier": 2.0,
    "n_tensor_samples": 96,
    "final_audit": {
        "samples": 2,
        "times": [0.1, 1.0, 10.0, 100.0],
        "full_vs_rom_thermal_tolerance": 5e-2,
        "tensor_relative_tolerance": 2e-1,
        "current_space_relative_tolerance": 2e-1,
        "outward_relative_tolerance": 2e-1,
        "projection_correction_limit": 2e-1,
        "reduced_dynamic_relative_tolerance": 1e-1,
        "integrator_relative_tolerance": 1e-4,
        "integrator_rtol": 1e-7,
        "integrator_atol": 1e-9,
        "integrator_max_step": 10.0,
        "circuit_condition_limit": 1e8,
        "operating_cases": [
            {"name": "current-controlled", "operating": [5.0, 0.0]},
            {"name": "circuit-controlled", "drive": {
                "voltage": [10.0, 0.0],
                "series_impedance": [0.1, 0.1],
            }},
        ],
    },
    "device": "cuda",
    "network": {
        "width": 128,
        "blocks": 3,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 240,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "patience": 40,
        "validation_interval": 2,
        "gradient_clip_norm": 10.0,
        "pod_relative_tail_tolerance": 1e-4,
        "physics_penalty_weight": 0.05,
        "z_weight": 1.0,
        "d_weight": 1.0,
        "h_weight": 1.0,
        "seed": 17,
        "dtype": "float32",
    },
}

PREDICTION = {
    "initial_temperature_rise": 0.0,
    "operating": [5.0, 0.0],
    # Voltage-driven example:
    # "drive": {"voltage": [10.0, 0.0], "series_impedance": [0.1, 0.1]},
    "geometry": None,
    "times": [0.0, 0.1, 1.0, 1000.0, "inf"],
    "method": "etd2_adaptive",
    "max_step": 100.0,
    "initial_step": 0.01,
    "rtol": 1e-5,
    "atol": 1e-8,
    "steady_tolerance": 1e-10,
    "steady_max_iterations": 40,
}

MONITOR = {
    "enabled": True, "auto_start": False, "log_dir": "results/uwpt/logs",
    "log_interval_s": 1.0, "refresh_ms": 300, "max_plot_points": 4000, "compute_threads": 1,
}

SETTINGS = {
    "ROOT": str(ROOT), "MODE": MODE, "FILES": FILES, "BACKGROUND": BACKGROUND,
    "DEFAULT_GEOMETRY": DEFAULT_GEOMETRY, "GEOMETRY_SAMPLING": GEOMETRY_SAMPLING,
    "PHYSICS": PHYSICS, "MATERIALS": MATERIALS, "REGIONS": REGIONS, "PORTS": PORTS,
    "TRAINING": TRAINING, "PREDICTION": PREDICTION, "MONITOR": MONITOR,
}


def main(argv=None):
    from sdfmpneo.unified_runtime import launch
    return launch(SETTINGS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
