"""统一 geometry→spatial Joule tensor + geometry-local thermal ROM 生产入口。

只修改本文件顶部配置：

    python run.py --mode train
    python run.py --mode predict

理论主链：

    geometry
      -> MLP: Z_field(g), D_vol(g), H_cell(g)
      -> cellwise PSD + exact sum(H_cell)=D_vol
      -> query geometry 的真实 M(g), K(g)
      -> 小型 geometry-local rational-Krylov thermal ROM
      -> explicit current/circuit + wire resistance
      -> temperature

Maxwell 只在离线 truth 生成时求解；在线推理没有 Maxwell/FGMRES/full-field
electromagnetic correction。离线 truth 使用 finite-cross-section stranded source、
Silver--Mueller 开放边界和 canonical local fine-minus-coarse self correction。
训练前只做 EM truth preflight，不再构造跨 geometry 的全局 thermal state basis。
训练/发布前的独立 Gate 会直接比较 full-cell thermal transient 与在线小 ROM，
并在 completely-held-out geometry 上检查 spatial Joule、current/circuit dynamics、
production integrator 和 steady state。
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
        "linear_direct_fallback_max_dofs": 60000,
        # The compatible transverse solve can use a bounded direct factorization
        # for medium local systems.  The 64k-edge case observed in production
        # stalls completely under ILU/LGMRES, while this 100k cap still excludes
        # the 118k/254k refined validation systems that must stay iterative.
        "linear_transverse_direct_max_dofs": 100000,
        "linear_transverse_direct_fallback_max_dofs": 100000,
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

# Production geometry is a constrained family, not the old 27D solver-
# perturbation box. Shape, turns, orientation and topology stay fixed.
# planar_scale jointly scales outer size, pitch, conductor width and corner
# radius. The historical 10th parameter was seawater_radius; in the current
# fixed open-boundary formulation that quantity is no longer an online geometry
# variable and is certified separately by BACKGROUND.open_boundary_check.
GEOMETRY_FAMILY = {
    "schema": "scaled_uwpt_family_v1",
    "parameters": {
        "tx_planar_scale": {"bounds": [0.97, 1.03]},
        "rx_planar_scale": {"bounds": [0.97, 1.03]},
        "tx_thickness_scale": {"bounds": [0.95, 1.05]},
        "rx_thickness_scale": {"bounds": [0.95, 1.05]},
        "rx_offset_x": {"bounds": [-0.0002, 0.0002]},
        "rx_offset_y": {"bounds": [-0.0002, 0.0002]},
        "rx_gap": {"bounds": [0.0348, 0.0352]},
        "tx_package_scale": {"bounds": [0.98, 1.02]},
        "rx_package_scale": {"bounds": [0.98, 1.02]},
    },
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
    "spatial_tensor_schema": "cellwise_joule_tensor_v1",
    # Online thermal ROM is built independently for each query geometry from
    # true M(g), K(g) and the predicted complete Hermitian current-source span.
    "online_thermal_relative_tolerance": 5e-2,
    "online_thermal_conditioning_limit": 1e10,
    "thermal_time_scales": [0.1, 1.0, 10.0],
    "thermal_trajectory_times": [0.1, 1.0, 10.0, 100.0],
    # Start from the existing expensive truth budget, then add only
    # high-coverage geometries if decoded physical validation still misses the
    # release margin.  Existing cached truth is always reused.
    "n_tensor_samples": 96,
    "tensor_enrichment": {
        "enabled": True,
        "max_samples": 256,
        "batch_size": 32,
        "candidate_pool": 2048,
    },
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
        # Global Z/D/outward head starts from a small expensive truth set
        # and is enlarged only by adaptive maximin enrichment; keep it compact
        # and regularized. The coordinate field head sees many cells per
        # geometry and can use more capacity.
        "global": {
            "width": 32,
            "blocks": 1,
            "activation": "silu",
        },
        "field": {
            "width": 128,
            "blocks": 3,
            "activation": "silu",
        },
    },
    "optimizer": {
        "epochs": 240,
        "batch_size": 16,
        "field_batch_size": 4096,
        "field_cells_per_geometry": 2048,
        "learning_rate": 1e-3,
        "global_learning_rate": 5e-4,
        "field_learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "global_weight_decay": 1e-3,
        "field_weight_decay": 1e-6,
        "patience": 40,
        "validation_interval": 2,
        "gradient_clip_norm": 10.0,
        "physics_penalty_weight": 0.05,
        "z_weight": 1.0,
        "d_weight": 1.0,
        "outward_weight": 1.0,
        "spatial_weight": 1.0,
        "field_density_weight": 1.0,
        "field_shape_weight": 1.0,
        "field_physical_weight": 1.0,
        "field_density_prior_strength": 0.9,
        "field_log_density_margin": 0.5,
        "field_full_validation_interval": 10,
        # Select the stopping point on the historical validation split, then
        # make the production fit use every expensive cached truth geometry.
        # Release certification remains the independent final held-out audit.
        "refit_all_truth": True,
        "refit_global_head": True,
        "refit_field_head": True,
        "refit_learning_rate_factor": 1.0,
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
    "DEFAULT_GEOMETRY": DEFAULT_GEOMETRY, "GEOMETRY_FAMILY": GEOMETRY_FAMILY,
    "PHYSICS": PHYSICS, "MATERIALS": MATERIALS, "REGIONS": REGIONS, "PORTS": PORTS,
    "TRAINING": TRAINING, "PREDICTION": PREDICTION, "MONITOR": MONITOR,
}


def main(argv=None):
    from sdfmpneo.unified_runtime import launch
    return launch(SETTINGS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
