"""统一几何、full-space sparse-neural-FGMRES、电磁-热求解器。

只修改本文件顶部配置：

    python run.py --mode train
    python run.py --mode predict

Maxwell 不构造全局解基/Maxwell rank。神经网络直接在真实 sparse Maxwell 耦合图上做
message passing，并输出 full edge-space correction；训练与推理共享同一组 solver steps，
FGMRES 只负责最终真实 sparse residual 闭环。thermal rank 仍由真实 Joule 热源与热方程
residual 自动决定。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODE = "train"
ROOT = Path(__file__).resolve().parent

FILES = {
    "model": "results/uwpt/model.npz",
    "predictions": "results/uwpt/predictions.json",
    "settings_dir": "results/uwpt",
    "training_checkpoint": "results/uwpt/model.training.pt",
}

BACKGROUND = {
    "bounds": [[-0.15, 0.15], [-0.15, 0.15], [-0.15, 0.15]],
    "core_center": [0.0, 0.0, 0.02],
    "core_half_extent": [0.09, 0.09, 0.09],
    "fine_step": 0.012,
    "growth": 1.5,
    "max_step": 0.03,
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
    "maxwell_residual_tolerance": 1e-7,
    "maxwell_max_iterations": 200,
    "maxwell_restart": 40,
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
    "basis_samples": 24,
    "thermal_basis_anchor_residual": 5e-2,
    "em_temperature_rise_bounds": [0.0, 80.0],
    "n_operator_samples": 96,
    "residual_training_steps": 3,
    "device": "cuda",
    "network": {
        "width": 32,
        "message_passing_steps": 3,
        "solver_steps": 3,
        "activation": "silu",
    },
    "optimizer": {
        "epochs": 120,
        "batch_size": 1,
        "learning_rate": 2e-3,
        "weight_decay": 1e-6,
        "patience": 20,
        "validation_interval": 2,
        "min_relative_improvement": 1e-3,
        "benchmark_samples_per_split": 4,
        "seed": 17,
        "dtype": "float32",
    },
}

PREDICTION = {
    "initial_temperature_rise": 0.0,
    "operating": [5.0, 0.0],
    "geometry": None,
    "times": [0.0, 0.001, 1.0, 1000.0, "inf"],
    "method": "etd2_adaptive",
    "max_step": 100.0,
    "initial_step": 0.001,
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
