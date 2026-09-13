"""统一几何、无解标签、residual-corrected 神经电磁-热求解器。

只修改本文件顶部配置：

    python run.py --mode train
    python run.py --mode predict

几何不再生成贴体四面体网格。线圈/封装通过解析几何改变固定多尺度背景中的
材料占据与激励；神经网络只预测 Maxwell 多右端项初解，最终解始终由真实背景
Maxwell residual 修正到指定容差。
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
        "shape": "circle",
        "turns": 1.5,
        "outer_half_size": 0.025,
        "pitch": 0.002,
        "conductor_width": 0.0015,
        "conductor_thickness": 0.001,
        "corner_radius": 0.012,
        "translation": [0.0, 0.0, 0.0],
        "angles": [0.0, 0.0, 0.0],
    },
    "receiver": {
        "shape": "circle",
        "turns": 1.5,
        "outer_half_size": 0.025,
        "pitch": 0.002,
        "conductor_width": 0.0015,
        "conductor_thickness": 0.001,
        "corner_radius": 0.012,
        "translation": [0.0, 0.0, 0.035],
        "angles": [0.0, 0.0, 0.0],
    },
    "package_half_extent": [0.035, 0.035, 0.005],
}

# 这些范围只用于教网络“如何更快找到解”，不是模型有效域，也不参与推理拒绝。
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
}

MATERIALS = {
    "tx_copper": {
        "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 1.0,
        "thermal_conductivity": 400.0,
        "volumetric_heat_capacity": 3.45e6,
    },
    "rx_copper": {
        "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 1.0,
        "thermal_conductivity": 400.0,
        "volumetric_heat_capacity": 3.45e6,
    },
    "tx_package": {
        "electrical_conductivity": 0.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 3.0,
        "thermal_conductivity": 0.2,
        "volumetric_heat_capacity": 1.5e6,
    },
    "rx_package": {
        "electrical_conductivity": 0.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 3.0,
        "thermal_conductivity": 0.2,
        "volumetric_heat_capacity": 1.5e6,
    },
    "seawater": {
        "electrical_conductivity": 5.0,
        "resistivity_temperature_coefficient": 0.0,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 80.0,
        "thermal_conductivity": 0.6,
        "volumetric_heat_capacity": 4.1e6,
    },
}

REGIONS = {
    "coil_materials": ["tx_copper", "rx_copper"],
    "package_materials": ["tx_package", "rx_package"],
    "seawater_material": "seawater",
}
PORTS = {"current_offset": None, "current_matrix": None}

THERMAL_RANK = 24
TRAINING = {
    "state_lower": [-0.03] * THERMAL_RANK,
    "state_upper": [0.12] * THERMAL_RANK,
    "seed": 17,
    "basis_samples": 24,
    "em_basis_max_rank": 64,
    "em_basis_anchor_residual": 1e-2,
    "n_operator_samples": 512,
    "device": "cuda",
    "network": {"width": 256, "blocks": 4, "activation": "silu"},
    "optimizer": {
        "epochs": 1000,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "patience": 150,
        "validation_interval": 5,
        "seed": 17,
        "dtype": "float32",
    },
}

PREDICTION = {
    "a0": [0.0] * THERMAL_RANK,
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
    "enabled": True,
    "auto_start": False,
    "log_dir": "results/uwpt/logs",
    "log_interval_s": 1.0,
    "refresh_ms": 300,
    "max_plot_points": 4000,
    "compute_threads": 1,
}

SETTINGS = {
    "ROOT": str(ROOT),
    "MODE": MODE,
    "FILES": FILES,
    "BACKGROUND": BACKGROUND,
    "DEFAULT_GEOMETRY": DEFAULT_GEOMETRY,
    "GEOMETRY_SAMPLING": GEOMETRY_SAMPLING,
    "PHYSICS": PHYSICS,
    "MATERIALS": MATERIALS,
    "REGIONS": REGIONS,
    "PORTS": PORTS,
    "THERMAL_RANK": THERMAL_RANK,
    "TRAINING": TRAINING,
    "PREDICTION": PREDICTION,
    "MONITOR": MONITOR,
}


def main(argv=None):
    from sdfmpneo.unified_runtime import launch

    return launch(SETTINGS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
