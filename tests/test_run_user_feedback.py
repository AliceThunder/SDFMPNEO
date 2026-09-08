from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_run_module():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("sdfmpneo_run", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_sample_ranges_print_geometry_initial_current_and_time(capsys):
    run = load_run_module()
    model = SimpleNamespace(
        geometry_names=("scale", "gap"),
        lower=np.array([0.9, 0.01]),
        upper=np.array([1.1, 0.02]),
        geometry_reference=np.array([1.0, 0.015]),
    )
    config = SimpleNamespace(
        initial_lower=(-0.1, -0.2), initial_upper=(0.1, 0.2),
        operating_lower=(0.0, 1.0), operating_upper=(10.0, 2.0),
        time_sampling="mixed_log", time_min=1e-6, time_horizon=1e5,
        include_steady_state=True, sample_count=64, validation_count=32,
        residual_tolerance=1e-5,
    )

    run.print_training_sample_ranges(model, config)
    text = capsys.readouterr().out
    assert "训练样本参数范围" in text
    assert "scale: [0.9, 1.1]" in text
    assert "gap: [0.01, 0.02]" in text
    assert "a0[0]: [-0.1, 0.1]" in text
    assert "U[1]: [1, 2]" in text
    assert "[0, 100000] s" in text
    assert "稳态残差点: 包含" in text
    assert "训练=64，独立检查=32" in text


def test_assembly_progress_emits_start_and_finish(capsys):
    run = load_run_module()
    old = run.MONITOR["assembly_progress_interval_s"]
    run.MONITOR["assembly_progress_interval_s"] = 0.01
    try:
        with run.assembly_progress():
            pass
    finally:
        run.MONITOR["assembly_progress_interval_s"] = old
    text = capsys.readouterr().out
    assert "[组装进度]" in text
    assert "完成" in text
    assert "总耗时" in text
