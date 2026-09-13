import importlib.util
from pathlib import Path


def _load_run():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("sdfmpneo_run_neural_defaults", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_defaults_use_one_fullspace_maxwell_path():
    run = _load_run()
    assert run.TRAINING["device"] == "cuda"
    assert "em_basis_anchor_residual" not in run.TRAINING
    assert "em_basis_max_rank" not in run.TRAINING
    assert run.TRAINING["residual_training_steps"] >= 1
    assert run.PHYSICS["maxwell_residual_tolerance"] > 0
    assert run.PHYSICS["maxwell_restart"] >= 1
    assert run.TRAINING["network"]["message_passing_steps"] >= 1
    assert run.TRAINING["network"]["solver_steps"] >= 2
    assert "polynomial_order" not in run.TRAINING["network"]
    assert "coefficient_limit" not in run.TRAINING["network"]


def test_training_defaults_use_one_shared_solver_step_setting_and_real_plateau_stop():
    run = _load_run()
    optimizer = run.TRAINING["optimizer"]
    assert "unroll_steps" not in optimizer
    assert optimizer["patience"] <= 30
    assert optimizer["min_relative_improvement"] >= 1e-3
    assert optimizer["benchmark_samples_per_split"] >= 1


def test_temperature_input_is_physical_temperature_rise_not_modal_box():
    run = _load_run()
    assert "initial_temperature_rise" in run.PREDICTION
    assert "state_lower" not in run.TRAINING
    assert "state_upper" not in run.TRAINING
