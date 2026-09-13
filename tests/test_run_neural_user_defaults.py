import importlib
import importlib.util
from pathlib import Path


def _load_run():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("sdfmpneo_run_neural_defaults", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_top_level_package_imports_after_neural_solver_rewrite():
    package = importlib.import_module("sdfmpneo")
    assert hasattr(package, "OperatorGraph")
    assert hasattr(package, "build_edge_residual_operator")
    assert not hasattr(package, "edge_group_ids")


def test_run_defaults_use_one_multiscale_fullspace_maxwell_path():
    run = _load_run()
    network = run.TRAINING["network"]
    assert run.TRAINING["device"] == "cuda"
    assert "em_basis_anchor_residual" not in run.TRAINING
    assert "em_basis_max_rank" not in run.TRAINING
    assert run.TRAINING["n_operator_samples"] == 96
    assert run.PHYSICS["maxwell_residual_tolerance"] > 0
    assert run.PHYSICS["maxwell_restart"] >= 1
    assert network["fine_message_steps"] >= 1
    assert network["coarse_levels"] >= 2
    assert network["coarse_message_steps"] >= 1
    assert network["fusion_message_steps"] >= 1
    assert network["solver_steps"] == 3
    assert "message_passing_steps" not in network
    assert "polynomial_order" not in network
    assert "coefficient_limit" not in network


def test_training_defaults_cover_fullspace_residuals_and_use_each_operator_update():
    run = _load_run()
    optimizer = run.TRAINING["optimizer"]
    assert "unroll_steps" not in optimizer
    assert optimizer["batch_size"] == 1
    assert optimizer["gradient_accumulation_steps"] == 1
    assert optimizer["batch_size"] * optimizer["gradient_accumulation_steps"] == 1
    assert optimizer["random_residual_vectors"] >= 2
    assert optimizer["smooth_residual_vectors"] >= 2
    assert optimizer["final_step_loss_weight"] >= 0.7
    assert optimizer["learning_rate"] == 2e-3
    assert optimizer["lr_decay_factor"] == 0.5
    assert optimizer["lr_plateau_patience"] >= 6
    assert optimizer["minimum_learning_rate"] <= 2.5e-4
    assert optimizer["patience"] >= 30
    assert optimizer["min_relative_improvement"] <= 5e-4
    assert optimizer["benchmark_samples_per_split"] >= 1


def test_temperature_input_is_physical_temperature_rise_not_modal_box():
    run = _load_run()
    assert "initial_temperature_rise" in run.PREDICTION
    assert "state_lower" not in run.TRAINING
    assert "state_upper" not in run.TRAINING
