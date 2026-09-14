import importlib.util
from pathlib import Path


def _load_run():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("sdfmpneo_run_tensor_defaults", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_defaults_use_geometry_to_tensor_and_geometry_aware_thermal_rom():
    run = _load_run()
    network = run.TRAINING["network"]
    boundary = run.BACKGROUND["open_boundary_check"]
    formulation = run.BACKGROUND["formulation_check"]
    mesh = run.BACKGROUND["mesh_check"]
    continuity = run.BACKGROUND["geometry_continuity_check"]
    final_audit = run.TRAINING["final_audit"]
    assert run.TRAINING["device"] == "cuda"
    assert run.TRAINING["n_tensor_samples"] >= 6
    assert run.TRAINING["basis_validation_samples"] >= 1
    assert run.TRAINING["thermal_basis_schema"] == "geometry_aware_bg_local_v1"
    assert run.TRAINING["thermal_basis_energy_tolerance"] > 0
    assert run.TRAINING["thermal_basis_conditioning_limit"] > 1
    assert run.TRAINING["thermal_time_scales"] == [0.1, 1.0, 10.0]
    assert min(run.TRAINING["thermal_time_scales"]) >= 0.1
    assert run.TRAINING["thermal_trajectory_times"] == [0.1, 1.0, 10.0, 100.0]
    assert max(run.TRAINING["thermal_trajectory_times"]) > max(run.TRAINING["thermal_time_scales"])
    assert "geometry_thermal" in run.FILES["model"]
    assert network["width"] >= 16
    assert network["blocks"] >= 1
    assert boundary["samples"] >= 1
    assert boundary["padding"] > 0
    assert 0 < boundary["relative_tolerance"] < 1
    assert formulation["samples"] >= 1
    assert 0 < formulation["relative_tolerance"] < 1
    assert mesh["samples"] >= 1
    assert 0 < mesh["refinement_factor"] < 1
    assert 0 < mesh["relative_tolerance"] < 1
    assert continuity["samples"] >= 1
    assert continuity["translation_step"] > 0
    assert continuity["angle_step"] > 0
    assert 0 < continuity["relative_change_limit"] < 1
    assert final_audit["samples"] >= 1
    assert final_audit["times"] == run.TRAINING["thermal_trajectory_times"]
    assert 0 < final_audit["full_vs_rom_thermal_tolerance"] < 1
    assert 0 < final_audit["tensor_relative_tolerance"] < 1
    assert 0 < final_audit["current_space_relative_tolerance"] < 1
    assert 0 < final_audit["outward_relative_tolerance"] < 1
    assert 0 < final_audit["projection_correction_limit"] < 1
    assert 0 < final_audit["reduced_dynamic_relative_tolerance"] < 1
    assert 0 < final_audit["integrator_relative_tolerance"] < 1
    assert 0 < final_audit["integrator_rtol"] < final_audit["integrator_relative_tolerance"]
    assert 0 < final_audit["integrator_atol"] < final_audit["integrator_rtol"]
    assert 0 < final_audit["integrator_max_step"] <= max(final_audit["times"])
    assert final_audit["circuit_condition_limit"] > 1
    cases = final_audit["operating_cases"]
    assert any("operating" in case for case in cases)
    assert any("drive" in case for case in cases)
    assert "fine_message_steps" not in network
    assert "coarse_levels" not in network
    assert "solver_steps" not in network
    assert "maxwell_residual_tolerance" not in run.PHYSICS
    assert "maxwell_max_iterations" not in run.PHYSICS
    assert "maxwell_restart" not in run.PHYSICS
    assert "n_operator_samples" not in run.TRAINING
    assert "residual_training_steps" not in run.TRAINING
    assert "em_temperature_rise_bounds" not in run.TRAINING


def test_model_artifact_requires_final_release_audit_version():
    from sdfmpneo.unified_model import FORMAT_VERSION

    assert FORMAT_VERSION >= 12


def test_training_defaults_are_matrix_aware_pod_not_krylov_residual_training():
    run = _load_run()
    optimizer = run.TRAINING["optimizer"]
    assert optimizer["batch_size"] > 1
    assert 0 < optimizer["pod_relative_tail_tolerance"] < 1
    assert optimizer["z_weight"] > 0
    assert optimizer["d_weight"] > 0
    assert optimizer["h_weight"] > 0
    assert optimizer["physics_penalty_weight"] >= 0
    assert "krylov_vectors_per_port" not in optimizer
    assert "port_loss_weight" not in optimizer
    assert "random_residual_vectors" not in optimizer
    assert "smooth_residual_vectors" not in optimizer
    assert "final_step_loss_weight" not in optimizer


def test_current_magnitude_phase_are_online_inputs_not_network_inputs():
    run = _load_run()
    assert "operating" in run.PREDICTION
    assert "initial_temperature_rise" in run.PREDICTION
    assert "state_lower" not in run.TRAINING
    assert "state_upper" not in run.TRAINING
    assert "operating_lower" not in run.TRAINING
    assert "operating_upper" not in run.TRAINING