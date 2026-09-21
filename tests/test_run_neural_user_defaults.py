import copy
import importlib.util
from pathlib import Path


def _load_run():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("sdfmpneo_run_tensor_defaults", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_defaults_use_spatial_joule_and_geometry_local_thermal_rom():
    run = _load_run()
    network = run.TRAINING["network"]
    boundary = run.BACKGROUND["open_boundary_check"]
    formulation = run.BACKGROUND["formulation_check"]
    mesh = run.BACKGROUND["mesh_check"]
    self_correction = run.BACKGROUND["self_correction"]
    continuity = run.BACKGROUND["geometry_continuity_check"]
    final_audit = run.TRAINING["final_audit"]
    assert run.TRAINING["device"] == "cuda"
    assert run.TRAINING["n_tensor_samples"] >= 6
    assert run.TRAINING["spatial_tensor_schema"] == "cellwise_joule_tensor_v1"
    assert run.TRAINING["online_thermal_relative_tolerance"] > 0
    assert run.TRAINING["online_thermal_conditioning_limit"] > 1
    assert "basis_samples" not in run.TRAINING
    assert "basis_design_pool_multiplier" not in run.TRAINING
    assert "basis_validation_samples" not in run.TRAINING
    assert "thermal_basis_schema" not in run.TRAINING
    assert "thermal_basis_design" not in run.TRAINING
    assert "thermal_basis_max_rank" not in run.TRAINING
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
    assert self_correction["enabled"] is True
    assert self_correction["samples"] >= 1
    assert 0 < self_correction["validation_fine_step"] < self_correction["fine_step"] < run.BACKGROUND["fine_step"]
    assert 0 < self_correction["relative_tolerance"] < 1
    assert self_correction["boundary_padding"] > 0
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


def test_model_artifact_requires_integrator_certified_release_version():
    from sdfmpneo.unified_model import FORMAT_VERSION

    assert FORMAT_VERSION >= 13


def test_physical_cache_requires_local_self_correction_certification_semantics():
    from sdfmpneo.unified_runtime import _CACHE_FORMAT

    assert _CACHE_FORMAT >= 18


def test_gate_sample_count_includes_local_self_reference_audit():
    from sdfmpneo.unified_runtime import _gate_sample_count

    run = _load_run()
    settings = copy.deepcopy(run.SETTINGS)
    settings["BACKGROUND"]["open_boundary_check"]["samples"] = 1
    settings["BACKGROUND"]["formulation_check"]["samples"] = 1
    settings["BACKGROUND"]["mesh_check"]["samples"] = 1
    settings["BACKGROUND"]["geometry_continuity_check"]["samples"] = 1
    settings["BACKGROUND"]["self_correction"]["samples"] = 4
    assert _gate_sample_count(settings) == 4


def test_final_release_settings_do_not_invalidate_physical_truth_cache():
    from sdfmpneo.unified_runtime import _signature

    run = _load_run()
    baseline = copy.deepcopy(run.SETTINGS)
    release_only = copy.deepcopy(baseline)
    release_only["TRAINING"]["final_audit"]["tensor_relative_tolerance"] *= 0.5
    release_only["TRAINING"]["optimizer"]["epochs"] += 1
    release_only["TRAINING"]["network"]["width"] += 8
    release_only["TRAINING"]["device"] = "cpu"
    assert _signature(baseline) == _signature(release_only)

    online_policy_change = copy.deepcopy(baseline)
    online_policy_change["TRAINING"]["thermal_time_scales"][0] *= 2.0
    online_policy_change["TRAINING"]["online_thermal_relative_tolerance"] *= 0.5
    assert _signature(baseline) == _signature(online_policy_change)

    truth_sampling_change = copy.deepcopy(baseline)
    truth_sampling_change["TRAINING"]["n_tensor_samples"] += 1
    assert _signature(baseline) != _signature(truth_sampling_change)

    correction_change = copy.deepcopy(baseline)
    correction_change["BACKGROUND"]["self_correction"]["fine_step"] *= 0.9
    assert _signature(baseline) != _signature(correction_change)


def test_training_defaults_are_matrix_aware_pod_not_krylov_residual_training():
    run = _load_run()
    optimizer = run.TRAINING["optimizer"]
    assert optimizer["batch_size"] > 1
    assert 0 < optimizer["pod_relative_tail_tolerance"] < 1
    assert optimizer["z_weight"] > 0
    assert optimizer["d_weight"] > 0
    assert optimizer["spatial_weight"] > 0
    assert "h_weight" not in optimizer
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


def test_online_thermal_policy_is_explicit_and_global_atlas_is_absent():
    run = _load_run()
    assert run.TRAINING["spatial_tensor_schema"] == "cellwise_joule_tensor_v1"
    assert 0 < run.TRAINING["online_thermal_relative_tolerance"] < 1
    assert run.TRAINING["online_thermal_conditioning_limit"] > 1
    for key in (
        "thermal_component_target_multiplier",
        "thermal_basis_design",
        "thermal_basis_schema",
        "basis_samples",
        "basis_validation_samples",
    ):
        assert key not in run.TRAINING


def test_online_thermal_policy_does_not_invalidate_em_preflight_signature():
    from sdfmpneo.unified_runtime import _preflight_signature

    run = _load_run()
    baseline = copy.deepcopy(run.SETTINGS)

    thermal_policy = copy.deepcopy(baseline)
    thermal_policy["TRAINING"]["online_thermal_relative_tolerance"] *= 0.5
    thermal_policy["TRAINING"]["online_thermal_conditioning_limit"] *= 0.5
    thermal_policy["TRAINING"]["thermal_time_scales"][0] *= 2.0
    assert _preflight_signature(baseline) == _preflight_signature(thermal_policy)

    em_physics = copy.deepcopy(baseline)
    em_physics["PHYSICS"]["frequency_hz"] *= 1.01
    assert _preflight_signature(baseline) != _preflight_signature(em_physics)
