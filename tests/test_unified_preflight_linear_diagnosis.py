import sdfmpneo.unified_corrected_truth_preflight as corrected


def test_local_self_linear_solve_failure_has_distinct_diagnosis():
    report = {
        "linear_solver_converged": False,
        "maximum_linear_relative_residual": 4.0e-6,
        "linear_relative_residual_tolerance": 1.0e-9,
        "fine_step": 0.003,
        "validation_fine_step": 0.00225,
        "maximum_relative_error": 4.8,
        "maximum_joule_total_power_relative_error": 1.0e-15,
        "joule_identity_tolerance": 1.0e-10,
    }
    diagnosis = corrected._local_self_failure_diagnosis(report)
    assert diagnosis["code"] == "local_self_linear_solve_not_converged"
    assert diagnosis["maximum_linear_relative_residual"] == 4.0e-6
    assert diagnosis["linear_relative_residual_tolerance"] == 1.0e-9
