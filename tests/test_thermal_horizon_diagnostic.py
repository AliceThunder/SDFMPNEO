from types import SimpleNamespace

import numpy as np

from sdfmpneo.training.thermal_horizon import (
    diagnostic_from_rank_report,
    finite_horizon_rank_diagnostic,
)


def test_finite_horizon_response_is_bounded_by_steady_envelope():
    lambdas = np.array([0.1, 1.0, 10.0, 100.0])
    steady = np.array([1.0, 0.4, 0.05, 0.005])
    result = finite_horizon_rank_diagnostic(
        lambdas,
        steady,
        horizons=(1.0, 10.0, 100.0),
        relative_tolerance=1e-2,
    )
    previous = None
    for key in ("1s", "10s", "100s"):
        response = np.asarray(result[key]["maximum_modal_response"])
        assert np.all(response <= steady + 1e-15)
        if previous is not None:
            assert np.all(response >= previous - 1e-15)
        previous = response


def test_rank_report_diagnostic_refuses_missing_full_decay_spectrum():
    model = SimpleNamespace(
        thermal_model=SimpleNamespace(lambdas=np.array([0.1, 1.0]))
    )
    report = {
        "maximum_modal_steady_response": [1.0, 0.4, 0.05],
        "full_dimension": 3,
        "relative_tolerance": 1e-3,
    }
    diagnostic = diagnostic_from_rank_report(model, report)
    assert not diagnostic["available"]
    assert "full_spectrum" in diagnostic["reason"]


def test_rank_report_diagnostic_is_available_when_full_rank_is_retained():
    model = SimpleNamespace(
        thermal_model=SimpleNamespace(lambdas=np.array([0.1, 1.0, 10.0]))
    )
    report = {
        "maximum_modal_steady_response": [1.0, 0.4, 0.05],
        "full_dimension": 3,
        "relative_tolerance": 1e-2,
        "absolute_tolerance": 0.0,
        "source_bound_safety_factor": 1.0,
        "boundary_fraction": 0.25,
    }
    diagnostic = diagnostic_from_rank_report(model, report, horizons=(1.0, 10.0))
    assert diagnostic["available"]
    assert set(diagnostic["horizons"]) == {"1s", "10s"}
