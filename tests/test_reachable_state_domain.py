import json

import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.domain import (
    load_reachable_state_domain_report,
    probe_reachable_state_domain,
    validated_state_bounds,
)


def _field(state, geometry, operating):
    del geometry
    u = float(np.asarray(operating)[0])
    a = np.asarray(state, dtype=float)
    return np.array([-a[0] + u, -2.0 * a[1] + 0.5 * u])


def _jacobian(state, geometry, operating):
    del state, geometry, operating
    return np.array([[-1.0, 0.0], [0.0, -2.0]])


def test_reachable_domain_probe_supports_fixed_initial_state_and_steady_roots(tmp_path):
    report = probe_reachable_state_domain(
        _field,
        initial_lower=np.zeros(2),
        initial_upper=np.zeros(2),
        geometry_lower=np.empty(0),
        geometry_upper=np.empty(0),
        operating_lower=np.array([0.0]),
        operating_upper=np.array([2.0]),
        trajectory_count=12,
        samples_per_trajectory=8,
        time_horizon=8.0,
        time_min=1e-4,
        seed=7,
        include_steady_state=True,
        physical_jacobian_factory=_jacobian,
        margin_fraction=0.1,
        absolute_margin=1e-6,
        physical_signature="physics-v1",
    )
    assert report.thermal_rank == 2
    assert report.successful_steady_states == 12
    assert report.failed_steady_states == 0
    assert np.all(report.suggested_state_lower < report.observed_state_lower)
    assert np.all(report.suggested_state_upper > report.observed_state_upper)
    assert np.all(report.suggested_state_lower <= 0.0)
    assert np.all(report.suggested_state_upper > 0.0)

    path = tmp_path / "domain.json"
    path.write_text(
        json.dumps(
            {
                "kind": "reachable_state_domain",
                "report": {
                    key: (value.tolist() if isinstance(value, np.ndarray) else value)
                    for key, value in report.__dict__.items()
                },
            }
        ),
        encoding="utf-8",
    )
    loaded = load_reachable_state_domain_report(path)
    lower, upper = validated_state_bounds(
        loaded,
        expected_physical_signature="physics-v1",
        geometry_lower=np.empty(0),
        geometry_upper=np.empty(0),
        operating_lower=np.array([0.0]),
        operating_upper=np.array([2.0]),
        thermal_rank=2,
    )
    np.testing.assert_allclose(lower, report.suggested_state_lower, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(upper, report.suggested_state_upper, rtol=0.0, atol=0.0)


def test_reachable_domain_report_is_fail_closed_on_signature_or_domain_mismatch():
    report = probe_reachable_state_domain(
        _field,
        initial_lower=np.zeros(2),
        initial_upper=np.zeros(2),
        geometry_lower=np.empty(0),
        geometry_upper=np.empty(0),
        operating_lower=np.array([0.0]),
        operating_upper=np.array([1.0]),
        trajectory_count=4,
        samples_per_trajectory=4,
        time_horizon=2.0,
        physical_signature="physics-v1",
        include_steady_state=False,
    )
    with pytest.raises(ValueError, match="physical signature"):
        validated_state_bounds(
            report,
            expected_physical_signature="physics-v2",
            geometry_lower=np.empty(0),
            geometry_upper=np.empty(0),
            operating_lower=np.array([0.0]),
            operating_upper=np.array([1.0]),
            thermal_rank=2,
        )
    with pytest.raises(ValueError, match="operating domain"):
        validated_state_bounds(
            report,
            expected_physical_signature="physics-v1",
            geometry_lower=np.empty(0),
            geometry_upper=np.empty(0),
            operating_lower=np.array([0.0]),
            operating_upper=np.array([2.0]),
            thermal_rank=2,
        )
