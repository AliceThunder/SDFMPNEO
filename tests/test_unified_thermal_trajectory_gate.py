from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
import pytest

import sdfmpneo.unified_thermal as thermal
from sdfmpneo.unified_runtime import _require_effective_thermal_basis


class SyntheticBackground:
    n_cells = 3

    def validate_geometry(self, geometry):
        return geometry


class SyntheticLibrary:
    time_scales = (0.1, 1.0, 10.0)
    conditioning_limit = 1e10

    def __init__(self, basis):
        self._basis = np.asarray(basis, float)

    def basis_for_geometry(self, background, geometry):
        return self._basis.copy()


def synthetic_cases(background, geometry):
    context = SimpleNamespace(
        line_heat_weights=(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        )
    )
    M = sp.diags([2.0, 3.0, 4.0], format="csr")
    K = sp.csr_matrix(
        np.array(
            [
                [3.0, -1.0, 0.0],
                [-1.0, 3.0, -1.0],
                [0.0, -1.0, 2.0],
            ]
        )
    )
    return context, M, K, ["volume[0]", "wire[0]"], [
        np.array([1.0, 0.2, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ]


def test_full_basis_passes_independent_trajectory_and_steady_audit(monkeypatch):
    monkeypatch.setattr(thermal, "_thermal_trajectory_cases", synthetic_cases)
    error, worst, diagnostics, times = thermal.audit_geometry_aware_thermal_trajectories(
        SyntheticBackground(),
        SyntheticLibrary(np.eye(3)),
        [object()],
    )
    assert times == (0.1, 1.0, 10.0, 100.0)
    assert error < 1e-10
    assert worst["composite_relative_error"] < 1e-10
    assert "volume[0]@steady" in diagnostics
    assert "initial[uniform]@100s" in diagnostics


def test_incomplete_basis_is_detected_by_trajectory_gate(monkeypatch):
    monkeypatch.setattr(thermal, "_thermal_trajectory_cases", synthetic_cases)
    error, worst, diagnostics, _ = thermal.audit_geometry_aware_thermal_trajectories(
        SyntheticBackground(),
        SyntheticLibrary(np.eye(3)[:, :2]),
        [object()],
        times=(0.1, 1.0, 10.0),
    )
    assert error > 5e-2
    assert worst["composite_relative_error"] == pytest.approx(error)
    assert diagnostics


def test_runtime_rejects_pre_trajectory_thermal_cache():
    with pytest.raises(RuntimeError, match="trajectory audit"):
        _require_effective_thermal_basis(
            {
                "converged": True,
                "basis_dimension": 4,
                "maximum_validation_relative_energy_error": 1e-3,
                "target_relative_error": 5e-2,
            }
        )
