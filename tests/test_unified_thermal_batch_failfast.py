import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sdfmpneo.unified_thermal as thermal


def test_grouped_resolvent_errors_match_scalar_evaluation():
    A = sp.diags([2.0, 3.0, 5.0, 7.0], format="csr")
    phi = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, -0.25],
        ],
        dtype=float,
    )
    anchors = []
    for j, b in enumerate(
        (
            np.array([1.0, 2.0, 0.5, -1.0]),
            np.array([0.0, 1.0, 3.0, 0.25]),
            np.array([2.0, -1.0, 0.0, 1.5]),
        )
    ):
        u = np.asarray(spla.spsolve(A, b), float)
        anchors.append(
            {
                "A": A,
                "b": b,
                "u": u,
                "denom2": float(np.dot(u, b)),
                "label": f"a{j}",
                "case_label": f"a{j}",
                "source_kind": "volume",
                "shift": 1.0,
                "geometry_index": 0,
                "rhs_norm": float(np.linalg.norm(b)),
            }
        )

    grouped = thermal._anchor_relative_errors_grouped(
        thermal._group_anchors(anchors), phi
    )
    assert len(grouped) == len(anchors)
    for anchor, grouped_error in grouped:
        scalar_error, _ = thermal._anchor_error(anchor, phi)
        assert grouped_error == pytest.approx(scalar_error, rel=1e-12, abs=1e-12)

    grouped_worst, grouped_anchor, _ = thermal._worst_anchor_grouped(
        thermal._group_anchors(anchors), phi
    )
    scalar_worst, scalar_anchor, _ = thermal._worst_anchor(anchors, phi)
    assert grouped_worst == pytest.approx(scalar_worst, rel=1e-12, abs=1e-12)
    assert grouped_anchor["label"] == scalar_anchor["label"]


def test_validation_resolvent_failure_skips_trajectory_audit(monkeypatch):
    class Geometry:
        n_ports = 1

    class Background:
        n_cells = 2
        cell_volumes = np.ones(2)

        @staticmethod
        def validate_geometry(geometry):
            return geometry

    geometry = Geometry()
    background = Background()

    monkeypatch.setattr(
        thermal,
        "_geometry_anchors",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        thermal,
        "_canonicalize_poses",
        lambda candidate, reference: candidate,
    )

    def fake_greedy(background, anchors, target, maximum_rank, monitor, label):
        if label == "background":
            return np.array([[1.0], [0.0]]), 1, "target_reached", 0.0
        return np.empty((2, 0)), 0, "target_reached", 0.0

    monkeypatch.setattr(thermal, "_greedy_basis", fake_greedy)
    monkeypatch.setattr(
        thermal,
        "_enrich_background_against_full_library",
        lambda *args, **kwargs: (
            np.array([[1.0], [0.0]]),
            0,
            "target_reached",
            0.0,
        ),
    )

    def fake_audit(
        background,
        library,
        geometries,
        shifts,
        monitor,
        role,
        *,
        anchor_sets=None,
    ):
        if role == "held-out validation":
            return 0.2, {"label": "held-out"}, {"volume@0.1s": 0.2}, 0
        return 0.0, {}, {}, 0

    monkeypatch.setattr(thermal, "_audit_geometries", fake_audit)

    def forbidden_trajectory(*args, **kwargs):
        raise AssertionError("trajectory audit must be skipped after resolvent failure")

    monkeypatch.setattr(
        thermal,
        "audit_geometry_aware_thermal_trajectories",
        forbidden_trajectory,
    )

    _, report = thermal.build_geometry_aware_thermal_library(
        background,
        geometry,
        [geometry],
        validation_geometries=[geometry],
        target_relative_error=0.05,
        time_scales=(0.1,),
        trajectory_times=(0.1, 1.0),
    )

    assert report.converged is False
    assert report.stop_reason == "validation_target_not_met"
    assert report.maximum_validation_relative_energy_error == pytest.approx(0.2)
    assert report.maximum_validation_trajectory_relative_error == 0.0
    assert report.trajectory_diagnostics["audit_skipped_validation_energy_error"] == pytest.approx(0.2)
