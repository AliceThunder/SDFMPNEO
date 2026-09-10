from __future__ import annotations

import numpy as np

from sdfmpneo.training.automatic_thermal_rank_runtime import (
    _resolved_training,
    select_rank_from_modal_response_envelope,
)


def test_modal_response_envelope_selects_smallest_resolved_prefix():
    envelope = np.array([1.0, 5.0e-2, 2.0e-4, 1.0e-6, 1.0e-8])
    rank, report = select_rank_from_modal_response_envelope(
        envelope,
        relative_tolerance=5.0e-4,
        safety_factor=1.0,
        unresolved=False,
    )
    assert rank == 2
    assert report["resolved_tail_norm"] <= report["selection_limit"]


def test_partial_probe_does_not_stop_while_boundary_modes_are_energetic():
    envelope = np.array([1.0, 1.0e-3, 8.0e-4, 7.0e-4])
    rank, _ = select_rank_from_modal_response_envelope(
        envelope,
        relative_tolerance=5.0e-3,
        safety_factor=1.0,
        boundary_fraction=0.1,
        unresolved=True,
    )
    assert rank == len(envelope)


def test_automatic_initial_coordinate_box_expands_to_selected_rank():
    training = {
        "initial_lower": [],
        "initial_upper": [],
        "operating_lower": [0.0, 0.0],
        "operating_upper": [10.0, 10.0],
    }
    resolved = _resolved_training(
        training, 6, {"initial_coordinate_bound": 0.15})
    assert resolved["initial_lower"] == [-0.15] * 6
    assert resolved["initial_upper"] == [0.15] * 6
    assert resolved["operating_lower"] == [0.0, 0.0]


def test_explicit_wrong_dimension_is_rejected_after_automatic_rank_selection():
    training = {
        "initial_lower": [-0.1, -0.1],
        "initial_upper": [0.1, 0.1],
    }
    try:
        _resolved_training(training, 4, {"initial_coordinate_bound": 0.1})
    except ValueError as exc:
        assert "different dimension" in str(exc)
    else:
        raise AssertionError("mismatched explicit modal bounds must not be silently resized")
