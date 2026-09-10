import numpy as np
import pytest

from sdfmpneo.training.automatic_thermal_rank import (
    resolve_training_bounds,
    select_rank_from_modal_response_envelope,
)


def test_modal_response_tail_selects_smallest_resolved_prefix():
    envelope = np.array([1.0, 0.1, 1e-5, 1e-6])
    rank, report = select_rank_from_modal_response_envelope(
        envelope, relative_tolerance=1e-3, safety_factor=2.0,
        boundary_fraction=0.25, unresolved=False)
    assert rank == 2
    assert report["resolved_tail_norm"] < report["selection_limit"]


def test_unresolved_boundary_prevents_false_early_truncation():
    envelope = np.array([1.0, 0.1, 1e-6, 2e-4])
    rank, _ = select_rank_from_modal_response_envelope(
        envelope, relative_tolerance=1e-3, safety_factor=2.0,
        boundary_fraction=0.25, unresolved=True)
    assert rank == len(envelope)


def test_automatic_initial_box_expands_to_selected_rank():
    training = {"initial_lower": [], "initial_upper": [], "operating_lower": [0.0], "operating_upper": [1.0]}
    resolved = resolve_training_bounds(training, 5, {"initial_coordinate_bound": 0.2})
    assert resolved["initial_lower"] == [-0.2] * 5
    assert resolved["initial_upper"] == [0.2] * 5


def test_explicit_initial_box_must_match_selected_rank():
    with pytest.raises(ValueError, match="thermal rank selected 3"):
        resolve_training_bounds({"initial_lower": [-1, -1], "initial_upper": [1, 1]}, 3, {})
