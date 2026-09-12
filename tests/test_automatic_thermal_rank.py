from pathlib import Path

import numpy as np
import pytest

from sdfmpneo.training.automatic_thermal_rank import (
    _load_selection_cache,
    _save_selection_cache,
    _thermal_rank_cache_path,
    _thermal_selection_cache_key,
    resolve_training_bounds,
    select_rank_from_modal_response_envelope,
)


def test_modal_response_tail_selects_smallest_resolved_prefix():
    envelope = np.array([1.0, 0.1, 1e-5, 1e-6])
    rank, report = select_rank_from_modal_response_envelope(
        envelope,
        relative_tolerance=1e-3,
        safety_factor=2.0,
        boundary_fraction=0.25,
        unresolved=False,
    )
    assert rank == 2
    assert report["resolved_tail_norm"] < report["selection_limit"]


def test_unresolved_boundary_prevents_false_early_truncation():
    envelope = np.array([1.0, 0.1, 1e-6, 2e-4])
    rank, _ = select_rank_from_modal_response_envelope(
        envelope,
        relative_tolerance=1e-3,
        safety_factor=2.0,
        boundary_fraction=0.25,
        unresolved=True,
    )
    assert rank == len(envelope)


def test_automatic_initial_box_expands_to_selected_rank():
    training = {
        "initial_lower": [],
        "initial_upper": [],
        "operating_lower": [0.0],
        "operating_upper": [1.0],
    }
    resolved = resolve_training_bounds(
        training, 5, {"initial_coordinate_bound": 0.2}
    )
    assert resolved["initial_lower"] == [-0.2] * 5
    assert resolved["initial_upper"] == [0.2] * 5


def test_explicit_initial_box_must_match_selected_rank():
    with pytest.raises(ValueError, match="thermal rank selected 3"):
        resolve_training_bounds(
            {"initial_lower": [-1, -1], "initial_upper": [1, 1]},
            3,
            {},
        )


def _cache_config(tmp_path):
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"same mesh")
    return tmp_path / "model.config.json", {
        "mesh": "mesh.msh",
        "frequency_hz": 1e5,
        "ambient_temperature": 293.15,
        "constitutive_relative_error": 1e-8,
        "materials": {
            "1": {
                "name": "material",
                "electrical_conductivity": 1.0,
                "thermal_conductivity": 2.0,
                "volumetric_heat_capacity": 3.0,
            }
        },
        "terminal_pairs": [[1, 2]],
        "port_names": ["p"],
        "current_offset": None,
        "current_matrix": None,
        "thermal_truncation": {
            "relative_tolerance": 1e-3,
            "cache": True,
        },
        "training": {
            "operating_lower": [0.0],
            "operating_upper": [1.0],
        },
    }


def test_selection_cache_round_trip_and_mesh_invalidation(tmp_path):
    config_path, config = _cache_config(tmp_path)
    key = _thermal_selection_cache_key(config_path, config)
    cache_path = _thermal_rank_cache_path(
        config_path, config["thermal_truncation"]
    )
    report = {
        "selected_rank": 198,
        "recommended_restart_coordinate_bound": [0.2] * 198,
    }
    _save_selection_cache(cache_path, key, 198, report)
    rank, cached = _load_selection_cache(cache_path, key)
    assert rank == 198
    assert cached["cache_hit"] is True
    assert cached["selection_cache_hit"] is True
    assert Path(cached["cache_path"]) == cache_path

    (tmp_path / "mesh.msh").write_bytes(b"changed mesh")
    changed_key = _thermal_selection_cache_key(config_path, config)
    assert changed_key != key
    assert _load_selection_cache(cache_path, changed_key) is None


def test_selection_cache_invalidates_when_physics_changes(tmp_path):
    config_path, config = _cache_config(tmp_path)
    original = _thermal_selection_cache_key(config_path, config)
    config["materials"]["1"]["thermal_conductivity"] = 4.0
    assert _thermal_selection_cache_key(config_path, config) != original
