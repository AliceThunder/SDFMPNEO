from copy import deepcopy

from sdfmpneo.unified_runtime import _background_signature_view, _signature


def _settings():
    return {
        "BACKGROUND": {
            "bounds": [[-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0]],
            "self_correction": {
                "physical_tolerance": 1.0e-6,
                "linear_transverse_direct_max_dofs": 12000,
                "linear_transverse_direct_fallback_max_dofs": 24000,
            },
        },
        "DEFAULT_GEOMETRY": {"dummy": 1.0},
        "GEOMETRY_SAMPLING": {"dummy": [0.9, 1.1]},
        "PHYSICS": {"frequency_hz": 85000.0},
        "MATERIALS": {"dummy": {"sigma": 1.0}},
        "REGIONS": {"dummy": "material"},
        "TRAINING": {
            "seed": 17,
            "n_tensor_samples": 96,
            "spatial_tensor_schema": "cellwise_joule_tensor_v1",
        },
    }


def test_background_signature_view_filters_solver_only_thresholds_without_mutation():
    settings = _settings()
    original = deepcopy(settings)

    background = _background_signature_view(settings)

    assert settings == original
    assert background["bounds"] == original["BACKGROUND"]["bounds"]
    assert background["self_correction"]["physical_tolerance"] == 1.0e-6
    assert "linear_transverse_direct_max_dofs" not in background["self_correction"]
    assert "linear_transverse_direct_fallback_max_dofs" not in background["self_correction"]


def test_signature_is_stable_across_solver_only_threshold_changes():
    first = _settings()
    second = deepcopy(first)
    second["BACKGROUND"]["self_correction"]["linear_transverse_direct_max_dofs"] = 32000
    second["BACKGROUND"]["self_correction"]["linear_transverse_direct_fallback_max_dofs"] = 64000

    assert _signature(first) == _signature(second)
