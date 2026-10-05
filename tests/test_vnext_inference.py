import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    MeshfreeVNextSystem,
    Scene,
    SuperellipseSpiral,
    run_system_inference,
    scene_to_dict,
)


def _scene():
    return Scene(
        (
            CoilObject(
                SuperellipseSpiral(
                    0.024,
                    0.020,
                    0.75,
                    0.001,
                    0.001,
                    exponent=3.0,
                    conductor_width=1e-3,
                    conductor_thickness=8e-4,
                ),
                ConductorMaterial(
                    5.8e7
                ),
                "coil",
            ),
        ),
        HomogeneousMedium(),
    )


def test_fast_json_inference_covers_ports_spatial_and_continuous_thermal():
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        )
    )
    request = {
        "scene": scene_to_dict(
            _scene()
        ),
        "frequency_hz": 40_000.0,
        "mode": "fast",
        "currents": [
            [2.0, 0.1]
        ],
        "spatial_queries": [
            {
                "coil_index": 0,
                "arc_fraction": 0.5,
                "xy": [0.0, 0.0],
            }
        ],
        "thermal": {
            "medium": {
                "conductivity": 0.6,
                "density": 1000.0,
                "heat_capacity": 4200.0,
                "ambient_temperature": 293.15,
            },
            "time": 5.0,
            "points": [
                [0.0, 0.0, 0.03]
            ],
            "quadrature": {
                "longitudinal_segments": 6,
                "radial_order": 2,
                "angular_order": 8,
            },
        },
    }
    output = run_system_inference(
        system,
        request,
    )
    assert output["mode"] == "fast"
    assert output["n_ports"] == 1
    assert output["power_closure_error"] < 1e-10
    assert output["reciprocity_defect"] < 1e-12
    assert output["coil_power"][0] > 0.0
    assert (
        output["spatial_queries"][0]["joule_density"]
        >= 0.0
    )
    assert (
        output["thermal"]["temperature"][0]
        > 293.15
    )
    assert (
        output["thermal"][
            "source_normalization_closure_error"
        ]
        < 1e-9
    )


def test_inference_rejects_unknown_mode():
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        )
    )
    try:
        run_system_inference(
            system,
            {
                "scene": scene_to_dict(
                    _scene()
                ),
                "frequency_hz": 40_000.0,
                "mode": "magic",
            },
        )
    except ValueError as exc:
        assert "mode" in str(exc)
    else:
        raise AssertionError(
            "unknown inference mode must be rejected"
        )


def test_fast_json_inference_supports_tensor_thermal_history_without_top_level_currents():
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=24,
        )
    )
    request = {
        "scene": scene_to_dict(
            _scene()
        ),
        "frequency_hz": 40_000.0,
        "mode": "fast",
        "thermal": {
            "medium": {
                "conductivity_tensor": [
                    [0.35, 0.04, 0.0],
                    [0.04, 0.65, 0.02],
                    [0.0, 0.02, 0.95],
                ],
                "density": 1000.0,
                "heat_capacity": 4200.0,
                "ambient_temperature": 293.15,
            },
            "points": [
                [0.0, 0.0, 0.03]
            ],
            "history": {
                "interval_edges": [
                    0.0,
                    1.0,
                    3.0,
                ],
                "interval_currents": [
                    [
                        [1.0, 0.0]
                    ],
                    [
                        [0.5, 0.2]
                    ],
                ],
                "observation_times": [
                    0.5,
                    1.5,
                    4.0,
                ],
            },
            "quadrature": {
                "longitudinal_segments": 6,
                "radial_order": 2,
                "angular_order": 8,
            },
        },
    }
    output = run_system_inference(
        system,
        request,
    )
    thermal = output[
        "thermal"
    ]
    history = thermal[
        "history"
    ]
    temperature = np.asarray(
        history[
            "temperature"
        ],
        dtype=float,
    )
    assert temperature.shape == (
        3,
        1,
    )
    assert np.all(
        np.isfinite(
            temperature
        )
    )
    assert (
        temperature[
            0,
            0,
        ]
        > thermal[
            "ambient_temperature"
        ]
    )
    assert (
        temperature[
            -1,
            0,
        ]
        > thermal[
            "ambient_temperature"
        ]
    )
    encoded_currents = np.asarray(
        history[
            "interval_currents"
        ],
        dtype=float,
    )
    assert encoded_currents.shape == (
        2,
        1,
        2,
    )
