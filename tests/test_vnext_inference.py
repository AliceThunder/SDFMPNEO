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
