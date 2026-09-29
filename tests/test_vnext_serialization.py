import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    PackageObject,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    scene_from_dict,
    scene_to_dict,
)


def test_tensor_thermal_package_material_round_trips_through_scene_serialization():
    tensor = np.asarray(
        [
            [0.42, 0.08, 0.0],
            [0.08, 0.73, 0.03],
            [0.0, 0.03, 1.10],
        ],
        dtype=float,
    )
    coil = CoilObject(
        SuperellipseSpiral(
            0.015,
            0.013,
            0.60,
            0.001,
            0.001,
            conductor_width=8.0e-4,
            conductor_thickness=6.0e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )
    package = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [
                    0.025,
                    0.022,
                    0.006,
                ]
            )
        ),
        IsotropicMaterial(
            relative_permittivity=3.0,
            conductivity=1.0e-4,
            thermal_conductivity_tensor=tensor,
            density=1180.0,
            heat_capacity=1350.0,
        ),
        "tensor-package",
    )
    scene = Scene(
        (
            coil,
        ),
        HomogeneousMedium(),
        (
            package,
        ),
    )

    payload = scene_to_dict(
        scene
    )
    stored = payload[
        "packages"
    ][
        0
    ][
        "material"
    ][
        "thermal_conductivity_tensor"
    ]
    assert np.allclose(
        np.asarray(
            stored,
            dtype=float,
        ),
        tensor,
    )

    restored = scene_from_dict(
        payload
    )
    material = restored.packages[
        0
    ].material
    assert (
        material.thermal_conductivity
        is None
    )
    assert np.allclose(
        material.thermal_conductivity_tensor,
        tensor,
    )
    assert np.isclose(
        material.density,
        1180.0,
    )
    assert np.isclose(
        material.heat_capacity,
        1350.0,
    )
