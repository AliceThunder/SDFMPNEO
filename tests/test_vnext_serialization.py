import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    MultiDebyeMaterial,
    PackageObject,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    TabulatedMaterial,
    TensorElectricMaterial,
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


def _serialization_base_scene(
    material,
):
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
        material,
        "tensor-package",
    )
    return Scene(
        (
            coil,
        ),
        HomogeneousMedium(),
        (
            package,
        ),
    )


def test_tensor_thermal_dispersive_package_materials_round_trip():
    tensor = np.asarray(
        [
            [0.31, 0.04, 0.01],
            [0.04, 0.62, 0.02],
            [0.01, 0.02, 0.94],
        ],
        dtype=float,
    )
    materials = (
        DebyeMaterial(
            relative_permittivity_static=8.0,
            relative_permittivity_infinite=3.2,
            relaxation_time=1.5e-6,
            relative_permeability=1.1,
            conductivity=2.0e-4,
            density=1170.0,
            heat_capacity=1420.0,
            thermal_conductivity_tensor=tensor,
        ),
        MultiDebyeMaterial(
            relative_permittivity_infinite=2.5,
            relaxation_strengths=(
                1.3,
                2.1,
            ),
            relaxation_times=(
                4.0e-7,
                2.0e-6,
            ),
            relative_permeability=1.2,
            conductivity=1.5e-4,
            density=1190.0,
            heat_capacity=1380.0,
            thermal_conductivity_tensor=tensor,
        ),
        TabulatedMaterial(
            frequencies_hz=(
                10_000.0,
                100_000.0,
                1_000_000.0,
            ),
            relative_permittivity_real=(
                5.5,
                4.2,
                3.1,
            ),
            loss_conductivity_values=(
                1.0e-5,
                4.0e-5,
                9.0e-5,
            ),
            relative_permeability=1.05,
            conductivity=8.0e-5,
            density=1210.0,
            heat_capacity=1320.0,
            thermal_conductivity_tensor=tensor,
        ),
    )

    for material in materials:
        scene = _serialization_base_scene(
            material
        )
        restored = scene_from_dict(
            scene_to_dict(
                scene
            )
        )
        actual = restored.packages[
            0
        ].material
        assert type(
            actual
        ) is type(
            material
        )
        assert np.allclose(
            actual.thermal_conductivity_tensor,
            tensor,
        )
        assert (
            actual.thermal_conductivity
            is None
        )
        assert np.isclose(
            actual.density,
            material.density,
        )
        assert np.isclose(
            actual.heat_capacity,
            material.heat_capacity,
        )
        probe = 85_000.0
        assert np.isclose(
            actual.relative_permittivity_at(
                probe
            ),
            material.relative_permittivity_at(
                probe
            ),
        )
        assert np.isclose(
            actual.loss_conductivity(
                probe
            ),
            material.loss_conductivity(
                probe
            ),
        )


def test_tensor_electric_background_and_package_round_trip():
    epsilon_background = np.diag([2.0, 3.0, 4.0])
    sigma_background = np.diag([1.0e-5, 2.0e-5, 3.0e-5])
    epsilon_package = np.diag([4.0, 5.0, 6.0])
    sigma_package = np.diag([2.0e-5, 3.0e-5, 4.0e-5])
    base = _serialization_base_scene(
        TensorElectricMaterial(
            relative_permittivity_tensor=epsilon_package,
            conductivity_tensor=sigma_package,
            relative_permeability=1.15,
        )
    )
    scene = Scene(
        base.coils,
        TensorElectricMaterial(
            relative_permittivity_tensor=epsilon_background,
            conductivity_tensor=sigma_background,
            relative_permeability=1.05,
            thermal_conductivity_tensor=np.diag([0.3, 0.5, 0.8]),
            density=1020.0,
            heat_capacity=1250.0,
        ),
        base.packages,
    )

    payload = scene_to_dict(scene)
    assert payload["medium"]["model"] == "tensor_electric"
    assert payload["packages"][0]["material"]["model"] == "tensor_electric"
    restored = scene_from_dict(payload)

    assert isinstance(restored.medium, TensorElectricMaterial)
    assert isinstance(restored.packages[0].material, TensorElectricMaterial)
    assert np.allclose(
        restored.medium.relative_permittivity_tensor,
        epsilon_background,
    )
    assert np.allclose(
        restored.medium.conductivity_tensor,
        sigma_background,
    )
    assert np.allclose(
        restored.medium.thermal_conductivity_tensor,
        np.diag([0.3, 0.5, 0.8]),
    )
    assert np.allclose(
        restored.packages[0].material.relative_permittivity_tensor,
        epsilon_package,
    )
    assert np.allclose(
        restored.packages[0].material.conductivity_tensor,
        sigma_package,
    )
