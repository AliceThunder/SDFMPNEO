import numpy as np
import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    PackageObject,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    encode_hybrid_scene_invariant,
    mvp_system_capabilities,
    scene_from_dict,
    scene_to_dict,
)
from sdfmpneo_vnext.hybrid_neural import (
    _dielectric_loss_gate,
)
from sdfmpneo_vnext.hybrid_spatial_neural import (
    _package_loss_gate,
)


def _coil():
    return CoilObject(
        SuperellipseSpiral(
            0.022,
            0.019,
            0.7,
            0.001,
            0.001,
            conductor_width=8e-4,
            conductor_thickness=6e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )


def _package(material):
    return PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [0.030, 0.026, 0.008]
            ),
            exponent_xy=3.0,
            exponent_z=3.0,
            pose=RigidPose(
                np.eye(3),
                np.asarray(
                    [0.0, 0.0, 0.004]
                ),
            ),
        ),
        material,
        "package",
    )


def test_debye_material_is_passive_and_has_correct_limits():
    material = DebyeMaterial(
        relative_permittivity_static=4.0,
        relative_permittivity_infinite=2.0,
        relaxation_time=1e-6,
    )
    low = (
        material.relative_permittivity_at(
            10.0
        )
    )
    high = (
        material.relative_permittivity_at(
            1.0e10
        )
    )
    mid_frequency = (
        1.0
        / (
            2.0
            * np.pi
            * material.relaxation_time
        )
    )
    mid = (
        material.relative_permittivity_at(
            mid_frequency
        )
    )

    assert np.isclose(
        low.real,
        4.0,
        rtol=2e-8,
    )
    assert np.isclose(
        high.real,
        2.0,
        rtol=2e-8,
    )
    assert mid.imag < 0.0
    assert (
        material.loss_conductivity(
            mid_frequency
        )
        > 0.0
    )


def test_debye_material_rejects_nonpassive_relaxation_strength():
    with pytest.raises(
        ValueError,
        match="Debye",
    ):
        DebyeMaterial(
            relative_permittivity_static=2.0,
            relative_permittivity_infinite=3.0,
            relaxation_time=1e-6,
        )


def test_debye_package_serialization_round_trip():
    material = DebyeMaterial(
        relative_permittivity_static=5.2,
        relative_permittivity_infinite=2.1,
        relaxation_time=2.5e-6,
        relative_permeability=1.05,
        conductivity=2e-4,
        thermal_conductivity=0.35,
        density=1150.0,
        heat_capacity=900.0,
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (_package(material),),
    )
    payload = scene_to_dict(
        scene
    )
    package_material = (
        payload[
            "packages"
        ][0][
            "material"
        ]
    )
    assert (
        package_material[
            "model"
        ]
        == "debye"
    )

    restored = scene_from_dict(
        payload
    )
    actual = (
        restored.packages[
            0
        ].material
    )
    assert isinstance(
        actual,
        DebyeMaterial,
    )
    assert np.isclose(
        actual.relative_permittivity_static,
        material.relative_permittivity_static,
    )
    assert np.isclose(
        actual.relative_permittivity_infinite,
        material.relative_permittivity_infinite,
    )
    assert np.isclose(
        actual.relaxation_time,
        material.relaxation_time,
    )
    assert np.isclose(
        actual.conductivity,
        material.conductivity,
    )


def test_constant_package_serialization_keeps_legacy_material_shape():
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (
            _package(
                IsotropicMaterial(
                    relative_permittivity=3.0,
                    conductivity=0.002,
                )
            ),
        ),
    )
    material = (
        scene_to_dict(
            scene
        )[
            "packages"
        ][0][
            "material"
        ]
    )
    assert "model" not in material
    assert (
        material[
            "relative_permittivity"
        ]
        == 3.0
    )


def test_hybrid_features_and_loss_gates_use_frequency_dependent_debye_response():
    material = DebyeMaterial(
        relative_permittivity_static=6.0,
        relative_permittivity_infinite=2.0,
        relaxation_time=2e-6,
        conductivity=0.0,
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (_package(material),),
    )
    low = (
        encode_hybrid_scene_invariant(
            scene,
            1.0e3,
        ).package_features[
            0
        ]
    )
    high = (
        encode_hybrid_scene_invariant(
            scene,
            1.0e7,
        ).package_features[
            0
        ]
    )
    assert not np.allclose(
        low,
        high,
    )

    relaxation_frequency = (
        1.0
        / (
            2.0
            * np.pi
            * material.relaxation_time
        )
    )
    assert (
        _dielectric_loss_gate(
            scene,
            relaxation_frequency,
        )
        == 1.0
    )
    assert np.array_equal(
        _package_loss_gate(
            scene,
            relaxation_frequency,
            np.asarray(
                [0],
                dtype=int,
            ),
        ),
        np.asarray(
            [1.0]
        ),
    )


def test_capabilities_report_existing_package_em_coupling_truthfully():
    capabilities = (
        mvp_system_capabilities()
    )
    assert (
        capabilities.package_dielectric_sie
    )
    assert (
        capabilities.package_em_coupling
    )
    assert not (
        capabilities.lossy_background_media
    )
    assert not (
        capabilities.retardation
    )
