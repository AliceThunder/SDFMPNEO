import numpy as np
import pytest

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    PackageObject,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    analytic_port_baseline,
    haar_rotation,
    scene_from_dict,
    scene_to_dict,
)


def _coil():
    return CoilObject(
        SuperellipseSpiral(
            0.025,
            0.021,
            0.7,
            0.001,
            0.001,
            conductor_width=1e-3,
            conductor_thickness=0.8e-3,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )


def test_superquadric_ellipsoid_volume_and_surface_implicit_equation():
    a, b, c = 0.04, 0.03, 0.02
    geometry = SuperquadricPackageGeometry(
        np.array([a, b, c]),
        exponent_xy=2.0,
        exponent_z=2.0,
    )
    assert np.isclose(
        geometry.volume,
        4.0
        * np.pi
        * a
        * b
        * c
        / 3.0,
        rtol=2e-14,
    )
    surface = geometry.surface_points(
        vertical_order=9,
        azimuthal_order=24,
    )
    assert np.max(
        np.abs(
            geometry.implicit(
                surface
            )
        )
    ) < 2e-12


def test_superquadric_common_rigid_motion_preserves_membership():
    geometry = SuperquadricPackageGeometry(
        np.array(
            [0.04, 0.03, 0.02]
        ),
        exponent_xy=4.0,
        exponent_z=3.0,
    )
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, 0.01, 0.005],
            [0.05, 0.0, 0.0],
        ]
    )
    rng = np.random.default_rng(
        51
    )
    pose = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.2, -0.1, 0.4]
        ),
    )
    moved = geometry.transformed(
        pose
    )
    world = pose.apply(
        local
    )
    assert np.array_equal(
        geometry.contains(
            local
        ),
        moved.contains(
            world
        ),
    )


def test_package_scene_round_trip_preserves_continuous_geometry_and_material():
    package = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.05, 0.04, 0.015]
            ),
            exponent_xy=5.0,
            exponent_z=4.0,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                0.25,
                translation=(
                    0.003,
                    -0.002,
                    0.01,
                ),
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=3.4,
            relative_permeability=1.0,
            conductivity=0.002,
            thermal_conductivity=0.25,
            density=1200.0,
            heat_capacity=1500.0,
        ),
        "encapsulation",
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (package,),
    )
    payload = scene_to_dict(
        scene
    )
    assert "packages" in payload
    restored = scene_from_dict(
        payload
    )
    assert len(
        restored.packages
    ) == 1
    loaded = restored.packages[0]
    assert loaded.name == package.name
    assert np.allclose(
        loaded.geometry.half_extents,
        package.geometry.half_extents,
    )
    assert np.allclose(
        loaded.geometry.pose.rotation,
        package.geometry.pose.rotation,
    )
    assert np.allclose(
        loaded.geometry.pose.translation,
        package.geometry.pose.translation,
    )
    assert np.isclose(
        loaded.material.relative_permittivity,
        3.4,
    )
    assert np.isclose(
        loaded.material.conductivity,
        0.002,
    )


def test_current_mvp_solver_rejects_package_instead_of_ignoring_it():
    package = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.05, 0.04, 0.015]
            )
        ),
        IsotropicMaterial(
            relative_permittivity=2.8
        ),
    )
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
        (package,),
    )
    with pytest.raises(
        NotImplementedError,
        match="package coupling",
    ):
        analytic_port_baseline(
            scene,
            85_000.0,
            segments_per_coil=24,
        )


def test_empty_packages_do_not_change_legacy_scene_serialization_shape():
    scene = Scene(
        (_coil(),),
        HomogeneousMedium(),
    )
    payload = scene_to_dict(
        scene
    )
    assert "packages" not in payload
