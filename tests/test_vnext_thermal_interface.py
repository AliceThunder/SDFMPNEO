import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    AnisotropicThermalMedium,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    HomogeneousThermalMedium,
    IsotropicMaterial,
    MeshfreeVNextSystem,
    MQSConfig,
    PackageObject,
    PreparedMultiThermalInterfaceField,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    ThermalSourceQuadrature,
    haar_rotation,
    scene_thermal_package_media,
)


def _source(
    positions,
):
    positions = np.asarray(
        positions,
        dtype=float,
    )
    count = len(
        positions
    )
    weights = np.full(
        count,
        1.0e-6,
        dtype=float,
    )
    matrices = np.full(
        (
            count,
            1,
            1,
        ),
        1.0e6,
        dtype=complex,
    )
    return ThermalSourceQuadrature(
        positions=positions,
        volume_weights=weights,
        coil_index=np.zeros(
            count,
            dtype=int,
        ),
        arc_fraction=np.zeros(
            count,
            dtype=float,
        ),
        xy=np.zeros(
            (
                count,
                2,
            ),
            dtype=float,
        ),
        dissipation_matrices=(
            matrices
        ),
        effective_radius=np.full(
            count,
            1.5e-3,
            dtype=float,
        ),
        normalization_closure_error=0.0,
        normalization_correction=1.0,
    )


def _geometry(
    center,
):
    return SuperquadricPackageGeometry(
        np.asarray(
            [
                0.012,
                0.010,
                0.009,
            ]
        ),
        exponent_xy=2.0,
        exponent_z=2.0,
        pose=RigidPose(
            np.eye(
                3
            ),
            np.asarray(
                center,
                dtype=float,
            ),
        ),
    )


def _background():
    return HomogeneousThermalMedium(
        conductivity=0.6,
        density=1000.0,
        heat_capacity=4000.0,
        ambient_temperature=293.15,
    )


def _regions(
    first,
    second,
):
    return (
        (
            0,
            first,
            HomogeneousThermalMedium(
                conductivity=0.22,
                density=1200.0,
                heat_capacity=1800.0,
                ambient_temperature=293.15,
            ),
        ),
        (
            1,
            second,
            HomogeneousThermalMedium(
                conductivity=1.4,
                density=900.0,
                heat_capacity=2400.0,
                ambient_temperature=293.15,
            ),
        ),
    )


def _field(
    first,
    second,
    source,
):
    return PreparedMultiThermalInterfaceField(
        source,
        _background(),
        _regions(
            first,
            second,
        ),
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        mfs_offset_fraction=0.12,
        stehfest_order=6,
        interface_residual_tolerance=2e-3,
        svd_rcond=1e-10,
    )


def test_multi_package_thermal_interface_has_finite_steady_and_transient_response():
    first = _geometry(
        (
            -0.028,
            0.0,
            0.0,
        )
    )
    second = _geometry(
        (
            0.028,
            0.0,
            0.0,
        )
    )
    source = _source(
        (
            (
                -0.028,
                0.0,
                0.0,
            ),
            (
                0.028,
                0.0,
                0.0,
            ),
        )
    )
    field = _field(
        first,
        second,
        source,
    )
    query = np.asarray(
        [
            [
                -0.028,
                0.0,
                0.0,
            ],
            [
                0.028,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.035,
            ],
        ]
    )
    currents = np.asarray(
        [
            1.0
            + 0.0j
        ]
    )

    steady = field.steady_temperature(
        query,
        currents,
    )
    transient = field.temperature_step(
        query,
        5.0,
        currents,
    )
    assert np.all(
        np.isfinite(
            steady
        )
    )
    assert np.all(
        np.isfinite(
            transient
        )
    )
    assert np.all(
        steady
        > field.medium.ambient_temperature
    )
    assert np.all(
        transient
        > field.medium.ambient_temperature
    )
    assert (
        field.maximum_interface_residual
        <= field.interface_residual_tolerance
    )


def test_multi_package_thermal_interface_is_common_se3_invariant():
    first = _geometry(
        (
            -0.028,
            0.0,
            0.0,
        )
    )
    second = _geometry(
        (
            0.028,
            0.0,
            0.0,
        )
    )
    source_positions = np.asarray(
        [
            [
                -0.028,
                0.0,
                0.0,
            ],
            [
                0.028,
                0.0,
                0.0,
            ],
        ]
    )
    field = _field(
        first,
        second,
        _source(
            source_positions
        ),
    )
    query = np.asarray(
        [
            [
                -0.028,
                0.0,
                0.002,
            ],
            [
                0.0,
                0.0,
                0.030,
            ],
        ]
    )
    currents = np.asarray(
        [
            0.9
            - 0.2j
        ]
    )
    reference_steady = (
        field.steady_temperature(
            query,
            currents,
        )
    )
    reference_step = (
        field.temperature_step(
            query,
            3.0,
            currents,
        )
    )

    rng = np.random.default_rng(
        991
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [
                0.17,
                -0.11,
                0.29,
            ]
        ),
    )
    moved_first = (
        first.transformed(
            common
        )
    )
    moved_second = (
        second.transformed(
            common
        )
    )
    moved_source = _source(
        common.apply(
            source_positions
        )
    )
    moved_field = _field(
        moved_first,
        moved_second,
        moved_source,
    )
    moved_query = common.apply(
        query
    )
    actual_steady = (
        moved_field.steady_temperature(
            moved_query,
            currents,
        )
    )
    actual_step = (
        moved_field.temperature_step(
            moved_query,
            3.0,
            currents,
        )
    )
    assert np.allclose(
        actual_steady,
        reference_steady,
        rtol=2e-6,
        atol=2e-7,
    )
    assert np.allclose(
        actual_step,
        reference_step,
        rtol=2e-5,
        atol=2e-6,
    )


def test_system_reference_continuous_thermal_field_dispatches_multiple_package_interfaces():
    coil = CoilObject(
        SuperellipseSpiral(
            0.020,
            0.018,
            0.65,
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
    first_geometry = _geometry(
        (
            -0.045,
            0.0,
            0.0,
        )
    )
    second_geometry = _geometry(
        (
            0.045,
            0.0,
            0.0,
        )
    )
    first_package = PackageObject(
        first_geometry,
        IsotropicMaterial(
            relative_permittivity=1.0,
            conductivity=0.0,
            thermal_conductivity=0.24,
            density=1180.0,
            heat_capacity=1700.0,
        ),
        "first",
    )
    second_package = PackageObject(
        second_geometry,
        IsotropicMaterial(
            relative_permittivity=1.0,
            conductivity=0.0,
            thermal_conductivity=1.25,
            density=920.0,
            heat_capacity=2300.0,
        ),
        "second",
    )
    scene = Scene(
        (
            coil,
        ),
        HomogeneousMedium(),
        (
            first_package,
            second_package,
        ),
    )
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=20,
        ),
        reference_config=MQSConfig(
            segments_per_turn=6,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
        dielectric_surface_vertical_order=6,
        dielectric_surface_azimuthal_order=12,
    )
    field = (
        system.reference_continuous_thermal_field(
            scene,
            40_000.0,
            _background(),
            longitudinal_segments=6,
            radial_order=3,
            angular_order=8,
            package_axial_order=2,
            package_radial_order=2,
            package_azimuthal_order=8,
            interface_vertical_order=4,
            interface_azimuthal_order=8,
            mfs_offset_fraction=0.12,
            stehfest_order=6,
            interface_residual_tolerance=5e-3,
            svd_rcond=1e-13,
        )
    )
    assert isinstance(
        field,
        PreparedMultiThermalInterfaceField,
    )
    currents = np.asarray(
        [
            1.0
            + 0.0j
        ]
    )
    temperature = field.temperature_step(
        np.asarray(
            [
                0.0,
                0.0,
                0.025,
            ]
        ),
        2.0,
        currents,
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        > field.medium.ambient_temperature
    )
    assert (
        field.maximum_interface_residual
        <= field.interface_residual_tolerance
    )


def _nested_thermal_geometries():
    outer = SuperquadricPackageGeometry(
        np.asarray(
            [0.032, 0.028, 0.014]
        ),
        exponent_xy=2.2,
        exponent_z=2.0,
    )
    inner = SuperquadricPackageGeometry(
        np.asarray(
            [0.016, 0.014, 0.006]
        ),
        exponent_xy=2.2,
        exponent_z=2.0,
    )
    return (
        outer,
        inner,
    )


def test_nested_thermal_media_are_compared_against_parent_region():
    background = _background()
    outer_geometry, inner_geometry = (
        _nested_thermal_geometries()
    )
    outer_material = IsotropicMaterial(
        relative_permittivity=1.0,
        thermal_conductivity=0.22,
        density=1200.0,
        heat_capacity=1800.0,
    )
    inner_material = IsotropicMaterial(
        relative_permittivity=1.0,
        thermal_conductivity=(
            background.conductivity
        ),
        density=background.density,
        heat_capacity=(
            background.heat_capacity
        ),
    )
    scene = Scene(
        (),
        HomogeneousMedium(),
        (
            PackageObject(
                outer_geometry,
                outer_material,
                "outer",
            ),
            PackageObject(
                inner_geometry,
                inner_material,
                "inner",
            ),
        ),
    )
    media = scene_thermal_package_media(
        scene,
        background,
    )
    assert tuple(
        index
        for index, _ in media
    ) == (
        0,
        1,
    )


def test_nested_package_thermal_interface_has_finite_se3_invariant_response():
    background = _background()
    outer_geometry, inner_geometry = (
        _nested_thermal_geometries()
    )
    outer_medium = HomogeneousThermalMedium(
        conductivity=0.24,
        density=1180.0,
        heat_capacity=1750.0,
        ambient_temperature=293.15,
    )
    inner_medium = HomogeneousThermalMedium(
        conductivity=1.15,
        density=930.0,
        heat_capacity=2350.0,
        ambient_temperature=293.15,
    )
    source_positions = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.023, 0.0, 0.0],
        ]
    )
    field = PreparedMultiThermalInterfaceField(
        _source(
            source_positions
        ),
        background,
        (
            (
                0,
                outer_geometry,
                outer_medium,
            ),
            (
                1,
                inner_geometry,
                inner_medium,
            ),
        ),
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        mfs_offset_fraction=0.10,
        stehfest_order=6,
        interface_residual_tolerance=5e-3,
        svd_rcond=1e-10,
    )
    query = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.023, 0.0, 0.0],
            [0.0, 0.0, 0.030],
        ]
    )
    currents = np.asarray(
        [1.0 + 0.0j]
    )
    steady = field.steady_temperature(
        query,
        currents,
    )
    transient = field.temperature_step(
        query,
        2.0,
        currents,
    )
    assert np.all(
        np.isfinite(
            steady
        )
    )
    assert np.all(
        np.isfinite(
            transient
        )
    )
    assert np.all(
        steady
        >= background.ambient_temperature
        - 1e-7
    )
    assert np.all(
        transient
        >= background.ambient_temperature
        - 1e-7
    )
    assert (
        field.maximum_interface_residual
        <= field.interface_residual_tolerance
    )

    rng = np.random.default_rng(
        1907
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [0.13, -0.09, 0.21]
        ),
    )
    moved_field = PreparedMultiThermalInterfaceField(
        _source(
            common.apply(
                source_positions
            )
        ),
        background,
        (
            (
                0,
                outer_geometry.transformed(
                    common
                ),
                outer_medium,
            ),
            (
                1,
                inner_geometry.transformed(
                    common
                ),
                inner_medium,
            ),
        ),
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        mfs_offset_fraction=0.10,
        stehfest_order=6,
        interface_residual_tolerance=5e-3,
        svd_rcond=1e-10,
    )
    moved = moved_field.temperature_step(
        common.apply(
            query
        ),
        2.0,
        currents,
    )
    assert np.allclose(
        moved,
        transient,
        rtol=5e-5,
        atol=5e-6,
    )


def test_system_reference_continuous_thermal_field_supports_nested_package_interfaces():
    coil = CoilObject(
        SuperellipseSpiral(
            0.012,
            0.010,
            0.60,
            8.0e-4,
            8.0e-4,
            conductor_width=7.0e-4,
            conductor_thickness=5.0e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )
    outer_geometry, inner_geometry = (
        _nested_thermal_geometries()
    )
    outer = PackageObject(
        outer_geometry,
        IsotropicMaterial(
            relative_permittivity=1.0,
            relative_permeability=1.0,
            conductivity=0.0,
            thermal_conductivity=0.25,
            density=1180.0,
            heat_capacity=1750.0,
        ),
        "outer",
    )
    inner = PackageObject(
        inner_geometry,
        IsotropicMaterial(
            relative_permittivity=1.0,
            relative_permeability=1.0,
            conductivity=0.0,
            thermal_conductivity=1.05,
            density=940.0,
            heat_capacity=2300.0,
        ),
        "inner",
    )
    scene = Scene(
        (
            coil,
        ),
        HomogeneousMedium(),
        (
            outer,
            inner,
        ),
    )
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=18,
        ),
        reference_config=MQSConfig(
            segments_per_turn=6,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
        dielectric_surface_vertical_order=4,
        dielectric_surface_azimuthal_order=8,
    )
    field = system.reference_continuous_thermal_field(
        scene,
        35_000.0,
        _background(),
        longitudinal_segments=5,
        radial_order=2,
        angular_order=8,
        package_axial_order=2,
        package_radial_order=2,
        package_azimuthal_order=8,
        interface_vertical_order=4,
        interface_azimuthal_order=8,
        mfs_offset_fraction=0.10,
        stehfest_order=6,
        interface_residual_tolerance=8e-3,
        svd_rcond=1e-10,
    )
    assert isinstance(
        field,
        PreparedMultiThermalInterfaceField,
    )
    temperature = field.temperature_step(
        np.asarray(
            [0.0, 0.0, 0.022]
        ),
        1.5,
        np.asarray(
            [1.0 + 0.0j]
        ),
    )
    assert np.isfinite(
        temperature
    )
    assert (
        temperature
        >= field.medium.ambient_temperature
        - 1e-7
    )
    assert (
        field.maximum_interface_residual
        <= field.interface_residual_tolerance
    )


def _anisotropic_package_scene():
    coil = CoilObject(
        SuperellipseSpiral(
            0.020,
            0.018,
            0.65,
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
        _geometry(
            (
                0.040,
                0.0,
                0.0,
            )
        ),
        IsotropicMaterial(
            relative_permittivity=1.0,
            conductivity=0.0,
            thermal_conductivity=0.24,
            density=1180.0,
            heat_capacity=1700.0,
        ),
        "thermal-inclusion",
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
    system = MeshfreeVNextSystem(
        AnalyticBaselineArtifact(
            segments_per_coil=20,
        ),
        reference_config=MQSConfig(
            segments_per_turn=6,
            min_segments=8,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
        dielectric_surface_vertical_order=6,
        dielectric_surface_azimuthal_order=12,
    )
    return (
        scene,
        system,
    )


def _prepare_anisotropic_package_field(
    scene,
    system,
    thermal_medium,
):
    return system.reference_continuous_thermal_field(
        scene,
        40_000.0,
        thermal_medium,
        longitudinal_segments=6,
        radial_order=3,
        angular_order=8,
        package_axial_order=2,
        package_radial_order=2,
        package_azimuthal_order=8,
        interface_vertical_order=4,
        interface_azimuthal_order=8,
        mfs_offset_fraction=0.12,
        stehfest_order=6,
        interface_residual_tolerance=5e-3,
        svd_rcond=1e-13,
    )


def test_package_thermal_interface_anisotropic_isotropic_limit_matches_scalar_background():
    scene, system = _anisotropic_package_scene()
    scalar = HomogeneousThermalMedium(
        conductivity=0.6,
        density=1000.0,
        heat_capacity=4000.0,
        ambient_temperature=293.15,
    )
    tensor = AnisotropicThermalMedium(
        conductivity_tensor=(
            0.6
            * np.eye(
                3
            )
        ),
        density=1000.0,
        heat_capacity=4000.0,
        ambient_temperature=293.15,
    )
    scalar_field = _prepare_anisotropic_package_field(
        scene,
        system,
        scalar,
    )
    tensor_field = _prepare_anisotropic_package_field(
        scene,
        system,
        tensor,
    )
    query = np.asarray(
        [
            [
                0.040,
                0.0,
                0.014,
            ],
            [
                0.0,
                0.0,
                0.030,
            ],
        ]
    )
    currents = np.asarray(
        [
            1.0 + 0.0j
        ]
    )
    scalar_temperature = scalar_field.temperature_step(
        query,
        2.0,
        currents,
    )
    tensor_temperature = tensor_field.temperature_step(
        query,
        2.0,
        currents,
    )
    assert np.allclose(
        tensor_temperature,
        scalar_temperature,
        rtol=3e-5,
        atol=3e-6,
    )


def test_anisotropic_background_package_interface_is_common_rotation_invariant():
    scene, system = _anisotropic_package_scene()
    conductivity = np.diag(
        [
            0.35,
            0.75,
            1.15,
        ]
    )
    medium = AnisotropicThermalMedium(
        conductivity_tensor=conductivity,
        density=1050.0,
        heat_capacity=3600.0,
        ambient_temperature=293.15,
    )
    field = _prepare_anisotropic_package_field(
        scene,
        system,
        medium,
    )
    query = np.asarray(
        [
            0.040,
            0.002,
            0.014,
        ]
    )
    currents = np.asarray(
        [
            1.0 + 0.0j
        ]
    )
    reference = field.temperature_step(
        query,
        2.5,
        currents,
    )

    rng = np.random.default_rng(
        17031
    )
    rotation = haar_rotation(
        rng
    )
    common = RigidPose(
        rotation,
        np.asarray(
            [
                0.11,
                -0.08,
                0.19,
            ]
        ),
    )
    moved_coil = CoilObject(
        scene.coils[
            0
        ].geometry.transformed(
            common
        ),
        scene.coils[
            0
        ].material,
        "coil",
    )
    moved_package = PackageObject(
        scene.packages[
            0
        ].geometry.transformed(
            common
        ),
        scene.packages[
            0
        ].material,
        "thermal-inclusion",
    )
    moved_scene = Scene(
        (
            moved_coil,
        ),
        scene.medium,
        (
            moved_package,
        ),
    )
    moved_medium = AnisotropicThermalMedium(
        conductivity_tensor=(
            rotation
            @ conductivity
            @ rotation.T
        ),
        density=medium.density,
        heat_capacity=medium.heat_capacity,
        ambient_temperature=(
            medium.ambient_temperature
        ),
    )
    moved_field = _prepare_anisotropic_package_field(
        moved_scene,
        system,
        moved_medium,
    )
    actual = moved_field.temperature_step(
        common.apply(
            query
        ),
        2.5,
        currents,
    )
    assert np.isclose(
        actual,
        reference,
        rtol=5e-5,
        atol=5e-6,
    )
    assert np.isfinite(
        actual
    )
    assert (
        actual
        > moved_medium.ambient_temperature
    )


def test_tensor_package_thermal_conductivity_isotropic_limit_matches_scalar_package():
    base_scene, system = _anisotropic_package_scene()
    geometry = base_scene.packages[
        0
    ].geometry
    scalar_package = PackageObject(
        geometry,
        IsotropicMaterial(
            relative_permittivity=1.0,
            conductivity=0.0,
            thermal_conductivity=0.24,
            density=1180.0,
            heat_capacity=1700.0,
        ),
        "scalar-package",
    )
    tensor_package = PackageObject(
        geometry,
        IsotropicMaterial(
            relative_permittivity=1.0,
            conductivity=0.0,
            thermal_conductivity_tensor=(
                0.24
                * np.eye(
                    3
                )
            ),
            density=1180.0,
            heat_capacity=1700.0,
        ),
        "tensor-package",
    )
    scalar_scene = Scene(
        base_scene.coils,
        base_scene.medium,
        (
            scalar_package,
        ),
    )
    tensor_scene = Scene(
        base_scene.coils,
        base_scene.medium,
        (
            tensor_package,
        ),
    )
    background = _background()
    scalar_field = _prepare_anisotropic_package_field(
        scalar_scene,
        system,
        background,
    )
    tensor_field = _prepare_anisotropic_package_field(
        tensor_scene,
        system,
        background,
    )
    query = np.asarray(
        [
            0.040,
            0.001,
            0.014,
        ]
    )
    currents = np.asarray(
        [
            1.0 + 0.0j
        ]
    )
    scalar_temperature = scalar_field.temperature_step(
        query,
        2.0,
        currents,
    )
    tensor_temperature = tensor_field.temperature_step(
        query,
        2.0,
        currents,
    )
    assert np.isclose(
        tensor_temperature,
        scalar_temperature,
        rtol=3e-5,
        atol=3e-6,
    )


def test_tensor_package_thermal_conductivity_is_common_rotation_invariant():
    base_scene, system = _anisotropic_package_scene()
    package_tensor = np.asarray(
        [
            [0.21, 0.06, 0.01],
            [0.06, 0.47, 0.04],
            [0.01, 0.04, 0.82],
        ],
        dtype=float,
    )
    package = PackageObject(
        base_scene.packages[
            0
        ].geometry,
        IsotropicMaterial(
            relative_permittivity=1.0,
            conductivity=0.0,
            thermal_conductivity_tensor=(
                package_tensor
            ),
            density=1180.0,
            heat_capacity=1700.0,
        ),
        "tensor-package",
    )
    scene = Scene(
        base_scene.coils,
        base_scene.medium,
        (
            package,
        ),
    )
    field = _prepare_anisotropic_package_field(
        scene,
        system,
        _background(),
    )
    query = np.asarray(
        [
            0.040,
            0.002,
            0.013,
        ]
    )
    currents = np.asarray(
        [
            1.0 + 0.0j
        ]
    )
    reference = field.temperature_step(
        query,
        2.5,
        currents,
    )

    rng = np.random.default_rng(
        91231
    )
    rotation = haar_rotation(
        rng
    )
    common = RigidPose(
        rotation,
        np.asarray(
            [
                -0.09,
                0.14,
                0.21,
            ]
        ),
    )
    moved_coil = CoilObject(
        scene.coils[
            0
        ].geometry.transformed(
            common
        ),
        scene.coils[
            0
        ].material,
        "coil",
    )
    moved_package = PackageObject(
        package.geometry.transformed(
            common
        ),
        IsotropicMaterial(
            relative_permittivity=1.0,
            conductivity=0.0,
            thermal_conductivity_tensor=(
                rotation
                @ package_tensor
                @ rotation.T
            ),
            density=1180.0,
            heat_capacity=1700.0,
        ),
        "tensor-package",
    )
    moved_scene = Scene(
        (
            moved_coil,
        ),
        scene.medium,
        (
            moved_package,
        ),
    )
    moved_field = _prepare_anisotropic_package_field(
        moved_scene,
        system,
        _background(),
    )
    actual = moved_field.temperature_step(
        common.apply(
            query
        ),
        2.5,
        currents,
    )
    assert np.isclose(
        actual,
        reference,
        rtol=6e-5,
        atol=6e-6,
    )
    assert np.isfinite(
        actual
    )
