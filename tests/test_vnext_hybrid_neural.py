import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DebyeMaterial,
    HomogeneousMedium,
    HybridSceneSamplerConfig,
    HybridTeacherSample,
    IsotropicMaterial,
    MeshfreeVNextSystem,
    PackageObject,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    SuperquadricPackageGeometry,
    analytic_port_baseline,
    encode_hybrid_scene_invariant,
    haar_rotation,
    sample_hybrid_package_scene,
)
from sdfmpneo_vnext.hybrid_neural import (
    HybridNeuralResidualArtifact,
    HybridNormalizer,
    HybridPhysicsFactoredResidualNet,
    train_hybrid_residual_surrogate,
)


def _scene(
    *,
    loss=0.003,
    swap_packages=False,
):
    copper = ConductorMaterial(
        5.8e7
    )
    coil_a = CoilObject(
        SuperellipseSpiral(
            0.026,
            0.022,
            0.8,
            0.0012,
            0.0012,
            exponent=3.0,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
        "a",
    )
    coil_b = CoilObject(
        SuperellipseSpiral(
            0.021,
            0.018,
            0.7,
            0.001,
            0.001,
            exponent=3.5,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                0.3,
                translation=(
                    0.006,
                    -0.002,
                    0.02,
                ),
            ),
        ),
        copper,
        "b",
    )
    first = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.034, 0.028, 0.009]
            ),
            exponent_xy=3.0,
            exponent_z=4.0,
            pose=RigidPose.from_axis_angle(
                (1.0, 0.0, 0.0),
                0.15,
                translation=(
                    0.002,
                    0.0,
                    0.006,
                ),
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=3.2,
            conductivity=loss,
        ),
        "p0",
    )
    second = PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [0.016, 0.013, 0.011]
            ),
            exponent_xy=2.5,
            exponent_z=3.0,
            pose=RigidPose.from_axis_angle(
                (0.0, 0.0, 1.0),
                -0.2,
                translation=(
                    0.045,
                    0.008,
                    0.012,
                ),
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=2.1,
            conductivity=0.0,
        ),
        "p1",
    )
    packages = (
        (second, first)
        if swap_packages
        else (first, second)
    )
    return Scene(
        (
            coil_a,
            coil_b,
        ),
        HomogeneousMedium(),
        packages,
    )


def _manual_sample(
    scene,
    frequency=85_000.0,
):
    encoded = (
        encode_hybrid_scene_invariant(
            scene,
            frequency,
        )
    )
    baseline = (
        analytic_port_baseline(
            Scene(
                scene.coils,
                scene.medium,
            ),
            frequency,
            segments_per_coil=32,
        )
    )
    target = (
        baseline.resistance
        + 1j
        * (
            2.0
            * np.pi
            * frequency
            * baseline.inductance
        )
    )
    n = len(
        scene.coils
    )
    channels = np.zeros(
        (
            n + 1,
            n,
            n,
        ),
        dtype=complex,
    )
    for index in range(
        n
    ):
        channels[
            index,
            index,
            index,
        ] = (
            target.real[
                index,
                index,
            ]
        )
    return HybridTeacherSample(
        scene=scene,
        frequency_hz=frequency,
        encoded=encoded,
        baseline_resistance=(
            baseline.resistance
        ),
        baseline_reactance=(
            target.imag
        ),
        target_impedance=target,
        target_dissipation_channels=(
            channels
        ),
        baseline_segments=32,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
        surface_residual=0.0,
        raw_potential_reciprocity_defect=0.0,
        power_closure_error=0.0,
    )


def _artifact(
    scene,
    *,
    background_conductivity_range=None,
    background_permittivity_range=None,
    package_permittivity_range=None,
    package_loss_conductivity_range=None,
    package_permeability_range=None,
    geometry_domain=None,
    artifact_schema=None,
):
    sample = _manual_sample(
        scene
    )
    normalizer = (
        HybridNormalizer.fit(
            (sample,)
        )
    )
    torch.manual_seed(
        29
    )
    model = (
        HybridPhysicsFactoredResidualNet(
            hidden_dim=24,
            factor_rank=3,
            depth=1,
        )
    )
    return HybridNeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=32,
        background_conductivity_range=(
            background_conductivity_range
        ),
        background_permittivity_range=(
            background_permittivity_range
        ),
        package_permittivity_range=(
            package_permittivity_range
        ),
        package_loss_conductivity_range=(
            package_loss_conductivity_range
        ),
        package_permeability_range=(
            package_permeability_range
        ),
        geometry_domain=(
            geometry_domain
        ),
        artifact_schema=(
            artifact_schema
        ),
    )


def _assert_structured_physics(
    prediction,
):
    impedance = (
        prediction.impedance
    )
    channels = (
        prediction.dissipation_channels
    )
    assert np.allclose(
        impedance,
        impedance.T,
        rtol=2e-6,
        atol=2e-7,
    )
    dissipation = 0.5 * (
        impedance
        + impedance.conj().T
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                dissipation
            )
        )
        >= -2e-7
    )
    assert np.allclose(
        np.sum(
            channels,
            axis=0,
        ),
        dissipation,
        rtol=3e-5,
        atol=3e-7,
    )
    for channel in channels:
        assert np.allclose(
            channel,
            channel.conj().T,
            atol=3e-6,
        )
        assert (
            np.min(
                np.linalg.eigvalsh(
                    channel
                )
            )
            >= -3e-6
        )


def test_random_hybrid_network_is_structurally_passive_reciprocal_and_closed():
    scene = _scene()
    prediction = (
        _artifact(
            scene
        ).predict_structured(
            scene,
            85_000.0,
        )
    )
    assert prediction.n_channels == (
        len(
            scene.coils
        )
        + 1
    )
    _assert_structured_physics(
        prediction
    )


def test_hybrid_network_is_common_se3_invariant():
    scene = _scene()
    artifact = _artifact(
        scene
    )
    reference = (
        artifact.predict_structured(
            scene,
            85_000.0,
        )
    )

    rng = np.random.default_rng(
        503
    )
    pose = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.21, -0.13, 0.31]
        ),
    )
    moved = Scene(
        tuple(
            CoilObject(
                coil.geometry.transformed(
                    pose
                ),
                coil.material,
                coil.name,
            )
            for coil in scene.coils
        ),
        scene.medium,
        tuple(
            PackageObject(
                package.geometry.transformed(
                    pose
                ),
                package.material,
                package.name,
            )
            for package
            in scene.packages
        ),
    )
    actual = (
        artifact.predict_structured(
            moved,
            85_000.0,
        )
    )
    assert np.allclose(
        actual.impedance,
        reference.impedance,
        rtol=3e-5,
        atol=3e-7,
    )
    assert np.allclose(
        actual.dissipation_channels,
        reference.dissipation_channels,
        rtol=5e-5,
        atol=5e-7,
    )


def test_hybrid_network_is_package_permutation_invariant():
    scene = _scene()
    artifact = _artifact(
        scene
    )
    reference = (
        artifact.predict_structured(
            scene,
            85_000.0,
        )
    )
    swapped = _scene(
        swap_packages=True
    )
    actual = (
        artifact.predict_structured(
            swapped,
            85_000.0,
        )
    )
    assert np.allclose(
        actual.impedance,
        reference.impedance,
        rtol=3e-5,
        atol=3e-7,
    )
    assert np.allclose(
        actual.dissipation_channels,
        reference.dissipation_channels,
        rtol=5e-5,
        atol=5e-7,
    )


def test_lossless_hybrid_network_has_exactly_zero_dielectric_channel():
    scene = _scene(
        loss=0.0
    )
    prediction = (
        _artifact(
            scene
        ).predict_structured(
            scene,
            85_000.0,
        )
    )
    _assert_structured_physics(
        prediction
    )
    assert np.allclose(
        prediction.dissipation_channels[
            -1
        ],
        0.0,
        rtol=0,
        atol=0,
    )


def test_hybrid_artifact_save_load_round_trip(tmp_path):
    scene = _scene()
    artifact = _artifact(
        scene
    )
    expected = (
        artifact.predict_structured(
            scene,
            85_000.0,
        )
    )
    path = (
        tmp_path
        / "hybrid.pt"
    )
    artifact.save(
        path
    )
    loaded = (
        HybridNeuralResidualArtifact.load(
            path
        )
    )
    actual = (
        loaded.predict_structured(
            scene,
            85_000.0,
        )
    )
    assert np.allclose(
        actual.impedance,
        expected.impedance,
        rtol=0,
        atol=0,
    )
    assert np.allclose(
        actual.dissipation_channels,
        expected.dissipation_channels,
        rtol=0,
        atol=0,
    )



def test_hybrid_artifact_is_accepted_by_unified_fast_port_runtime():
    scene = _scene()
    artifact = _artifact(
        scene
    )
    assert artifact.supports_packages
    system = MeshfreeVNextSystem(
        artifact
    )
    direct = artifact.predict_structured(
        scene,
        85_000.0,
    )
    through_system = system.fast_ports(
        scene,
        85_000.0,
    )
    assert np.allclose(
        through_system.impedance,
        direct.impedance,
        rtol=0,
        atol=0,
    )
    assert np.allclose(
        through_system.dissipation_channels,
        direct.dissipation_channels,
        rtol=0,
        atol=0,
    )


def _with_background(
    scene,
    *,
    conductivity,
    relative_permittivity=2.5,
):
    return Scene(
        scene.coils,
        HomogeneousMedium(
            relative_permittivity=(
                relative_permittivity
            ),
            relative_permeability=(
                scene.medium.relative_permeability
            ),
            conductivity=(
                conductivity
            ),
        ),
        scene.packages,
    )


def test_hybrid_fast_port_artifact_accepts_certified_lossy_background_domain():
    base = _scene(
        loss=0.0
    )
    scene = _with_background(
        base,
        conductivity=1.0e-3,
    )
    artifact = _artifact(
        base,
        background_conductivity_range=(
            0.0,
            2.0e-3,
        ),
    )
    assert artifact.supports_lossy_background
    prediction = artifact.predict_structured(
        scene,
        85_000.0,
    )
    _assert_structured_physics(
        prediction
    )
    assert (
        prediction.dissipation_channels[
            -1
        ].real.max()
        > 0.0
    )

    system = MeshfreeVNextSystem(
        artifact
    )
    through_system = system.fast_ports(
        scene,
        85_000.0,
    )
    assert np.allclose(
        through_system.impedance,
        prediction.impedance,
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        through_system.dissipation_channels,
        prediction.dissipation_channels,
        rtol=0.0,
        atol=0.0,
    )


def test_hybrid_fast_port_artifact_rejects_background_conductivity_outside_training_domain():
    base = _scene()
    artifact = _artifact(
        base,
        background_conductivity_range=(
            0.0,
            1.0e-3,
        ),
    )
    outside = _with_background(
        base,
        conductivity=2.0e-3,
    )
    with pytest.raises(
        ValueError,
        match="outside the hybrid artifact training domain",
    ):
        artifact.predict_structured(
            outside,
            85_000.0,
        )


def test_legacy_hybrid_artifact_remains_lossless_background_only():
    base = _scene()
    artifact = _artifact(
        base
    )
    assert not artifact.supports_lossy_background
    lossy = _with_background(
        base,
        conductivity=1.0e-4,
    )
    with pytest.raises(
        ValueError,
        match="not trained/certified for a lossy",
    ):
        artifact.predict_structured(
            lossy,
            85_000.0,
        )


def test_hybrid_artifact_round_trip_preserves_lossy_background_domain(tmp_path):
    base = _scene()
    artifact = _artifact(
        base,
        background_conductivity_range=(
            0.0,
            3.0e-3,
        ),
        background_permittivity_range=(
            1.0,
            6.0,
        ),
        package_permittivity_range=(
            1.5,
            6.0,
        ),
        package_loss_conductivity_range=(
            0.0,
            1.0e-2,
        ),
        package_permeability_range=(
            1.0,
            4.0,
        ),
    )
    path = (
        tmp_path
        / "hybrid-lossy-domain.pt"
    )
    artifact.save(
        path
    )
    loaded = HybridNeuralResidualArtifact.load(
        path
    )
    assert loaded.supports_lossy_background
    assert loaded.background_conductivity_range == (
        0.0,
        3.0e-3,
    )
    assert loaded.background_permittivity_range == (
        1.0,
        6.0,
    )
    assert loaded.package_permittivity_range == (
        1.5,
        6.0,
    )
    assert loaded.package_loss_conductivity_range == (
        0.0,
        1.0e-2,
    )
    assert loaded.package_permeability_range == (
        1.0,
        4.0,
    )
    lossy = _with_background(
        base,
        conductivity=1.0e-3,
    )
    expected = artifact.predict_structured(
        lossy,
        85_000.0,
    )
    actual = loaded.predict_structured(
        lossy,
        85_000.0,
    )
    assert np.allclose(
        actual.impedance,
        expected.impedance,
        rtol=0.0,
        atol=0.0,
    )
    assert np.allclose(
        actual.dissipation_channels,
        expected.dissipation_channels,
        rtol=0.0,
        atol=0.0,
    )


def test_schema1_hybrid_artifact_loads_as_lossless_background_only(tmp_path):
    base = _scene()
    artifact = _artifact(
        base
    )
    path = (
        tmp_path
        / "hybrid-schema2.pt"
    )
    artifact.save(
        path
    )
    try:
        payload = torch.load(
            path,
            weights_only=False,
        )
    except TypeError:
        payload = torch.load(
            path
        )
    payload[
        "schema"
    ] = 1
    payload.pop(
        "background_conductivity_range",
        None,
    )
    legacy_path = (
        tmp_path
        / "hybrid-schema1.pt"
    )
    torch.save(
        payload,
        legacy_path,
    )
    loaded = HybridNeuralResidualArtifact.load(
        legacy_path
    )
    assert not loaded.supports_lossy_background
    with pytest.raises(
        ValueError,
        match="not trained/certified for a lossy",
    ):
        loaded.predict_structured(
            _with_background(
                base,
                conductivity=1.0e-4,
            ),
            85_000.0,
        )


def test_hybrid_training_uses_declared_background_conductivity_domain():
    base = _scene(
        loss=0.0
    )
    train_scene = _with_background(
        base,
        conductivity=5.0e-4,
    )
    validation_scene = _with_background(
        base,
        conductivity=1.5e-3,
    )
    artifact, report = (
        train_hybrid_residual_surrogate(
            (
                _manual_sample(
                    train_scene
                ),
            ),
            validation_samples=(
                _manual_sample(
                    validation_scene
                ),
            ),
            hidden_dim=8,
            factor_rank=1,
            depth=1,
            epochs=1,
            patience=1,
            background_conductivity_range=(
                0.0,
                2.0e-3,
            ),
        )
    )
    assert report.epochs == 1
    assert artifact.supports_lossy_background
    assert (
        artifact.background_conductivity_range
        == (
            0.0,
            2.0e-3,
        )
    )
    artifact.predict_structured(
        validation_scene,
        85_000.0,
    )


def test_hybrid_training_rejects_validation_outside_declared_background_domain():
    base = _scene(
        loss=0.0
    )
    train_scene = _with_background(
        base,
        conductivity=5.0e-4,
    )
    validation_scene = _with_background(
        base,
        conductivity=3.0e-3,
    )
    with pytest.raises(
        ValueError,
        match="outside the declared hybrid port training domain",
    ):
        train_hybrid_residual_surrogate(
            (
                _manual_sample(
                    train_scene
                ),
            ),
            validation_samples=(
                _manual_sample(
                    validation_scene
                ),
            ),
            hidden_dim=8,
            factor_rank=1,
            depth=1,
            epochs=1,
            patience=1,
            background_conductivity_range=(
                0.0,
                2.0e-3,
            ),
        )


def test_hybrid_fast_port_artifact_accepts_debye_background_in_effective_domain():
    base = _scene(
        loss=0.0
    )
    medium = DebyeMaterial(
        relative_permittivity_static=20.0,
        relative_permittivity_infinite=4.0,
        relaxation_time=2.0e-6,
        conductivity=0.0,
    )
    scene = Scene(
        base.coils,
        medium,
        base.packages,
    )
    frequency = 85_000.0
    effective_loss = medium.loss_conductivity(
        frequency
    )
    effective_epsilon = float(
        np.real(
            medium.relative_permittivity_at(
                frequency
            )
        )
    )
    assert effective_loss > 0.0
    artifact = _artifact(
        base,
        background_conductivity_range=(
            0.0,
            2.0e-4,
        ),
        background_permittivity_range=(
            3.0,
            21.0,
        ),
    )
    prediction = artifact.predict_structured(
        scene,
        frequency,
    )
    _assert_structured_physics(
        prediction
    )
    assert (
        3.0
        <= effective_epsilon
        <= 21.0
    )
    assert (
        prediction.dissipation_channels[
            -1
        ].real.max()
        > 0.0
    )


def test_hybrid_fast_port_artifact_rejects_effective_background_permittivity_outside_domain():
    base = _scene()
    artifact = _artifact(
        base,
        background_conductivity_range=(
            0.0,
            1.0e-3,
        ),
        background_permittivity_range=(
            1.0,
            6.0,
        ),
    )
    outside = _with_background(
        base,
        conductivity=0.0,
        relative_permittivity=12.0,
    )
    with pytest.raises(
        ValueError,
        match="effective relative permittivity",
    ):
        artifact.predict_structured(
            outside,
            85_000.0,
        )


def test_hybrid_fast_port_rejects_package_effective_permittivity_outside_domain():
    base = _scene()
    artifact = _artifact(
        base,
        package_permittivity_range=(
            1.5,
            6.0,
        ),
        package_loss_conductivity_range=(
            0.0,
            1.0e-2,
        ),
    )
    first = base.packages[
        0
    ]
    outside = Scene(
        base.coils,
        base.medium,
        (
            PackageObject(
                first.geometry,
                IsotropicMaterial(
                    relative_permittivity=15.0,
                    conductivity=first.material.conductivity,
                ),
                first.name,
            ),
            base.packages[
                1
            ],
        ),
    )
    with pytest.raises(
        ValueError,
        match="package effective relative permittivity",
    ):
        artifact.predict_structured(
            outside,
            85_000.0,
        )


def _magnetic_package_scene(
    base,
    relative_permeability,
):
    first = base.packages[
        0
    ]
    return Scene(
        base.coils,
        base.medium,
        (
            PackageObject(
                first.geometry,
                IsotropicMaterial(
                    relative_permittivity=3.2,
                    relative_permeability=(
                        relative_permeability
                    ),
                    conductivity=0.003,
                ),
                first.name,
            ),
            base.packages[
                1
            ],
        ),
    )


def test_hybrid_fast_port_fails_closed_without_declared_magnetic_package_domain():
    base = _scene()
    artifact = _artifact(
        base
    )
    magnetic = _magnetic_package_scene(
        base,
        1.2,
    )
    with pytest.raises(
        ValueError,
        match="not trained/certified for magnetic package contrast",
    ):
        artifact.predict_structured(
            magnetic,
            85_000.0,
        )


def test_hybrid_fast_port_accepts_magnetic_package_inside_declared_domain():
    base = _scene()
    artifact = _artifact(
        base,
        package_permeability_range=(
            1.0,
            1.5,
        ),
    )
    magnetic = _magnetic_package_scene(
        base,
        1.2,
    )
    prediction = artifact.predict_structured(
        magnetic,
        85_000.0,
    )
    _assert_structured_physics(
        prediction
    )

    outside = _magnetic_package_scene(
        base,
        1.8,
    )
    with pytest.raises(
        ValueError,
        match="relative permeability",
    ):
        artifact.predict_structured(
            outside,
            85_000.0,
        )


def test_hybrid_fast_geometry_domain_is_se3_invariant_and_fails_closed_outside_package_pose():
    config = HybridSceneSamplerConfig(
        package_center_offset_fraction_range=(
            0.05,
            0.30,
        ),
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            1009
        ),
        config,
    )
    artifact = _artifact(
        scene,
        geometry_domain=(
            config.geometry_domain_metadata()
        ),
    )
    reference = artifact.predict_structured(
        scene,
        frequency,
    )
    _assert_structured_physics(
        reference
    )

    common = RigidPose(
        haar_rotation(
            np.random.default_rng(
                1013
            )
        ),
        np.asarray(
            [0.17, -0.09, 0.23]
        ),
    )
    moved = Scene(
        tuple(
            CoilObject(
                coil.geometry.transformed(
                    common
                ),
                coil.material,
                coil.name,
            )
            for coil in scene.coils
        ),
        scene.medium,
        tuple(
            PackageObject(
                package.geometry.transformed(
                    common
                ),
                package.material,
                package.name,
            )
            for package in scene.packages
        ),
    )
    moved_prediction = (
        artifact.predict_structured(
            moved,
            frequency,
        )
    )
    assert np.allclose(
        moved_prediction.impedance,
        reference.impedance,
        rtol=4e-5,
        atol=4e-7,
    )

    package = scene.packages[
        0
    ]
    displaced = PackageObject(
        package.geometry.transformed(
            RigidPose(
                np.eye(
                    3
                ),
                np.asarray(
                    [0.20, 0.0, 0.0]
                ),
            )
        ),
        package.material,
        package.name,
    )
    outside = Scene(
        scene.coils,
        scene.medium,
        (
            displaced,
        ),
    )
    with pytest.raises(
        ValueError,
        match="package center offset",
    ):
        artifact.predict_structured(
            outside,
            frequency,
        )

    oversized_geometry = SuperquadricPackageGeometry(
        package.geometry.half_extents
        * 4.0,
        exponent_xy=(
            package.geometry.exponent_xy
        ),
        exponent_z=(
            package.geometry.exponent_z
        ),
        pose=(
            package.geometry.pose
        ),
    )
    oversized = Scene(
        scene.coils,
        scene.medium,
        (
            PackageObject(
                oversized_geometry,
                package.material,
                package.name,
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="package enclosure scale",
    ):
        artifact.predict_structured(
            oversized,
            frequency,
        )


def test_legacy_hybrid_port_fingerprint_ignores_newer_geometry_domain_and_resaves_stably(tmp_path):
    scene = _scene()
    legacy_a = _artifact(
        scene,
        geometry_domain={
            "tag": "first",
        },
        artifact_schema=4,
    )
    legacy_b = _artifact(
        scene,
        geometry_domain={
            "tag": "second",
        },
        artifact_schema=4,
    )
    assert (
        legacy_a.fingerprint()
        == legacy_b.fingerprint()
    )

    current_a = _artifact(
        scene,
        geometry_domain={
            "tag": "first",
        },
        artifact_schema=5,
    )
    current_b = _artifact(
        scene,
        geometry_domain={
            "tag": "second",
        },
        artifact_schema=5,
    )
    assert (
        current_a.fingerprint()
        != current_b.fingerprint()
    )

    schema5_mu_a = _artifact(
        scene,
        package_permeability_range=(
            1.0,
            1.0,
        ),
        artifact_schema=5,
    )
    schema5_mu_b = _artifact(
        scene,
        package_permeability_range=(
            1.0,
            4.0,
        ),
        artifact_schema=5,
    )
    assert (
        schema5_mu_a.fingerprint()
        == schema5_mu_b.fingerprint()
    )

    schema6_mu_a = _artifact(
        scene,
        package_permeability_range=(
            1.0,
            1.0,
        ),
        artifact_schema=6,
    )
    schema6_mu_b = _artifact(
        scene,
        package_permeability_range=(
            1.0,
            4.0,
        ),
        artifact_schema=6,
    )
    assert (
        schema6_mu_a.fingerprint()
        != schema6_mu_b.fingerprint()
    )

    path = (
        tmp_path
        / "legacy-hybrid-port.pt"
    )
    legacy_a.save(
        path
    )
    loaded = HybridNeuralResidualArtifact.load(
        path
    )
    assert loaded.artifact_schema == 4
    assert (
        loaded.fingerprint()
        == legacy_a.fingerprint()
    )


def test_hybrid_fast_rejects_package_surface_intersecting_finite_conductor():
    base = _scene()
    first_coil = base.coils[
        0
    ]
    crossing = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [
                    first_coil.geometry.outer_a,
                    first_coil.geometry.outer_b,
                    0.5
                    * first_coil.geometry.conductor_thickness,
                ]
            ),
            exponent_xy=2.0,
            exponent_z=2.0,
        ),
        base.packages[
            0
        ].material,
        "crossing",
    )
    scene = Scene(
        base.coils,
        base.medium,
        (
            crossing,
            base.packages[
                1
            ],
        ),
    )
    artifact = _artifact(
        base
    )
    with pytest.raises(
        ValueError,
        match="package surface intersects",
    ):
        artifact.predict_structured(
            scene,
            85_000.0,
        )


def test_hybrid_fast_geometry_domain_accepts_declared_nested_package_count():
    config = HybridSceneSamplerConfig(
        package_count_range=(
            3,
            3,
        ),
        nested_package_probability=1.0,
        nested_package_scale_range=(
            1.15,
            1.25,
        ),
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            3301
        ),
        config,
    )
    artifact = _artifact(
        scene,
        geometry_domain=(
            config.geometry_domain_metadata()
        ),
        package_permittivity_range=(
            1.0,
            8.0,
        ),
        package_loss_conductivity_range=(
            0.0,
            1.0e-2,
        ),
        package_permeability_range=(
            1.0,
            1.0,
        ),
    )
    prediction = artifact.predict_structured(
        scene,
        frequency,
    )
    _assert_structured_physics(
        prediction
    )

    outside = Scene(
        scene.coils,
        scene.medium,
        (
            scene.packages[
                0
            ],
        ),
    )
    with pytest.raises(
        ValueError,
        match="package count",
    ):
        artifact.predict_structured(
            outside,
            frequency,
        )


def test_hybrid_fast_exact_dc_hard_gates_reactance_and_keeps_conduction_loss_channel():
    config = HybridSceneSamplerConfig(
        dc_probability=1.0,
        dc_conductive_probability=1.0,
        package_count_range=(
            1,
            1,
        ),
        background_conductivity_range=(
            8.0e-4,
            2.0e-3,
        ),
        dielectric_conductivity_range=(
            1.0e-3,
            4.0e-3,
        ),
        lossless_probability=1.0,
        lossy_background_probability=0.0,
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            4217
        ),
        config,
    )
    assert (
        frequency
        == 0.0
    )
    background_domain = (
        config.background_domain_metadata()
    )
    package_domain = (
        config.package_domain_metadata()
    )
    artifact = _artifact(
        scene,
        background_conductivity_range=tuple(
            background_domain[
                "effective_loss_conductivity_range"
            ]
        ),
        background_permittivity_range=tuple(
            background_domain[
                "effective_relative_permittivity_range"
            ]
        ),
        package_permittivity_range=tuple(
            package_domain[
                "effective_relative_permittivity_range"
            ]
        ),
        package_loss_conductivity_range=tuple(
            package_domain[
                "effective_loss_conductivity_range"
            ]
        ),
        package_permeability_range=tuple(
            package_domain[
                "relative_permeability_range"
            ]
        ),
        geometry_domain=(
            config.geometry_domain_metadata()
        ),
    )
    prediction = artifact.predict_structured(
        scene,
        frequency,
    )
    _assert_structured_physics(
        prediction
    )
    assert np.allclose(
        prediction.impedance.imag,
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    environment = prediction.dissipation_channels[
        -1
    ]
    assert (
        np.linalg.norm(
            environment
        )
        > 0.0
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                0.5
                * (
                    environment
                    + environment.conj().T
                )
            )
        )
        >= -1e-10
    )


def test_hybrid_fast_geometry_domain_accepts_declared_free_inclusion_and_legacy_domain_rejects_it():
    config = HybridSceneSamplerConfig(
        package_count_range=(
            1,
            1,
        ),
        free_inclusion_probability=1.0,
        free_inclusion_center_radius_fraction_range=(
            0.75,
            1.25,
        ),
        free_inclusion_half_extent_fraction_range=(
            0.12,
            0.25,
        ),
    )
    scene, frequency = sample_hybrid_package_scene(
        np.random.default_rng(
            3529
        ),
        config,
    )
    assert all(
        scene.packages[
            0
        ].geometry.classify_conductor(
            coil.geometry,
            longitudinal_segments=64,
            section_points=16,
            tolerance=1e-10,
        )
        == "outside"
        for coil in scene.coils
    )

    package_domain = (
        config.package_domain_metadata()
    )
    artifact = _artifact(
        scene,
        geometry_domain=(
            config.geometry_domain_metadata()
        ),
        package_permittivity_range=tuple(
            package_domain[
                "effective_relative_permittivity_range"
            ]
        ),
        package_loss_conductivity_range=tuple(
            package_domain[
                "effective_loss_conductivity_range"
            ]
        ),
        package_permeability_range=tuple(
            package_domain[
                "relative_permeability_range"
            ]
        ),
    )
    prediction = artifact.predict_structured(
        scene,
        frequency,
    )
    _assert_structured_physics(
        prediction
    )

    legacy_domain = (
        HybridSceneSamplerConfig().geometry_domain_metadata()
    )
    legacy = _artifact(
        scene,
        geometry_domain=legacy_domain,
    )
    with pytest.raises(
        ValueError,
        match="free material inclusions",
    ):
        legacy.predict_structured(
            scene,
            frequency,
        )
