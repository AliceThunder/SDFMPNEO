import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
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
)
from sdfmpneo_vnext.hybrid_neural import (
    HybridNeuralResidualArtifact,
    HybridNormalizer,
    HybridPhysicsFactoredResidualNet,
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
