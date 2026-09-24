import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    TeacherSample,
    analytic_port_baseline,
    encode_scene_invariant,
    haar_rotation,
)
from sdfmpneo_vnext.spatial_neural import (
    NeuralSpatialLossArtifact,
    SpatialFeatureNormalizer,
    SpatialLossShapeNet,
)


def _scene():
    copper = ConductorMaterial(5.8e7)
    first = CoilObject(
        SuperellipseSpiral(
            0.030,
            0.026,
            0.9,
            0.0015,
            0.0015,
            exponent=3.5,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
        ),
        copper,
        "a",
    )
    second = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.020,
            0.75,
            0.0012,
            0.0012,
            exponent=4.0,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                0.35,
                translation=(0.006, 0.0, 0.020),
            ),
        ),
        copper,
        "b",
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
    )


def _normalizer(scene, frequency):
    encoded = encode_scene_invariant(
        scene,
        frequency,
    )
    baseline = analytic_port_baseline(
        scene,
        frequency,
        segments_per_coil=32,
    )
    target = (
        baseline.resistance
        + 1j
        * 2.0
        * np.pi
        * frequency
        * baseline.inductance
    )
    sample = TeacherSample(
        scene,
        frequency,
        encoded,
        baseline.resistance,
        target.imag,
        target,
        32,
        None,
        None,
        "mqs",
    )
    return SpatialFeatureNormalizer.fit(
        (sample,)
    )


def _artifact(scene, frequency):
    torch.manual_seed(13)
    model = SpatialLossShapeNet(
        hidden_dim=20,
        factor_rank=2,
        depth=1,
    ).double()
    return NeuralSpatialLossArtifact(
        AnalyticBaselineArtifact(
            segments_per_coil=32,
        ),
        model,
        _normalizer(
            scene,
            frequency,
        ),
        integration_segments_per_turn=6,
        integration_min_segments=8,
        integration_radial_order=2,
        integration_angular_order=8,
    )


def test_random_spatial_network_is_psd_and_closes_each_coil_channel():
    scene = _scene()
    frequency = 70_000.0
    artifact = _artifact(
        scene,
        frequency,
    )
    prediction = artifact.predict_structured(
        scene,
        frequency,
    )

    for coil in range(2):
        integrated = artifact.integrated_matrix(
            scene,
            frequency,
            coil,
        )
        assert np.allclose(
            integrated,
            prediction.dissipation_channels[
                coil
            ],
            rtol=3e-8,
            atol=3e-10,
        )
        for arc, xy in (
            (0.1, (0.0, 0.0)),
            (0.5, (2e-4, -1e-4)),
            (0.9, (-2e-4, 1e-4)),
        ):
            matrix = (
                artifact.local_dissipation_matrix(
                    scene,
                    frequency,
                    coil,
                    arc,
                    xy,
                )
            )
            assert np.allclose(
                matrix,
                matrix.conj().T,
                atol=1e-10,
            )
            assert (
                np.min(
                    np.linalg.eigvalsh(
                        matrix
                    )
                )
                >= -1e-10
            )


def test_spatial_field_is_common_se3_invariant():
    scene = _scene()
    frequency = 70_000.0
    artifact = _artifact(
        scene,
        frequency,
    )
    original = artifact.local_dissipation_matrix(
        scene,
        frequency,
        1,
        0.43,
        (1.5e-4, -1.0e-4),
    )

    rng = np.random.default_rng(7)
    common = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.2, -0.4, 0.3]
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
    )
    moved_value = (
        artifact.local_dissipation_matrix(
            moved,
            frequency,
            1,
            0.43,
            (1.5e-4, -1.0e-4),
        )
    )
    assert np.allclose(
        original,
        moved_value,
        rtol=5e-10,
        atol=5e-11,
    )


def test_spatial_field_is_zero_outside_conductor_section():
    scene = _scene()
    frequency = 70_000.0
    artifact = _artifact(
        scene,
        frequency,
    )
    matrix = artifact.local_dissipation_matrix(
        scene,
        frequency,
        0,
        0.5,
        (1.0, 1.0),
    )
    assert np.allclose(
        matrix,
        0.0,
        atol=0.0,
    )
