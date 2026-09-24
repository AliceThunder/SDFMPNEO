import numpy as np

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorLossField,
    ConductorMaterial,
    DenseMQSTeacher,
    HomogeneousMedium,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    UniformLossFieldDecoder,
)


def _scene():
    copper = ConductorMaterial(5.8e7)
    a = CoilObject(
        SuperellipseSpiral(
            0.026,
            0.022,
            0.8,
            0.0012,
            0.0012,
            exponent=3.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.8e-3,
        ),
        copper,
        "a",
    )
    b = CoilObject(
        SuperellipseSpiral(
            0.022,
            0.019,
            0.7,
            0.001,
            0.001,
            exponent=4.0,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.005, 0.0, 0.018]),
            ),
        ),
        copper,
        "b",
    )
    return Scene((a, b), HomogeneousMedium())


def _config():
    return MQSConfig(
        segments_per_turn=8,
        min_segments=10,
        section_degree=1,
        radial_order=4,
        angular_order=24,
        line_order=2,
    )


def test_teacher_continuous_local_loss_integrates_to_coil_channels():
    scene = _scene()
    teacher = DenseMQSTeacher(
        scene,
        30_000.0,
        _config(),
    )
    result = teacher.solve()
    channels = result.coil_dissipation_matrices()

    for coil_index in range(len(scene.coils)):
        segments = [
            segment
            for segment in teacher._segments
            if segment.coil == coil_index
        ]
        integrated = np.zeros_like(
            channels[coil_index]
        )
        for local_index, segment in enumerate(segments):
            arc = (
                local_index + 0.5
            ) / len(segments)
            quadrature = segment.basis.quadrature
            for xy, weight in zip(
                quadrature.xy,
                quadrature.weights,
            ):
                matrix = teacher.local_dissipation_matrix(
                    result,
                    coil_index,
                    arc,
                    xy,
                )
                assert np.allclose(
                    matrix,
                    matrix.conj().T,
                    atol=1e-12,
                )
                assert (
                    np.min(
                        np.linalg.eigvalsh(matrix)
                    )
                    >= -1e-11
                )
                integrated += (
                    matrix
                    * weight
                    * segment.length
                )

        assert np.allclose(
            integrated,
            channels[coil_index],
            rtol=2e-10,
            atol=2e-12,
        )

    currents = np.array(
        [1.1 - 0.2j, -0.4 + 0.3j]
    )
    field = ConductorLossField(
        teacher,
        result,
        currents,
    )
    matrix = field.local_dissipation_matrix(
        0,
        0.4,
        (0.0, 0.0),
    )
    expected = 0.5 * np.real(
        np.vdot(
            currents,
            matrix @ currents,
        )
    )
    assert np.isclose(
        field.local_joule_density(
            0,
            0.4,
            (0.0, 0.0),
        ),
        expected,
        rtol=1e-13,
        atol=1e-15,
    )


def test_uniform_fast_field_is_psd_zero_outside_and_power_closed():
    scene = _scene()
    artifact = AnalyticBaselineArtifact(
        segments_per_coil=48,
    )
    decoder = UniformLossFieldDecoder(
        artifact,
        length_segments=48,
    )
    frequency = 30_000.0
    prediction = artifact.predict_structured(
        scene,
        frequency,
    )

    for coil_index, coil in enumerate(scene.coils):
        matrix = decoder.local_dissipation_matrix(
            scene,
            frequency,
            coil_index,
            0.37,
            (0.0, 0.0),
        )
        assert np.allclose(
            matrix,
            matrix.conj().T,
            atol=1e-14,
        )
        assert (
            np.min(
                np.linalg.eigvalsh(matrix)
            )
            >= -1e-14
        )
        volume = (
            coil.geometry.polyline(48).total_length
            * coil.geometry.cross_section_area
        )
        assert np.allclose(
            matrix * volume,
            prediction.dissipation_channels[
                coil_index
            ],
            rtol=2e-12,
            atol=2e-14,
        )

        outside = decoder.local_dissipation_matrix(
            scene,
            frequency,
            coil_index,
            0.37,
            (1.0, 1.0),
        )
        assert np.allclose(
            outside,
            0.0,
            atol=0.0,
        )
