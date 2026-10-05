import numpy as np
import pytest

from sdfmpneo_vnext import (
    AnalyticBaselineArtifact,
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    MQSConfig,
    MeshfreeVNextSystem,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    TensorElectricMaterial,
    haar_rotation,
)


def _coil():
    return CoilObject(
        SuperellipseSpiral(
            0.022,
            0.019,
            0.70,
            0.0012,
            0.0010,
            conductor_width=8.0e-4,
            conductor_thickness=6.0e-4,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )


def _system():
    return MeshfreeVNextSystem(
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


def test_tensor_background_isotropic_limit_matches_scalar_reference():
    epsilon_r = 3.2
    conductivity = 1.4e-3
    frequency = 65_000.0
    coil = _coil()
    scalar_scene = Scene(
        (
            coil,
        ),
        HomogeneousMedium(
            relative_permittivity=epsilon_r,
            relative_permeability=1.0,
            conductivity=conductivity,
        ),
    )
    tensor_scene = Scene(
        (
            coil,
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=(
                epsilon_r
                * np.eye(
                    3
                )
            ),
            conductivity_tensor=(
                conductivity
                * np.eye(
                    3
                )
            ),
        ),
    )
    system = _system()
    scalar = system.reference_ports(
        scalar_scene,
        frequency,
    )
    tensor = system.reference_ports(
        tensor_scene,
        frequency,
    )

    assert np.allclose(
        tensor.impedance,
        scalar.impedance,
        rtol=3e-5,
        atol=3e-8,
    )
    assert np.allclose(
        tensor.dissipation_channels,
        scalar.dissipation_channels,
        rtol=5e-5,
        atol=5e-9,
    )


def test_tensor_background_reference_is_common_rotation_invariant():
    frequency = 80_000.0
    epsilon = np.asarray(
        [
            [2.1, 0.25, 0.0],
            [0.25, 4.4, 0.18],
            [0.0, 0.18, 6.2],
        ],
        dtype=float,
    )
    _, axes = np.linalg.eigh(
        epsilon
    )
    conductivity = (
        axes
        @ np.diag(
            np.asarray(
                [
                    8.0e-4,
                    1.5e-3,
                    2.4e-3,
                ]
            )
        )
        @ axes.T
    )
    coil = _coil()
    scene = Scene(
        (
            coil,
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=epsilon,
            conductivity_tensor=conductivity,
        ),
    )
    system = _system()
    reference = system.reference_ports(
        scene,
        frequency,
    )

    rng = np.random.default_rng(
        4117
    )
    rotation = haar_rotation(
        rng
    )
    common = RigidPose(
        rotation,
        np.asarray(
            [
                0.13,
                -0.09,
                0.21,
            ]
        ),
    )
    moved = Scene(
        (
            CoilObject(
                coil.geometry.transformed(
                    common
                ),
                coil.material,
                "coil",
            ),
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=(
                rotation
                @ epsilon
                @ rotation.T
            ),
            conductivity_tensor=(
                rotation
                @ conductivity
                @ rotation.T
            ),
        ),
    )
    actual = system.reference_ports(
        moved,
        frequency,
    )

    assert np.allclose(
        actual.impedance,
        reference.impedance,
        rtol=5e-5,
        atol=5e-8,
    )
    assert np.allclose(
        actual.dissipation_channels,
        reference.dissipation_channels,
        rtol=8e-5,
        atol=8e-9,
    )


def test_tensor_background_spatial_loss_uses_tensor_conductivity_and_closes():
    scene = Scene(
        (
            _coil(),
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=np.diag(
                [
                    2.0,
                    3.5,
                    5.5,
                ]
            ),
            conductivity_tensor=np.diag(
                [
                    5.0e-4,
                    1.4e-3,
                    3.0e-3,
                ]
            ),
        ),
    )
    system = _system()
    spatial = system.reference_spatial(
        scene,
        70_000.0,
        background_radial_order=10,
        background_angular_order=32,
        maximum_raw_closure_error=0.75,
        normalized_closure_tolerance=2e-6,
    )

    assert spatial.package_integrated_channels.shape == (
        0,
        1,
        1,
    )
    target = spatial.port_prediction.dissipation_channels[
        spatial.environment_channel_index
    ]
    assert np.allclose(
        spatial.background_integrated_channel,
        target,
        rtol=3e-6,
        atol=3e-9,
    )
    assert spatial.normalization_closure_error < 3e-6

    points = np.asarray(
        [
            [0.0, 0.0, 0.030],
            [0.035, 0.012, 0.020],
        ]
    )
    matrices = spatial.background_dissipation_matrices(
        points
    )
    assert np.all(
        np.isfinite(
            matrices
        )
    )
    assert np.all(
        np.linalg.eigvalsh(
            matrices
        )
        >= -1e-11
    )


def test_tensor_background_exact_dc_uses_environment_current_and_closes_loss():
    scene = Scene(
        (
            _coil(),
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=np.diag(
                [
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
            conductivity_tensor=np.diag(
                [
                    8.0e-4,
                    1.6e-3,
                    2.7e-3,
                ]
            ),
        ),
    )
    system = _system()
    result = system.reference_result(
        scene,
        0.0,
    )

    assert result.mixed_result.node_environment_current is not None
    assert np.allclose(
        result.mixed_result.node_charge,
        0.0,
        atol=0.0,
        rtol=0.0,
    )
    assert (
        np.linalg.norm(
            result.mixed_result.node_environment_current
        )
        > 0.0
    )
    assert np.allclose(
        result.impedance.imag,
        0.0,
        atol=1e-10,
        rtol=0.0,
    )
    assert (
        np.linalg.norm(
            result.dielectric_dissipation_matrix
        )
        > 0.0
    )
    assert result.power_closure_error < 5e-6

    spatial = system.reference_spatial(
        scene,
        0.0,
        background_radial_order=10,
        background_angular_order=32,
        maximum_raw_closure_error=0.75,
        normalized_closure_tolerance=3e-6,
    )
    target = spatial.port_prediction.dissipation_channels[
        spatial.environment_channel_index
    ]
    assert np.allclose(
        spatial.background_integrated_channel,
        target,
        rtol=4e-6,
        atol=4e-9,
    )


def test_scalar_fast_artifact_fails_closed_for_tensor_background():
    scene = Scene(
        (
            _coil(),
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=np.diag(
                [
                    2.0,
                    3.0,
                    4.0,
                ]
            ),
        ),
    )
    system = _system()
    with pytest.raises(
        NotImplementedError,
        match="tensor-electric",
    ):
        system.fast_ports(
            scene,
            60_000.0,
        )
