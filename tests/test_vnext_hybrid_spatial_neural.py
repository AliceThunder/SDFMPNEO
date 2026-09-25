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
from sdfmpneo_vnext.hybrid_spatial_neural import (
    HybridSpatialLossArtifact,
    HybridSpatialLossShapeNet,
)


def _scene():
    coil = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.020,
            0.8,
            0.0012,
            0.0012,
            exponent=3.2,
            conductor_width=1.0e-3,
            conductor_thickness=0.7e-3,
        ),
        ConductorMaterial(
            5.8e7
        ),
        "coil",
    )
    lossy = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [0.032, 0.026, 0.008]
            ),
            exponent_xy=3.0,
            exponent_z=3.5,
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
            relative_permittivity=3.0,
            conductivity=0.002,
        ),
        "lossy",
    )
    lossless = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray(
                [0.015, 0.012, 0.009]
            ),
            exponent_xy=2.5,
            exponent_z=3.0,
            pose=RigidPose.from_axis_angle(
                (0.0, 0.0, 1.0),
                -0.2,
                translation=(
                    0.04,
                    0.006,
                    0.012,
                ),
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=2.0,
            conductivity=0.0,
        ),
        "lossless",
    )
    return Scene(
        (coil,),
        HomogeneousMedium(),
        (lossy, lossless),
    )


def _port_artifact(scene):
    frequency = 75_000.0
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
                (),
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
    conductor = (
        0.8
        * target.real
    )
    dielectric = (
        0.2
        * target.real
    )
    sample = HybridTeacherSample(
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
        target_dissipation_channels=np.stack(
            (
                conductor,
                dielectric,
            ),
            axis=0,
        ).astype(
            complex
        ),
        baseline_segments=32,
        surface_vertical_order=8,
        surface_azimuthal_order=16,
        surface_residual=0.0,
        raw_potential_reciprocity_defect=0.0,
        power_closure_error=0.0,
    )
    normalizer = HybridNormalizer.fit(
        (sample,)
    )
    torch.manual_seed(
        101
    )
    model = (
        HybridPhysicsFactoredResidualNet(
            hidden_dim=20,
            factor_rank=2,
            depth=1,
        )
    )
    return HybridNeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=32,
    )


def _spatial_artifact(
    scene,
):
    port = _port_artifact(
        scene
    )
    torch.manual_seed(
        103
    )
    model = HybridSpatialLossShapeNet(
        port.model.hidden_dim,
        port.model.coil_pair_dim,
        port.model.cross_dim,
        field_hidden_dim=20,
        factor_rank=2,
        depth=1,
    )
    return HybridSpatialLossArtifact(
        port,
        model,
        conductor_longitudinal_points=6,
        conductor_radial_order=2,
        conductor_angular_order=8,
        package_axial_order=3,
        package_radial_order=2,
        package_azimuthal_order=8,
    )


def _assert_psd(
    matrix,
):
    matrix = np.asarray(
        matrix,
        dtype=complex,
    )
    assert np.allclose(
        matrix,
        matrix.conj().T,
        atol=3e-6,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                matrix
            )
        )
        >= -3e-6
    )


def test_hybrid_spatial_is_psd_closed_and_gates_lossless_package():
    scene = _scene()
    artifact = _spatial_artifact(
        scene
    )
    prepared = artifact.prepare(
        scene,
        75_000.0,
    )
    assert (
        prepared.normalization_closure_error
        < 2e-5
    )

    conductor = (
        prepared.local_dissipation_matrix(
            0,
            0.42,
            (0.0, 0.0),
        )
    )
    lossy = (
        prepared.package_local_dissipation_matrix(
            0,
            np.zeros(3),
        )
    )
    lossless = (
        prepared.package_local_dissipation_matrix(
            1,
            np.zeros(3),
        )
    )
    _assert_psd(
        conductor
    )
    _assert_psd(
        lossy
    )
    assert np.allclose(
        lossless,
        0.0,
        rtol=0,
        atol=0,
    )


def test_hybrid_package_world_and_local_queries_agree():
    scene = _scene()
    prepared = _spatial_artifact(
        scene
    ).prepare(
        scene,
        75_000.0,
    )
    local = np.asarray(
        [0.002, -0.001, 0.001]
    )
    world = (
        scene.packages[
            0
        ].geometry.local_to_world(
            local
        )
    )
    local_matrix = (
        prepared.package_local_dissipation_matrix(
            0,
            local,
        )
    )
    world_matrix = (
        prepared.package_dissipation_matrix(
            0,
            world,
        )
    )
    assert np.allclose(
        local_matrix,
        world_matrix,
        rtol=2e-6,
        atol=2e-8,
    )


def test_hybrid_spatial_is_common_se3_invariant():
    scene = _scene()
    artifact = _spatial_artifact(
        scene
    )
    reference = artifact.prepare(
        scene,
        75_000.0,
    )
    conductor_reference = (
        reference.local_dissipation_matrix(
            0,
            0.33,
            (0.8e-4, -0.4e-4),
        )
    )
    package_reference = (
        reference.package_local_dissipation_matrix(
            0,
            (0.001, 0.0, 0.001),
        )
    )

    rng = np.random.default_rng(
        109
    )
    common = RigidPose(
        haar_rotation(
            rng
        ),
        np.asarray(
            [0.21, -0.13, 0.31]
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
    actual = artifact.prepare(
        moved,
        75_000.0,
    )
    assert np.allclose(
        actual.local_dissipation_matrix(
            0,
            0.33,
            (0.8e-4, -0.4e-4),
        ),
        conductor_reference,
        rtol=5e-5,
        atol=5e-7,
    )
    assert np.allclose(
        actual.package_local_dissipation_matrix(
            0,
            (0.001, 0.0, 0.001),
        ),
        package_reference,
        rtol=5e-5,
        atol=5e-7,
    )


def test_unified_runtime_accepts_package_aware_fast_spatial_artifact():
    scene = _scene()
    spatial = _spatial_artifact(
        scene
    )
    system = MeshfreeVNextSystem(
        spatial.port_artifact,
        spatial_artifact=spatial,
    )
    prepared = system.fast_spatial(
        scene,
        75_000.0,
    )
    assert (
        prepared.normalization_closure_error
        < 2e-5
    )
    assert (
        prepared.dielectric_channel_index
        == len(
            scene.coils
        )
    )
