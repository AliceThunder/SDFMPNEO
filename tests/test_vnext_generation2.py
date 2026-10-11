from types import SimpleNamespace

import numpy as np
import torch

from sdfmpneo_vnext.generation2_features import (
    GENERATION2_COIL_FEATURE_DIM,
    GENERATION2_CROSS_FEATURE_DIM,
    GENERATION2_PACKAGE_FEATURE_DIM,
    GENERATION2_PAIR_FEATURE_DIM,
    encode_generation2_scene,
)
from sdfmpneo_vnext.generation2_port import (
    Generation2Normalizer,
    Generation2PortArtifact,
    Generation2PortNet,
    forward_generation2_port_batch,
)
from sdfmpneo_vnext.generation2_spatial import (
    Generation2SpatialNet,
    generation2_batched_spatial_shape_loss,
)
from sdfmpneo_vnext.generation2_split import generation2_partition
from sdfmpneo_vnext.generation2_training import _initialize_baseline_residual_heads
from sdfmpneo_vnext.geometry import RigidPose, SuperellipseSpiral
from sdfmpneo_vnext.hybrid_training_data import PackageSpatialLossSamples
from sdfmpneo_vnext.package_geometry import SuperquadricPackageGeometry
from sdfmpneo_vnext.scene import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    PackageObject,
    Scene,
    TensorElectricMaterial,
)
from sdfmpneo_vnext.training_data import SpatialLossSamples


def test_generation2_partition_is_prefix_stable_and_disjoint():
    small = generation2_partition(512, seed=2027)
    large = generation2_partition(1024, seed=2027)
    small_membership = small.membership()
    large_membership = large.membership()
    assert set(small_membership) == set(range(512))
    assert set(large_membership) == set(range(1024))
    assert all(
        large_membership[index] == split
        for index, split in small_membership.items()
    )
    assert set(small.train_indices).isdisjoint(small.validation_indices)
    assert set(small.train_indices).isdisjoint(small.test_indices)
    assert set(small.validation_indices).isdisjoint(small.test_indices)


def _scene():
    conductor = ConductorMaterial(5.8e7)
    coil0 = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.021,
            1.0,
            1.2e-3,
            1.1e-3,
            conductor_width=1.0e-3,
            conductor_thickness=6.0e-4,
        ),
        conductor,
    )
    coil1 = CoilObject(
        SuperellipseSpiral(
            0.023,
            0.020,
            1.1,
            1.1e-3,
            1.0e-3,
            conductor_width=9.0e-4,
            conductor_thickness=5.0e-4,
            pose=RigidPose(translation=np.asarray([0.0, 0.0, 0.04])),
        ),
        conductor,
    )
    package = PackageObject(
        SuperquadricPackageGeometry(
            np.asarray([0.008, 0.006, 0.005]),
            pose=RigidPose.from_axis_angle(
                (0.0, 0.0, 1.0),
                0.4,
                translation=(0.06, 0.01, 0.015),
            ),
        ),
        TensorElectricMaterial(
            relative_permittivity_tensor=np.diag([2.5, 4.0, 7.0]),
            conductivity_tensor=np.diag([2e-5, 5e-5, 9e-5]),
        ),
    )
    background = TensorElectricMaterial(
        relative_permittivity_tensor=np.diag([1.2, 1.5, 1.8]),
        conductivity_tensor=np.diag([1e-6, 2e-6, 3e-6]),
    )
    return Scene((coil0, coil1), background, (package,))


def test_generation2_scene_features_have_explicit_physics_channels():
    encoded = encode_generation2_scene(_scene(), 85_000.0)
    assert encoded.coil.node_features.shape == (2, GENERATION2_COIL_FEATURE_DIM)
    assert encoded.coil.pair_features.shape == (2, 2, GENERATION2_PAIR_FEATURE_DIM)
    assert encoded.package_features.shape == (1, GENERATION2_PACKAGE_FEATURE_DIM)
    assert encoded.coil_package_features.shape == (2, 1, GENERATION2_CROSS_FEATURE_DIM)
    assert encoded.package_pair_features.shape == (1, 1, GENERATION2_PAIR_FEATURE_DIM)
    assert np.all(np.isfinite(encoded.coil.node_features))
    assert np.all(np.isfinite(encoded.package_features))
    assert np.all(np.isfinite(encoded.coil_package_features))


def _synthetic_port_batch(model, *, dtype=torch.float64):
    batch = 2
    ports = 2
    packages = 1
    generator = torch.Generator().manual_seed(7)
    coil = torch.randn(
        batch,
        ports,
        model.coil_dim,
        generator=generator,
        dtype=dtype,
    )
    coil_pair = torch.randn(
        batch,
        ports,
        ports,
        model.coil_pair_dim,
        generator=generator,
        dtype=dtype,
    )
    package = torch.randn(
        batch,
        packages,
        model.package_dim,
        generator=generator,
        dtype=dtype,
    )
    cross = torch.randn(
        batch,
        ports,
        packages,
        model.cross_dim,
        generator=generator,
        dtype=dtype,
    )
    package_pair = torch.randn(
        batch,
        packages,
        packages,
        model.package_pair_dim,
        generator=generator,
        dtype=dtype,
    )
    baseline_r = torch.tensor(
        [
            [[2.0, 0.25], [0.25, 1.3]],
            [[1.7, -0.15], [-0.15, 1.1]],
        ],
        dtype=dtype,
    )
    baseline_x = torch.tensor(
        [
            [[3.0, 0.7], [0.7, 2.1]],
            [[2.7, -0.4], [-0.4, 1.9]],
        ],
        dtype=dtype,
    )
    gates = torch.ones(batch, dtype=dtype)
    return coil, coil_pair, package, cross, package_pair, baseline_r, baseline_x, gates


def test_generation2_port_starts_from_baseline_and_channels_close_power():
    model = Generation2PortNet(
        hidden_dim=16,
        factor_rank=2,
        depth=1,
        interaction_rounds=2,
    ).to(dtype=torch.float64)
    _initialize_baseline_residual_heads(model)
    batch = _synthetic_port_batch(model)
    resistance, reactance, channels = forward_generation2_port_batch(
        model,
        *batch[:5],
        batch[5],
        batch[6],
        reactance_scale=1.0,
        dielectric_loss_gate=batch[7],
        reactance_gate=batch[7],
    )
    assert torch.allclose(resistance, batch[5], rtol=1e-10, atol=1e-11)
    assert torch.allclose(reactance, batch[6], rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        torch.sum(channels, dim=1).real,
        resistance,
        rtol=2e-7,
        atol=2e-9,
    )
    assert torch.max(torch.abs(torch.sum(channels, dim=1).imag)) < 2e-9
    for scene_channels in channels:
        for channel in scene_channels:
            eigenvalues = torch.linalg.eigvalsh(channel)
            assert float(torch.min(eigenvalues.real)) >= -1e-9


def test_generation2_total_psd_decoder_allows_signed_baseline_correction():
    model = Generation2PortNet(
        hidden_dim=16,
        factor_rank=2,
        depth=1,
        interaction_rounds=1,
    ).to(dtype=torch.float64)
    _initialize_baseline_residual_heads(model)
    batch = _synthetic_port_batch(model)

    with torch.no_grad():
        model.resistance_log_diag_head.bias.fill_(-np.log(2.0))
    lower, _, _ = forward_generation2_port_batch(
        model,
        *batch[:5],
        batch[5],
        batch[6],
        reactance_scale=1.0,
        dielectric_loss_gate=batch[7],
        reactance_gate=batch[7],
    )
    assert torch.allclose(lower, 0.5 * batch[5], rtol=1e-9, atol=1e-10)
    assert torch.all(torch.linalg.eigvalsh(lower) > 0.0)

    with torch.no_grad():
        model.resistance_log_diag_head.bias.fill_(np.log(2.0))
    upper, _, _ = forward_generation2_port_batch(
        model,
        *batch[:5],
        batch[5],
        batch[6],
        reactance_scale=1.0,
        dielectric_loss_gate=batch[7],
        reactance_gate=batch[7],
    )
    assert torch.allclose(upper, 2.0 * batch[5], rtol=1e-9, atol=1e-10)
    assert torch.all(torch.linalg.eigvalsh(upper) > 0.0)


def _neutral_normalizer():
    return Generation2Normalizer(
        np.zeros(GENERATION2_COIL_FEATURE_DIM),
        np.ones(GENERATION2_COIL_FEATURE_DIM),
        np.zeros(GENERATION2_PAIR_FEATURE_DIM),
        np.ones(GENERATION2_PAIR_FEATURE_DIM),
        np.zeros(GENERATION2_PACKAGE_FEATURE_DIM),
        np.ones(GENERATION2_PACKAGE_FEATURE_DIM),
        np.zeros(GENERATION2_CROSS_FEATURE_DIM),
        np.ones(GENERATION2_CROSS_FEATURE_DIM),
        np.zeros(GENERATION2_PAIR_FEATURE_DIM),
        np.ones(GENERATION2_PAIR_FEATURE_DIM),
        1.0,
    )


def test_generation2_spatial_training_target_does_not_call_port_prediction(monkeypatch):
    source = _scene()
    scene = Scene(source.coils, HomogeneousMedium(), source.packages)
    port_model = Generation2PortNet(
        hidden_dim=16,
        factor_rank=2,
        depth=1,
        interaction_rounds=1,
    ).to(dtype=torch.float64)
    port = Generation2PortArtifact(
        port_model,
        _neutral_normalizer(),
        baseline_segments=16,
        material_domain={},
        device="cpu",
    )
    spatial = Generation2SpatialNet(
        16,
        context_hidden_dim=16,
        context_rounds=1,
        context_depth=1,
        field_hidden_dim=16,
        factor_rank=2,
        depth=1,
    ).to(dtype=torch.float64)

    channels = np.asarray(
        [
            [[1.0, 0.1], [0.1, 0.5]],
            [[0.4, -0.05], [-0.05, 0.8]],
            [[0.2, 0.03], [0.03, 0.3]],
        ],
        dtype=complex,
    )
    conductor = SpatialLossSamples(
        np.asarray([0, 1], dtype=int),
        np.asarray([0.25, 0.75], dtype=float),
        np.zeros((2, 2), dtype=float),
        np.ones(2, dtype=float),
        np.asarray([channels[0], channels[1]], dtype=complex),
    )
    package = PackageSpatialLossSamples(
        np.asarray([0], dtype=int),
        np.zeros((1, 3), dtype=float),
        np.ones(1, dtype=float),
        channels[2:3],
    )
    sample = SimpleNamespace(
        scene=scene,
        frequency_hz=85_000.0,
        target_dissipation_channels=channels,
        conductor_spatial_loss=conductor,
        package_spatial_loss=package,
        background_spatial_loss=None,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("canonical Spatial training must not call Port prediction")

    monkeypatch.setattr(port, "predict_structured", forbidden)
    value = generation2_batched_spatial_shape_loss(
        spatial,
        port,
        (sample,),
        device="cpu",
        normalization={
            "conductor_longitudinal_points": 4,
            "conductor_radial_order": 2,
            "conductor_angular_order": 8,
            "package_axial_order": 2,
            "package_radial_order": 2,
            "package_azimuthal_order": 8,
            "background_segments_per_turn": 4,
            "background_radial_order": 3,
            "background_angular_order": 8,
        },
    )
    assert torch.isfinite(value)
