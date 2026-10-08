import numpy as np
import pytest

pytest.importorskip("torch")

from sdfmpneo_vnext import (
    HomogeneousThermalMedium,
    HybridSceneSamplerConfig,
    MQSConfig,
    MeshfreeVNextSystem,
)
from sdfmpneo_vnext.tensor_bundle import (
    load_tensor_bundle,
    publish_tensor_bundle,
)
from sdfmpneo_vnext.tensor_neural import train_tensor_hybrid_residual_surrogate
from sdfmpneo_vnext.tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from sdfmpneo_vnext.tensor_spatial_neural import (
    TensorHybridSpatialLossArtifact,
    train_tensor_hybrid_spatial_loss_surrogate,
)
from sdfmpneo_vnext.tensor_spatial_training_data import (
    TensorHybridSpatialTeacherSample,
)
from sdfmpneo_vnext.tensor_training_data import TensorHybridTeacherSample


CFG = MQSConfig(
    segments_per_turn=4,
    min_segments=6,
    section_degree=0,
    radial_order=2,
    angular_order=8,
    line_order=2,
)


def _sampler(*, tensor_background=False):
    return TensorHybridSceneSamplerConfig(
        base=HybridSceneSamplerConfig(
            package_count_range=(1, 1),
            nested_package_probability=0.0,
            graded_package_probability=0.0,
            free_inclusion_probability=0.0,
            lossless_probability=0.0,
            lossy_background_probability=1.0 if tensor_background else 0.0,
            debye_package_probability=0.0,
            multi_debye_package_probability=0.0,
            dc_probability=0.0,
        ),
        tensor_package_probability=1.0,
        tensor_background_probability=1.0 if tensor_background else 0.0,
        tensor_relative_permittivity_range=(2.0, 5.0),
        tensor_conductivity_range=(5.0e-5, 4.0e-4),
        tensor_lossless_probability=0.0,
    )


def _sample(seed, *, tensor_background=False):
    scene, frequency = sample_tensor_hybrid_scene(
        np.random.default_rng(seed),
        _sampler(tensor_background=tensor_background),
    )
    port = TensorHybridTeacherSample.generate(
        scene,
        frequency,
        teacher_config=CFG,
        baseline_segments=16,
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        magnetic_volume_axial_order=2,
        magnetic_volume_radial_order=2,
        magnetic_volume_azimuthal_order=8,
        maximum_raw_magnetic_reciprocity_defect=0.25,
        package_volume_axial_order=2,
        package_volume_radial_order=2,
        package_volume_azimuthal_order=8,
        background_radial_order=3,
        background_angular_order=8,
    )
    return TensorHybridSpatialTeacherSample.generate(
        port,
        teacher_config=CFG,
        package_volume_axial_order=2,
        package_volume_radial_order=2,
        package_volume_azimuthal_order=8,
        background_radial_order=3,
        background_angular_order=8,
        maximum_raw_spatial_closure_error=0.6,
    )


def _train(sample):
    port, _ = train_tensor_hybrid_residual_surrogate(
        (sample.port,),
        hidden_dim=12,
        factor_rank=2,
        depth=1,
        epochs=1,
        patience=1,
        seed=53,
    )
    spatial, report = train_tensor_hybrid_spatial_loss_surrogate(
        port,
        (sample,),
        field_hidden_dim=12,
        factor_rank=2,
        depth=1,
        epochs=1,
        patience=1,
        conductor_longitudinal_points=4,
        conductor_radial_order=2,
        conductor_angular_order=8,
        package_axial_order=2,
        package_radial_order=2,
        package_azimuthal_order=8,
        background_segments_per_turn=4,
        background_radial_order=3,
        background_angular_order=8,
        seed=59,
    )
    assert np.isfinite(report.final_loss)
    return port, spatial


def test_tensor_spatial_fast_trains_prepares_round_trips_and_bundles(tmp_path):
    sample = _sample(1201)
    port, spatial = _train(sample)
    prepared = spatial.prepare(sample.scene, sample.frequency_hz)
    assert spatial.supports_tensor_electric
    assert np.isfinite(prepared.normalization_closure_error)
    assert prepared.normalization_closure_error < 1e-6

    conductor = sample.conductor_spatial_loss
    matrices = prepared.local_dissipation_matrices(
        conductor.coil_index[:4],
        conductor.arc_fraction[:4],
        conductor.xy[:4],
    )
    assert np.all(np.isfinite(matrices))
    for matrix in matrices:
        assert np.min(np.linalg.eigvalsh(0.5 * (matrix + matrix.conj().T))) >= -1e-9

    path = tmp_path / "tensor-spatial.pt"
    spatial.save(path)
    restored = TensorHybridSpatialLossArtifact.load(path, port)
    restored_prepared = restored.prepare(sample.scene, sample.frequency_hz)
    restored_matrices = restored_prepared.local_dissipation_matrices(
        conductor.coil_index[:4],
        conductor.arc_fraction[:4],
        conductor.xy[:4],
    )
    assert np.allclose(restored_matrices, matrices, rtol=0.0, atol=1e-12)

    system = MeshfreeVNextSystem(port, spatial_artifact=restored)
    system_field = system.fast_spatial(sample.scene, sample.frequency_hz)
    assert system_field.normalization_closure_error < 1e-6
    thermal = system.fast_continuous_thermal_field(
        sample.scene,
        sample.frequency_hz,
        HomogeneousThermalMedium(
            conductivity=0.45,
            density=1100.0,
            heat_capacity=1300.0,
        ),
        longitudinal_segments=4,
        radial_order=2,
        angular_order=8,
        package_axial_order=2,
        package_radial_order=2,
        package_azimuthal_order=8,
        background_radial_order=3,
        background_angular_order=8,
    )
    value = thermal.temperature_step(
        np.array([0.0, 0.0, 0.05]),
        0.1,
        np.ones(len(sample.scene.coils), dtype=complex),
    )
    assert np.isfinite(value)

    bundle_path = tmp_path / "tensor-bundle"
    manifest = publish_tensor_bundle(
        bundle_path,
        port,
        spatial_artifact=restored,
    )
    assert manifest["artifact_family"] == "tensor_hybrid"
    loaded = load_tensor_bundle(bundle_path)
    loaded_ports = loaded.system.fast_ports(sample.scene, sample.frequency_hz)
    expected_ports = port.predict_structured(sample.scene, sample.frequency_hz)
    assert np.allclose(
        loaded_ports.impedance,
        expected_ports.impedance,
        rtol=0.0,
        atol=1e-12,
    )
    loaded_field = loaded.system.fast_spatial(sample.scene, sample.frequency_hz)
    assert loaded_field.normalization_closure_error < 1e-6


def test_tensor_spatial_fast_supports_lossy_tensor_background():
    sample = _sample(1210, tensor_background=True)
    port, spatial = _train(sample)
    assert spatial.supports_lossy_background
    prepared = spatial.prepare(sample.scene, sample.frequency_hz)
    background = sample.background_spatial_loss
    assert background is not None
    world = sample.scene.coils[0].geometry.pose.apply(
        background.root_local_position[:3]
    )
    matrices = prepared.background_dissipation_matrices(world)
    assert np.all(np.isfinite(matrices))
    assert np.linalg.norm(matrices) > 0.0
