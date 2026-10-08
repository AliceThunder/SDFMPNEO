import numpy as np
import pytest

pytest.importorskip("torch")

from sdfmpneo_vnext import (
    HybridSceneSamplerConfig,
    MQSConfig,
    MeshfreeVNextSystem,
    TensorElectricMaterial,
)
from sdfmpneo_vnext.tensor_neural import train_tensor_hybrid_residual_surrogate
from sdfmpneo_vnext.tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from sdfmpneo_vnext.tensor_spatial_neural import (
    train_tensor_hybrid_spatial_loss_surrogate,
)
from sdfmpneo_vnext.tensor_teacher_pipeline import generate_tensor_teacher_once


CFG = MQSConfig(
    segments_per_turn=4,
    min_segments=6,
    section_degree=0,
    radial_order=2,
    angular_order=8,
    line_order=2,
)


def _background_only_sample(seed=1210):
    sampler = TensorHybridSceneSamplerConfig(
        base=HybridSceneSamplerConfig(
            package_count_range=(1, 1),
            nested_package_probability=0.0,
            graded_package_probability=0.0,
            free_inclusion_probability=0.0,
            lossless_probability=0.0,
            lossy_background_probability=0.0,
            debye_package_probability=0.0,
            multi_debye_package_probability=0.0,
            dc_probability=0.0,
        ),
        tensor_package_probability=0.0,
        tensor_background_probability=0.0,
        background_only_probability=1.0,
        tensor_relative_permittivity_range=(2.0, 5.0),
        tensor_conductivity_range=(5.0e-5, 4.0e-4),
        tensor_lossless_probability=0.0,
    )
    scene, frequency_hz = sample_tensor_hybrid_scene(
        np.random.default_rng(seed),
        sampler,
    )
    assert scene.packages == ()
    assert isinstance(scene.medium, TensorElectricMaterial)
    assert scene.medium.loss_conductivity(frequency_hz) > 0.0

    sample = generate_tensor_teacher_once(
        scene,
        frequency_hz,
        teacher_config=CFG,
        baseline_segments=16,
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        magnetic_volume_axial_order=2,
        magnetic_volume_radial_order=2,
        magnetic_volume_azimuthal_order=8,
        maximum_raw_magnetic_reciprocity_defect=0.25,
        include_spatial=True,
        package_volume_axial_order=2,
        package_volume_radial_order=2,
        package_volume_azimuthal_order=8,
        background_radial_order=3,
        background_angular_order=8,
        energy_volume_axial_order=2,
        energy_volume_radial_order=2,
        energy_volume_azimuthal_order=8,
        energy_background_radial_order=3,
        energy_background_angular_order=8,
        maximum_raw_spatial_closure_error=0.6,
    )
    assert len(sample.package_spatial_loss.package_index) == 0
    assert sample.background_spatial_loss is not None
    return sample


def test_tensor_background_only_trains_port_and_spatial_fast():
    sample = _background_only_sample()

    port, port_report = train_tensor_hybrid_residual_surrogate(
        (sample.port,),
        hidden_dim=12,
        factor_rank=2,
        depth=1,
        epochs=1,
        patience=1,
        seed=71,
    )
    assert np.isfinite(port_report.final_loss)
    assert tuple(port.material_domain["package_count"]) == (0, 0)
    assert port.material_domain["package_epsilon"] is None
    assert port.material_domain["package_sigma"] is None
    assert port.material_domain["package_mu"] is None

    prediction = port.predict_structured(sample.scene, sample.frequency_hz)
    assert np.all(np.isfinite(prediction.impedance))
    assert prediction.dissipation_channels.shape[0] == len(sample.scene.coils) + 1
    assert prediction.power_closure_error() < 1e-6

    spatial, spatial_report = train_tensor_hybrid_spatial_loss_surrogate(
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
        seed=73,
    )
    assert np.isfinite(spatial_report.final_loss)
    assert spatial.supports_lossy_background

    prepared = spatial.prepare(sample.scene, sample.frequency_hz)
    assert np.isfinite(prepared.normalization_closure_error)
    assert prepared.normalization_closure_error < 1e-6

    background = sample.background_spatial_loss
    root_pose = sample.scene.coils[0].geometry.pose
    world = root_pose.apply(background.root_local_position[:3])
    matrices = prepared.background_dissipation_matrices(world)
    assert np.all(np.isfinite(matrices))
    assert np.linalg.norm(matrices) > 0.0

    system = MeshfreeVNextSystem(port, spatial_artifact=spatial)
    system_prediction = system.fast_ports(sample.scene, sample.frequency_hz)
    assert np.all(np.isfinite(system_prediction.impedance))
    system_field = system.fast_spatial(sample.scene, sample.frequency_hz)
    assert system_field.normalization_closure_error < 1e-6
