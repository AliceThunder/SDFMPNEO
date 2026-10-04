import numpy as np
import pytest

pytest.importorskip(
    "torch"
)

from sdfmpneo_vnext import (
    HybridSceneSamplerConfig,
    MQSConfig,
    TensorElectricMaterial,
)
from sdfmpneo_vnext.tensor_neural import (
    TensorHybridNeuralResidualArtifact,
    train_tensor_hybrid_residual_surrogate,
)
from sdfmpneo_vnext.tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from sdfmpneo_vnext.tensor_training_data import (
    TensorHybridTeacherSample,
)


CFG = MQSConfig(
    segments_per_turn=4,
    min_segments=6,
    section_degree=0,
    radial_order=2,
    angular_order=8,
    line_order=2,
)


def _sampler():
    return TensorHybridSceneSamplerConfig(
        base=HybridSceneSamplerConfig(
            package_count_range=(
                1,
                1,
            ),
            nested_package_probability=0.0,
            graded_package_probability=0.0,
            free_inclusion_probability=0.0,
            lossless_probability=0.0,
            lossy_background_probability=0.0,
            debye_package_probability=0.0,
            multi_debye_package_probability=0.0,
            dc_probability=0.0,
        ),
        tensor_package_probability=1.0,
        tensor_background_probability=0.0,
        tensor_relative_permittivity_range=(
            2.0,
            5.0,
        ),
        tensor_conductivity_range=(
            5.0e-5,
            4.0e-4,
        ),
        tensor_lossless_probability=0.0,
    )


def _teacher_sample(
    seed,
):
    scene, frequency = sample_tensor_hybrid_scene(
        np.random.default_rng(
            seed
        ),
        _sampler(),
    )
    assert any(
        isinstance(
            package.material,
            TensorElectricMaterial,
        )
        for package in scene.packages
    )
    return TensorHybridTeacherSample.generate(
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
    )


def test_tensor_fast_port_pipeline_trains_predicts_and_round_trips(tmp_path):
    samples = (
        _teacher_sample(
            901
        ),
        _teacher_sample(
            902
        ),
    )
    artifact, report = train_tensor_hybrid_residual_surrogate(
        samples,
        hidden_dim=16,
        factor_rank=2,
        depth=1,
        epochs=3,
        learning_rate=2e-3,
        patience=3,
        seed=13,
    )
    assert report.samples == 2
    assert report.epochs >= 1
    assert np.isfinite(
        report.final_loss
    )

    sample = samples[
        0
    ]
    prediction = artifact.predict_structured(
        sample.scene,
        sample.frequency_hz,
    )
    assert np.all(
        np.isfinite(
            prediction.impedance
        )
    )
    assert np.allclose(
        prediction.impedance,
        prediction.impedance.T,
        rtol=0.0,
        atol=1e-10,
    )
    assert np.allclose(
        np.sum(
            prediction.dissipation_channels,
            axis=0,
        ),
        0.5
        * (
            prediction.impedance
            + prediction.impedance.conj().T
        ).real,
        rtol=2e-8,
        atol=2e-9,
    )
    for channel in prediction.dissipation_channels:
        assert (
            np.min(
                np.linalg.eigvalsh(
                    0.5
                    * (
                        channel
                        + channel.conj().T
                    )
                )
            )
            >= -2e-9
        )

    path = tmp_path / "tensor-port.pt"
    artifact.save(
        path
    )
    restored = TensorHybridNeuralResidualArtifact.load(
        path
    )
    restored_prediction = restored.predict_structured(
        sample.scene,
        sample.frequency_hz,
    )
    assert np.allclose(
        restored_prediction.impedance,
        prediction.impedance,
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        restored_prediction.dissipation_channels,
        prediction.dissipation_channels,
        rtol=0.0,
        atol=1e-12,
    )


def test_tensor_fast_artifact_fails_closed_outside_material_eigenvalue_domain():
    sample = _teacher_sample(
        930
    )
    artifact, _ = train_tensor_hybrid_residual_surrogate(
        (
            sample,
        ),
        hidden_dim=12,
        factor_rank=2,
        depth=1,
        epochs=1,
        patience=1,
        seed=19,
    )
    package = sample.scene.packages[
        0
    ]
    material = package.material
    assert isinstance(
        material,
        TensorElectricMaterial,
    )
    stretched = TensorElectricMaterial(
        relative_permittivity_tensor=(
            50.0
            * np.asarray(
                material.relative_permittivity_tensor,
                dtype=float,
            )
        ),
        conductivity_tensor=(
            material.conductivity_tensor
        ),
        relative_permeability=(
            material.relative_permeability
        ),
    )
    from sdfmpneo_vnext import PackageObject, Scene

    moved = Scene(
        sample.scene.coils,
        sample.scene.medium,
        (
            PackageObject(
                package.geometry,
                stretched,
                package.name,
            ),
        ),
    )
    with pytest.raises(
        ValueError,
        match="permittivity principal values",
    ):
        artifact.predict_structured(
            moved,
            sample.frequency_hz,
        )
