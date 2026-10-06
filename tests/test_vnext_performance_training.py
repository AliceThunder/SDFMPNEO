import numpy as np
import pytest


torch = pytest.importorskip("torch")

from sdfmpneo_vnext import HybridSceneSamplerConfig, Scene
from sdfmpneo_vnext.analytic_baseline import analytic_port_baseline
from sdfmpneo_vnext.performance import (
    train_tensor_hybrid_residual_surrogate_accelerated,
)
from sdfmpneo_vnext.tensor_features import encode_tensor_hybrid_scene_invariant
from sdfmpneo_vnext.tensor_sampling import (
    TensorHybridSceneSamplerConfig,
    sample_tensor_hybrid_scene,
)
from sdfmpneo_vnext.tensor_training_data import TensorHybridTeacherSample


def _sampler():
    return TensorHybridSceneSamplerConfig(
        base=HybridSceneSamplerConfig(
            package_count_range=(1, 1),
            nested_package_probability=0.0,
            graded_package_probability=0.0,
            free_inclusion_probability=0.0,
            lossy_background_probability=0.0,
            debye_package_probability=0.0,
            multi_debye_package_probability=0.0,
            dc_probability=0.0,
        ),
        tensor_package_probability=1.0,
        tensor_background_probability=0.0,
        tensor_relative_permittivity_range=(2.0, 4.0),
        tensor_conductivity_range=(1e-5, 2e-4),
        tensor_lossless_probability=0.0,
    )


def _synthetic_sample(seed):
    scene, frequency_hz = sample_tensor_hybrid_scene(
        np.random.default_rng(seed),
        _sampler(),
    )
    baseline = analytic_port_baseline(
        Scene(scene.coils, scene.medium, ()),
        frequency_hz,
        segments_per_coil=8,
    )
    resistance = np.asarray(baseline.resistance, dtype=float)
    reactance = 2.0 * np.pi * frequency_hz * np.asarray(
        baseline.inductance,
        dtype=float,
    )
    n = len(scene.coils)
    target_impedance = resistance + 1j * reactance
    channels = np.repeat(
        (resistance / (n + 1))[None, :, :].astype(complex),
        n + 1,
        axis=0,
    )
    return TensorHybridTeacherSample(
        scene=scene,
        frequency_hz=frequency_hz,
        encoded=encode_tensor_hybrid_scene_invariant(scene, frequency_hz),
        baseline_resistance=resistance,
        baseline_reactance=reactance,
        target_impedance=target_impedance,
        target_dissipation_channels=channels,
        baseline_segments=8,
        surface_vertical_order=4,
        surface_azimuthal_order=8,
        surface_residual=0.0,
        raw_potential_reciprocity_defect=0.0,
        power_closure_error=0.0,
        magnetic_surface_residual=0.0,
        raw_magnetic_reciprocity_defect=0.0,
    )


def test_accelerated_tensor_training_runs_vectorized_batches():
    samples = tuple(_synthetic_sample(seed) for seed in (301, 302, 303))
    artifact, report = train_tensor_hybrid_residual_surrogate_accelerated(
        samples[:2],
        validation_samples=samples[2:],
        hidden_dim=8,
        factor_rank=2,
        depth=1,
        epochs=2,
        learning_rate=1e-3,
        patience=2,
        batch_size=2,
        precision="float64",
        device="cpu",
        seed=9,
    )
    assert report.samples == 2
    assert report.epochs >= 1
    assert np.isfinite(report.final_loss)
    assert next(artifact.model.parameters()).dtype == torch.float64
    prediction = artifact.predict_structured(
        samples[0].scene,
        samples[0].frequency_hz,
    )
    assert prediction.impedance.shape == samples[0].target_impedance.shape
    assert np.all(np.isfinite(prediction.impedance))
    assert np.allclose(
        np.sum(prediction.dissipation_channels, axis=0),
        prediction.impedance.real,
        rtol=1e-8,
        atol=1e-9,
    )
