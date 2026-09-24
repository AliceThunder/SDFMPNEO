import numpy as np

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
)
from sdfmpneo_vnext.active_learning import (
    scene_regime_vector,
    score_active_learning_candidates,
    select_diverse_candidates,
)
from sdfmpneo_vnext.uncertainty import (
    FastErrorCalibrator,
    calibrated_fast_predict,
    fit_fast_error_calibrator,
)


def _scene(offset=0.018):
    copper = ConductorMaterial(5.8e7)
    first = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.022,
            0.8,
            0.0012,
            0.0012,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
        "a",
    )
    second = CoilObject(
        SuperellipseSpiral(
            0.020,
            0.018,
            0.7,
            0.0010,
            0.0010,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.004, 0.0, offset]),
            ),
        ),
        copper,
        "b",
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
    )


def _sample(scene, frequency):
    baseline = analytic_port_baseline(
        scene,
        frequency,
        segments_per_coil=32,
    )
    target = baseline.impedance.copy()
    return TeacherSample(
        scene,
        frequency,
        encode_scene_invariant(
            scene,
            frequency,
        ),
        baseline.resistance,
        target.imag,
        target,
        32,
    )


class _ScaledArtifact:
    def __init__(self, scale):
        self.scale = float(scale)

    def predict(self, scene, frequency_hz):
        return (
            self.scale
            * analytic_port_baseline(
                scene,
                frequency_hz,
                segments_per_coil=32,
            ).impedance
        )


def test_regime_vector_and_candidate_selection_are_deterministic():
    training = (
        _sample(
            _scene(0.018),
            50_000.0,
        ),
    )
    candidates = (
        (
            _scene(0.020),
            60_000.0,
        ),
        (
            _scene(0.030),
            90_000.0,
        ),
        (
            _scene(0.040),
            120_000.0,
        ),
    )
    artifacts = (
        _ScaledArtifact(1.00),
        _ScaledArtifact(1.02),
    )
    first = score_active_learning_candidates(
        artifacts,
        training,
        candidates,
        baseline_segments=32,
    )
    second = score_active_learning_candidates(
        artifacts,
        training,
        candidates,
        baseline_segments=32,
    )
    assert [
        x.acquisition_score
        for x in first
    ] == [
        x.acquisition_score
        for x in second
    ]
    selected = select_diverse_candidates(
        first,
        2,
    )
    assert len(
        selected
    ) == 2
    assert all(
        np.isfinite(
            item.acquisition_score
        )
        for item in selected
    )
    assert (
        scene_regime_vector(
            candidates[0][0],
            candidates[0][1],
        ).ndim
        == 1
    )


def test_validation_only_error_calibrator_round_trip(tmp_path):
    validation = tuple(
        _sample(
            _scene(
                0.018 + 0.004 * index
            ),
            50_000.0
            + 10_000.0 * index,
        )
        for index in range(4)
    )
    artifacts = (
        _ScaledArtifact(1.05),
        _ScaledArtifact(1.07),
    )
    calibrator = fit_fast_error_calibrator(
        artifacts,
        validation,
        quantile=0.75,
        baseline_segments=32,
    )
    assert (
        calibrator.validation_samples
        == len(validation)
    )
    assert calibrator.scale >= 0.0
    path = tmp_path / "calibrator.json"
    calibrator.save(path)
    loaded = FastErrorCalibrator.load(
        path
    )
    assert loaded == calibrator

    prediction = calibrated_fast_predict(
        artifacts,
        loaded,
        _scene(0.025),
        80_000.0,
        baseline_segments=32,
    )
    assert prediction.relative_error_bound >= 0.0
    assert prediction.indicator >= 0.0
    assert prediction.impedance.shape == (
        2,
        2,
    )
