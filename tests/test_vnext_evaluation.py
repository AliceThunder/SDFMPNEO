import numpy as np

from sdfmpneo_vnext import (
    HomogeneousMedium,
    CoilObject,
    ConductorMaterial,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    TeacherSample,
    audit_surrogate,
    analytic_port_baseline,
    encode_scene_invariant,
)


def _sample():
    copper = ConductorMaterial(5.8e7)
    a = CoilObject(
        SuperellipseSpiral(
            0.03,
            0.026,
            0.8,
            0.0015,
            0.0015,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
    )
    b = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.021,
            0.7,
            0.0012,
            0.0012,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.005, 0.0, 0.02]),
            ),
        ),
        copper,
    )
    scene = Scene((a, b), HomogeneousMedium())
    frequency = 65_000.0
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


class _ExactArtifact:
    def __init__(self, sample):
        self.sample = sample

    def predict(self, scene, frequency_hz):
        return self.sample.target_impedance.copy()


class _BadArtifact:
    def predict(self, scene, frequency_hz):
        return np.array(
            [
                [-0.1 + 0.2j, 1.0 + 0.0j],
                [0.0 + 0.0j, -0.2 + 0.1j],
            ],
            dtype=complex,
        )


def test_surrogate_audit_passes_exact_physical_artifact():
    sample = _sample()
    audit = audit_surrogate(
        _ExactArtifact(sample),
        (sample,),
    )
    assert audit.passed
    assert audit.maximum_relative_error == 0.0
    assert audit.maximum_reciprocity_defect < 1e-14
    assert audit.minimum_dissipation_eigenvalue > 0.0


def test_surrogate_audit_rejects_nonreciprocal_active_artifact():
    sample = _sample()
    audit = audit_surrogate(
        _BadArtifact(),
        (sample,),
    )
    assert not audit.passed
    assert audit.maximum_reciprocity_defect > 0.0
    assert audit.minimum_dissipation_eigenvalue < 0.0
