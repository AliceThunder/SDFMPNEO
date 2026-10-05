import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    TeacherSample,
)
from sdfmpneo_vnext.reference import MixedReferenceArtifact
from sdfmpneo_vnext.spatial_evaluation import audit_spatial_surrogate


def _scene():
    copper = ConductorMaterial(5.8e7)
    first = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.021,
            0.70,
            0.0010,
            0.0010,
            exponent=3.0,
            conductor_width=0.9e-3,
            conductor_thickness=0.7e-3,
        ),
        copper,
        "a",
    )
    second = CoilObject(
        SuperellipseSpiral(
            0.020,
            0.018,
            0.65,
            0.0009,
            0.0009,
            exponent=3.5,
            conductor_width=0.8e-3,
            conductor_thickness=0.6e-3,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                0.31,
                translation=(0.004, 0.0, 0.018),
            ),
        ),
        copper,
        "b",
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
    )


def test_reference_passes_offgrid_and_crossgrid_spatial_audit():
    config = MQSConfig(
        segments_per_turn=7,
        min_segments=8,
        section_degree=1,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )
    scene = _scene()
    sample = TeacherSample.generate(
        scene,
        35_000.0,
        teacher_config=config,
        baseline_segments=24,
        reference_backend="mixed",
    )
    artifact = MixedReferenceArtifact(
        config=config,
    )
    report = audit_spatial_surrogate(
        artifact,
        (sample,),
        mean_relative_error_limit=1e-8,
        maximum_relative_error_limit=1e-8,
        maximum_probe_joule_error_limit=1e-8,
        mean_offgrid_relative_error_limit=1e-8,
        maximum_offgrid_relative_error_limit=1e-8,
        maximum_offgrid_probe_joule_error_limit=1e-8,
        cross_grid_closure_tolerance=5e-2,
        channel_closure_tolerance=1e-8,
    )
    assert report.passed
    assert report.mean_offgrid_relative_error < 1e-8
    assert report.maximum_offgrid_probe_joule_error < 1e-8
    assert report.maximum_cross_grid_closure_error < 5e-2
    assert report.minimum_local_eigenvalue >= -1e-10
