import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    ConductorLossField,
    DenseMQSTeacher,
    HomogeneousMedium,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    certify_port_result,
    impedance_convergence,
)


def _scene():
    copper = ConductorMaterial(5.8e7)
    first = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.022,
            0.8,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ),
        copper,
    )
    second = CoilObject(
        SuperellipseSpiral(
            0.021,
            0.019,
            0.8,
            0.001,
            0.001,
            exponent=3.0,
            conductor_width=1e-3,
            conductor_thickness=8e-4,
        ).transformed(
            RigidPose(
                np.eye(3),
                np.array([0.0, 0.0, 0.018]),
            )
        ),
        copper,
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
    )


def test_continuous_loss_and_coil_power_close():
    cfg = MQSConfig(
        segments_per_turn=10,
        min_segments=12,
        section_degree=1,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )
    teacher = DenseMQSTeacher(
        _scene(),
        30_000.0,
        cfg,
    )
    result = teacher.solve()
    currents = np.array(
        [
            1.0 + 0.2j,
            -0.3 + 0.4j,
        ]
    )
    field = ConductorLossField(
        teacher,
        result,
        currents,
    )
    powers = field.coil_power()
    assert np.all(powers >= 0)
    assert np.isclose(
        np.sum(powers),
        result.conductor_power(currents),
        rtol=1e-12,
        atol=1e-14,
    )
    assert (
        field.local_joule_density(
            0,
            0.5,
            (0.0, 0.0),
        )
        >= 0
    )
    assert (
        field.local_joule_density(
            0,
            0.5,
            (1.0, 1.0),
        )
        == 0
    )


def test_port_certificate_passes_dense_mqs():
    cfg = MQSConfig(
        segments_per_turn=10,
        min_segments=12,
        section_degree=1,
        radial_order=3,
        angular_order=12,
        line_order=2,
    )
    result = DenseMQSTeacher(
        _scene(),
        20_000.0,
        cfg,
    ).solve()
    cert = certify_port_result(
        result,
        residual_tolerance=1e-8,
        power_tolerance=1e-8,
    )
    assert cert.certified
    assert (
        cert.reciprocity_defect
        < 1e-9
    )


def test_convergence_report_is_deterministic():
    scene = _scene()
    cfgs = [
        MQSConfig(
            segments_per_turn=8,
            min_segments=10,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
        MQSConfig(
            segments_per_turn=10,
            min_segments=12,
            section_degree=0,
            radial_order=3,
            angular_order=12,
            line_order=2,
        ),
    ]
    report = impedance_convergence(
        scene,
        15_000.0,
        cfgs,
        tolerance=1.0,
    )
    assert len(report.steps) == 2
    assert (
        report.steps[0].relative_change
        is None
    )
    assert np.isfinite(
        report.steps[1].relative_change
    )
    assert report.converged
