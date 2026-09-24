import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DenseMixedConductorTeacher,
    HomogeneousMedium,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
)

COPPER = ConductorMaterial(5.8e7)
CFG = MQSConfig(
    segments_per_turn=10,
    min_segments=12,
    section_degree=1,
    radial_order=3,
    angular_order=12,
    line_order=2,
)


def coil(radius, z=0.0):
    return CoilObject(
        SuperellipseSpiral(
            radius,
            radius,
            0.95,
            0.0,
            0.0,
            conductor_width=1e-3,
            conductor_thickness=1e-3,
            pose=RigidPose(
                np.eye(3),
                np.array([0.0, 0.0, z]),
            ),
        ),
        COPPER,
    )


def test_mixed_dc_uses_same_finite_equations_and_matches_resistance():
    scene = Scene(
        (coil(0.025),),
        HomogeneousMedium(),
    )
    result = DenseMixedConductorTeacher(
        scene,
        0.0,
        CFG,
    ).solve()
    nseg = max(
        CFG.min_segments,
        CFG.segments_per_turn,
    )
    poly = scene.coils[0].geometry.polyline(
        nseg
    )
    expected = (
        poly.total_length
        / (
            COPPER.conductivity
            * scene.coils[
                0
            ].geometry.cross_section_area
        )
    )
    assert np.isclose(
        result.impedance[0, 0].real,
        expected,
        rtol=5e-10,
    )
    assert (
        abs(
            result.impedance[0, 0].imag
        )
        < 1e-12
    )
    assert (
        result.normalized_residual
        < 1e-10
    )
    assert (
        result.continuity_residual(
            np.array([1.0 + 0j]),
            0.0,
        )
        < 1e-10
    )


def test_mixed_reciprocity_power_and_continuity_at_frequency():
    scene = Scene(
        (
            coil(0.025),
            coil(
                0.02,
                z=0.018,
            ),
        ),
        HomogeneousMedium(),
    )
    f = 20_000.0
    result = DenseMixedConductorTeacher(
        scene,
        f,
        CFG,
    ).solve()
    assert np.allclose(
        result.impedance,
        result.impedance.T,
        rtol=2e-8,
        atol=2e-9,
    )
    currents = np.array(
        [
            1.1 - 0.3j,
            -0.4 + 0.7j,
        ]
    )
    assert np.isclose(
        result.port_power(currents),
        result.conductor_power(currents),
        rtol=2e-7,
        atol=1e-10,
    )
    assert (
        result.continuity_residual(
            currents,
            2 * np.pi * f,
        )
        < 1e-10
    )
    assert (
        result.normalized_residual
        < 1e-10
    )


def test_mixed_rejects_closed_coincident_terminals():
    closed = CoilObject(
        SuperellipseSpiral(
            0.025,
            0.025,
            1.0,
            0.0,
            0.0,
            conductor_width=1e-3,
            conductor_thickness=1e-3,
        ),
        COPPER,
    )
    scene = Scene(
        (closed,),
        HomogeneousMedium(),
    )
    try:
        DenseMixedConductorTeacher(
            scene,
            10_000.0,
            CFG,
        ).solve()
    except ValueError as exc:
        assert (
            "terminal" in str(exc)
            or "charge nodes" in str(exc)
        )
    else:
        raise AssertionError(
            "closed coincident terminals must be rejected"
        )
