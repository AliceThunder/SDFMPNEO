import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DenseMQSTeacher,
    HomogeneousMedium,
    MQSConfig,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    coaxial_circular_mutual_inductance,
    haar_rotation,
)

COPPER = ConductorMaterial(5.8e7)
CFG = MQSConfig(
    segments_per_turn=12,
    min_segments=16,
    section_degree=1,
    radial_order=3,
    angular_order=16,
    line_order=2,
)


def circular_coil(radius, z=0.0, pose=None):
    p = pose or RigidPose(
        np.eye(3),
        np.array([0.0, 0.0, z]),
    )
    return CoilObject(
        SuperellipseSpiral(
            radius,
            radius,
            1.0,
            0.0,
            0.0,
            exponent=2.0,
            conductor_width=8e-4,
            conductor_thickness=8e-4,
            pose=p,
        ),
        COPPER,
    )


def test_dc_resistance_and_power_closure():
    scene = Scene(
        (circular_coil(0.03),),
        HomogeneousMedium(),
    )
    teacher = DenseMQSTeacher(scene, 0.0, CFG)
    result = teacher.solve()
    poly = scene.coils[0].geometry.polyline(
        max(CFG.min_segments, CFG.segments_per_turn)
    )
    expected = (
        poly.total_length
        / (
            COPPER.conductivity
            * scene.coils[0].geometry.cross_section_area
        )
    )
    assert np.isclose(
        result.impedance[0, 0].real,
        expected,
        rtol=1e-10,
    )
    assert abs(result.impedance[0, 0].imag) < 1e-14
    i = np.array([1.2 - 0.7j])
    assert np.isclose(
        result.port_power(i),
        result.conductor_power(i),
        rtol=2e-12,
        atol=1e-15,
    )


def test_reciprocity_passivity_and_mutual_inductance():
    a, b, gap = 0.03, 0.025, 0.02
    scene = Scene(
        (
            circular_coil(a),
            circular_coil(b, z=gap),
        ),
        HomogeneousMedium(),
    )
    f = 50_000.0
    result = DenseMQSTeacher(scene, f, CFG).solve()
    assert np.allclose(
        result.impedance,
        result.impedance.T,
        rtol=2e-10,
        atol=2e-11,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(result.impedance.real)
        )
        >= -1e-10
    )
    m_num = (
        result.impedance[0, 1].imag
        / (2 * np.pi * f)
    )
    m_ref = coaxial_circular_mutual_inductance(
        a,
        b,
        gap,
    )
    assert np.isclose(
        m_num,
        m_ref,
        rtol=0.08,
    )


def test_common_rigid_motion_leaves_port_impedance_invariant():
    base = Scene(
        (
            circular_coil(0.03),
            circular_coil(0.024, z=0.018),
        ),
        HomogeneousMedium(),
    )
    rng = np.random.default_rng(17)
    common = RigidPose(
        haar_rotation(rng),
        np.array([0.11, -0.07, 0.19]),
    )
    moved = Scene(
        tuple(
            CoilObject(
                c.geometry.transformed(common),
                c.material,
                c.name,
            )
            for c in base.coils
        ),
        base.medium,
    )
    z0 = DenseMQSTeacher(
        base,
        40_000.0,
        CFG,
    ).solve().impedance
    z1 = DenseMQSTeacher(
        moved,
        40_000.0,
        CFG,
    ).solve().impedance
    assert np.allclose(
        z0,
        z1,
        rtol=3e-10,
        atol=3e-11,
    )
