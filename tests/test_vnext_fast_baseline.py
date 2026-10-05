import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    HomogeneousMedium,
    RigidPose,
    Scene,
    SuperellipseSpiral,
    analytic_port_baseline,
    encode_scene_invariant,
    haar_rotation,
)


COPPER = ConductorMaterial(5.8e7)


def _scene():
    a = CoilObject(
        SuperellipseSpiral(
            0.032,
            0.027,
            1.0,
            0.002,
            0.002,
            exponent=3.5,
            conductor_width=1.2e-3,
            conductor_thickness=0.8e-3,
        ),
        COPPER,
        "a",
    )
    b = CoilObject(
        SuperellipseSpiral(
            0.026,
            0.021,
            0.8,
            0.0015,
            0.0015,
            exponent=4.0,
            conductor_width=1.0e-3,
            conductor_thickness=0.7e-3,
            pose=RigidPose.from_axis_angle(
                (0.0, 1.0, 0.0),
                np.deg2rad(31.0),
                translation=(0.008, -0.004, 0.022),
            ),
        ),
        COPPER,
        "b",
    )
    return Scene(
        (a, b),
        HomogeneousMedium(),
    )


def test_analytic_baseline_is_reciprocal_passive_and_rigid_invariant():
    scene = _scene()
    base = analytic_port_baseline(
        scene,
        85_000.0,
        segments_per_coil=64,
    )
    assert np.allclose(
        base.impedance,
        base.impedance.T,
        rtol=0,
        atol=1e-12,
    )
    assert (
        np.min(
            np.linalg.eigvalsh(
                base.resistance
            )
        )
        > 0.0
    )

    rng = np.random.default_rng(9)
    common = RigidPose(
        haar_rotation(rng),
        np.array([0.2, -0.1, 0.3]),
    )
    moved = Scene(
        tuple(
            CoilObject(
                c.geometry.transformed(common),
                c.material,
                c.name,
            )
            for c in scene.coils
        ),
        scene.medium,
    )
    moved_base = analytic_port_baseline(
        moved,
        85_000.0,
        segments_per_coil=64,
    )
    assert np.allclose(
        base.impedance,
        moved_base.impedance,
        rtol=2e-12,
        atol=2e-12,
    )


def test_scene_encoding_is_exactly_common_se3_invariant_up_to_roundoff():
    scene = _scene()
    encoded = encode_scene_invariant(
        scene,
        85_000.0,
    )

    rng = np.random.default_rng(12)
    common = RigidPose(
        haar_rotation(rng),
        np.array([-0.4, 0.7, 0.2]),
    )
    moved = Scene(
        tuple(
            CoilObject(
                c.geometry.transformed(common),
                c.material,
                c.name,
            )
            for c in scene.coils
        ),
        scene.medium,
    )
    encoded_moved = encode_scene_invariant(
        moved,
        85_000.0,
    )

    assert np.allclose(
        encoded.node_features,
        encoded_moved.node_features,
        rtol=0,
        atol=1e-13,
    )
    assert np.allclose(
        encoded.pair_features,
        encoded_moved.pair_features,
        rtol=0,
        atol=2e-13,
    )
    assert np.isclose(
        encoded.length_scale,
        encoded_moved.length_scale,
        rtol=0,
        atol=1e-15,
    )
