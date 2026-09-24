import numpy as np

from sdfmpneo_vnext import (
    CoilObject,
    ConductorMaterial,
    DenseMQSTeacher,
    HomogeneousMedium,
    MQSConfig,
    MatrixFreeMQSOperator,
    RigidPose,
    Scene,
    SuperellipseSpiral,
)


def _scene():
    copper = ConductorMaterial(
        5.8e7
    )
    first = CoilObject(
        SuperellipseSpiral(
            0.024,
            0.021,
            0.65,
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
            0.017,
            0.60,
            0.0009,
            0.0009,
            exponent=4.0,
            conductor_width=0.8e-3,
            conductor_thickness=0.6e-3,
            pose=RigidPose(
                np.eye(3),
                np.array(
                    [0.004, 0.0, 0.016]
                ),
            ),
        ),
        copper,
        "b",
    )
    return Scene(
        (first, second),
        HomogeneousMedium(),
    )


def _config():
    return MQSConfig(
        segments_per_turn=6,
        min_segments=6,
        section_degree=1,
        radial_order=2,
        angular_order=8,
        line_order=2,
    )


def test_matrix_free_inductance_matches_dense_galerkin():
    scene = _scene()
    config = _config()
    frequency = 35_000.0
    dense = DenseMQSTeacher(
        scene,
        frequency,
        config,
    )
    resistance, inductance, constraint, port_map = (
        dense.assemble()
    )
    operator = MatrixFreeMQSOperator(
        scene,
        frequency,
        config,
        chunk_size=37,
    )

    assert np.allclose(
        np.diag(
            resistance
        ),
        operator.resistance_diagonal,
        rtol=0,
        atol=1e-15,
    )
    assert np.allclose(
        constraint,
        operator.constraint_matrix,
        rtol=0,
        atol=1e-15,
    )
    assert np.allclose(
        port_map,
        operator.port_map,
        rtol=0,
        atol=0,
    )

    rng = np.random.default_rng(
        19
    )
    coefficients = (
        rng.normal(
            size=inductance.shape[0]
        )
        + 1j
        * rng.normal(
            size=inductance.shape[0]
        )
    )
    expected = (
        inductance
        @ coefficients
    )
    actual = (
        operator.apply_inductance(
            coefficients
        )
    )
    assert np.allclose(
        actual,
        expected,
        rtol=2e-11,
        atol=2e-13,
    )


def test_matrix_free_kkt_matches_dense_kkt_action():
    scene = _scene()
    config = _config()
    frequency = 42_000.0
    dense = DenseMQSTeacher(
        scene,
        frequency,
        config,
    )
    resistance, inductance, constraint, _ = (
        dense.assemble()
    )
    current_matrix = (
        resistance.astype(
            complex
        )
        + 1j
        * 2.0
        * np.pi
        * frequency
        * inductance
    )
    dense_kkt = np.block(
        [
            [
                current_matrix,
                -constraint.T.astype(
                    complex
                ),
            ],
            [
                constraint.astype(
                    complex
                ),
                np.zeros(
                    (
                        constraint.shape[0],
                        constraint.shape[0],
                    ),
                    dtype=complex,
                ),
            ],
        ]
    )

    operator = MatrixFreeMQSOperator(
        scene,
        frequency,
        config,
        chunk_size=29,
    )
    rng = np.random.default_rng(
        23
    )
    vector = (
        rng.normal(
            size=dense_kkt.shape[0]
        )
        + 1j
        * rng.normal(
            size=dense_kkt.shape[0]
        )
    )
    assert np.allclose(
        operator.apply_kkt(
            vector
        ),
        dense_kkt
        @ vector,
        rtol=2e-11,
        atol=2e-12,
    )
