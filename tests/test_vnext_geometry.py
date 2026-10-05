import numpy as np

from sdfmpneo_vnext import (
    RigidPose,
    SuperellipseSpiral,
    polynomial_section_basis,
    superellipse_area,
    haar_rotation,
)


def test_superellipse_area_matches_quadrature():
    coil = SuperellipseSpiral(
        0.04,
        0.03,
        1.25,
        0.003,
        0.002,
        exponent=4.0,
        conductor_width=0.003,
        conductor_thickness=0.0015,
        cross_section_exponent=4.0,
    )
    basis = polynomial_section_basis(
        coil.conductor_width,
        coil.conductor_thickness,
        coil.cross_section_exponent,
        degree=2,
        radial_order=8,
        angular_order=96,
    )
    exact = superellipse_area(
        coil.conductor_width,
        coil.conductor_thickness,
        coil.cross_section_exponent,
    )
    assert np.isclose(basis.area, exact, rtol=3e-4)
    assert np.isclose(coil.cross_section_area, exact, rtol=1e-14)


def test_bishop_frames_and_common_pose():
    base = SuperellipseSpiral(
        0.04,
        0.03,
        1.2,
        0.002,
        0.002,
        exponent=3.5,
    )
    poly = base.polyline(40)
    assert np.allclose(
        np.sum(poly.tangents * poly.normal1, axis=1),
        0.0,
        atol=1e-10,
    )
    assert np.allclose(
        np.sum(poly.tangents * poly.normal2, axis=1),
        0.0,
        atol=1e-10,
    )
    assert np.allclose(
        np.sum(poly.normal1 * poly.normal2, axis=1),
        0.0,
        atol=1e-10,
    )
    rng = np.random.default_rng(5)
    pose = RigidPose(
        haar_rotation(rng),
        np.array([0.2, -0.1, 0.3]),
    )
    moved = base.transformed(pose).sample_centerline(91)
    expected = pose.apply(base.sample_centerline(91))
    assert np.allclose(
        moved,
        expected,
        rtol=0,
        atol=3e-12,
    )


def test_section_basis_is_orthonormal():
    basis = polynomial_section_basis(
        0.004,
        0.0015,
        3.0,
        degree=2,
        radial_order=8,
        angular_order=96,
    )
    gram = basis.values.T @ (
        basis.quadrature.weights[:, None] * basis.values
    )
    assert np.allclose(
        gram,
        np.eye(basis.n_modes),
        atol=2e-11,
    )
    assert abs(basis.moments[0]) > 0
