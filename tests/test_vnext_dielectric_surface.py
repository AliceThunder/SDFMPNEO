import numpy as np

from sdfmpneo_vnext import (
    DielectricSurfaceSolver,
    HomogeneousMedium,
    IsotropicMaterial,
    PackageObject,
    RigidPose,
    SuperquadricPackageGeometry,
    haar_rotation,
)
from sdfmpneo_vnext.scene import EPS0


def _sphere(
    radius=0.02,
    epsilon_r=4.0,
    pose=None,
):
    return PackageObject(
        SuperquadricPackageGeometry(
            np.array(
                [radius, radius, radius]
            ),
            exponent_xy=2.0,
            exponent_z=2.0,
            pose=(
                pose
                or RigidPose.identity()
            ),
        ),
        IsotropicMaterial(
            relative_permittivity=epsilon_r,
        ),
        "sphere",
    )


def test_dielectric_sphere_uniform_field_matches_analytic_polarizability():
    radius = 0.02
    epsilon_r = 4.0
    background = HomogeneousMedium(
        relative_permittivity=1.0,
    )
    solver = DielectricSurfaceSolver(
        (_sphere(radius, epsilon_r),),
        background,
        85_000.0,
        vertical_order=16,
        azimuthal_order=32,
    )
    electric_field = np.array(
        [0.0, 0.0, 1.0],
        dtype=complex,
    )
    result = solver.solve_uniform_field(
        electric_field
    )
    expected = (
        4.0
        * np.pi
        * EPS0
        * radius**3
        * (
            epsilon_r - 1.0
        )
        / (
            epsilon_r + 2.0
        )
        * electric_field
    )
    dipole = result.induced_dipole()
    assert result.normalized_residual < 1e-11
    assert np.allclose(
        dipole,
        expected,
        rtol=0.04,
        atol=2e-19,
    )
    charge_scale = (
        EPS0
        * radius**2
        * np.linalg.norm(
            electric_field
        )
    )
    assert (
        abs(
            result.net_equivalent_charge
        )
        < 1e-10
        * charge_scale
    )


def test_dielectric_surface_solver_zeroes_invisible_interface():
    package = _sphere(
        epsilon_r=2.5,
    )
    background = HomogeneousMedium(
        relative_permittivity=2.5,
    )
    result = DielectricSurfaceSolver(
        (package,),
        background,
        50_000.0,
        vertical_order=10,
        azimuthal_order=20,
    ).solve_uniform_field(
        np.array(
            [0.3, -0.4, 0.2]
        )
    )
    assert np.allclose(
        result.equivalent_density,
        0.0,
        rtol=0,
        atol=0,
    )
    assert np.allclose(
        result.induced_dipole(),
        0.0,
        rtol=0,
        atol=0,
    )


def test_dielectric_surface_response_is_common_se3_equivariant():
    geometry = SuperquadricPackageGeometry(
        np.array(
            [0.03, 0.022, 0.016]
        ),
        exponent_xy=2.0,
        exponent_z=2.0,
    )
    material = IsotropicMaterial(
        relative_permittivity=3.2,
    )
    package = PackageObject(
        geometry,
        material,
        "body",
    )
    background = HomogeneousMedium()
    field = np.array(
        [0.7, -0.2, 0.4],
        dtype=complex,
    )
    reference = DielectricSurfaceSolver(
        (package,),
        background,
        70_000.0,
        vertical_order=12,
        azimuthal_order=24,
    ).solve_uniform_field(
        field
    )

    rng = np.random.default_rng(
        113
    )
    pose = RigidPose(
        haar_rotation(rng),
        np.array(
            [0.2, -0.15, 0.35]
        ),
    )
    moved_package = PackageObject(
        geometry.transformed(
            pose
        ),
        material,
        "body",
    )
    moved_field = (
        field
        @ pose.rotation.T
    )
    moved = DielectricSurfaceSolver(
        (moved_package,),
        background,
        70_000.0,
        vertical_order=12,
        azimuthal_order=24,
    ).solve_uniform_field(
        moved_field
    )

    expected_dipole = (
        reference.induced_dipole()
        @ pose.rotation.T
    )
    actual_dipole = (
        moved.induced_dipole(
            origin=pose.translation
        )
    )
    assert np.allclose(
        actual_dipole,
        expected_dipole,
        rtol=2e-10,
        atol=2e-19,
    )
    assert np.allclose(
        moved.equivalent_density,
        reference.equivalent_density,
        rtol=3e-10,
        atol=3e-12,
    )



def test_dielectric_surface_multi_rhs_matches_independent_solves():
    solver = DielectricSurfaceSolver(
        (_sphere(0.018, 3.5),),
        HomogeneousMedium(),
        90_000.0,
        vertical_order=10,
        azimuthal_order=20,
    )
    field_a = np.array(
        [0.6, -0.2, 0.1],
        dtype=complex,
    )
    field_b = np.array(
        [-0.1, 0.4, 0.5],
        dtype=complex,
    )
    rhs = np.stack(
        (
            -solver.normals @ field_a,
            -solver.normals @ field_b,
        ),
        axis=1,
    )
    density, residual = (
        solver.solve_density_matrix(
            rhs
        )
    )
    reference_a = (
        solver.solve_uniform_field(
            field_a
        )
    )
    reference_b = (
        solver.solve_uniform_field(
            field_b
        )
    )
    assert density.shape == (
        len(solver.weights),
        2,
    )
    assert np.all(
        residual < 1e-11
    )
    assert np.allclose(
        density[:, 0],
        reference_a.equivalent_density,
        rtol=2e-12,
        atol=2e-13,
    )
    assert np.allclose(
        density[:, 1],
        reference_b.equivalent_density,
        rtol=2e-12,
        atol=2e-13,
    )
