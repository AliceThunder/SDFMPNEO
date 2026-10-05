import numpy as np

from sdfmpneo.unified_geometry import PackageGeometry, Pose
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground


MATERIALS = {
    "wire": {
        "electrical_conductivity": 5.8e7,
        "relative_permittivity": 1.0,
        "relative_permeability": 1.0,
        "thermal_conductivity": 1.0,
        "volumetric_heat_capacity": 1.0,
    },
    "package": {
        "electrical_conductivity": 0.0,
        "relative_permittivity": 3.0,
        "relative_permeability": 1.0,
        "thermal_conductivity": 1.0,
        "volumetric_heat_capacity": 1.0,
    },
    "sea": {
        "electrical_conductivity": 5.0,
        "relative_permittivity": 80.0,
        "relative_permeability": 1.0,
        "thermal_conductivity": 1.0,
        "volumetric_heat_capacity": 1.0,
    },
}


def _axis(step):
    values = np.arange(-0.04, 0.0400001, float(step))
    if values[-1] < 0.039999:
        values = np.r_[values, 0.04]
    return values


def _background(step):
    axis = _axis(step)
    return OpenBoundaryBackground(
        axis,
        axis,
        axis,
        frequency_hz=1.0e5,
        materials=MATERIALS,
        coil_materials=("wire",),
        package_materials=("package",),
        seawater_material="sea",
        ambient_temperature=293.15,
    )


def test_rotated_package_volume_is_mesh_and_phase_invariant():
    package = PackageGeometry(
        np.array([0.013, 0.009, 0.0035]),
        Pose(np.array([0.0031, -0.0027, 0.0043]), np.array([0.31, -0.23, 0.41])),
    )
    exact = float(8.0 * np.prod(package.half_extent))
    volumes = []
    for step in (0.012, 0.009):
        bg = _background(step)
        fraction = np.asarray(bg._package_fraction(package), float)
        assert bg.package_fraction_model == "exact_obb_cartesian_cell_intersection_v1"
        assert np.min(fraction) >= 0.0
        assert np.max(fraction) <= 1.0
        represented = float(np.dot(fraction, bg.cell_volumes))
        volumes.append(represented)
        assert np.isclose(represented, exact, rtol=2e-9, atol=1e-14)
    assert np.isclose(volumes[0], volumes[1], rtol=2e-9, atol=1e-14)
