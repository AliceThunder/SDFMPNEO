import types

import numpy as np

import sdfmpneo.unified_longitudinal_patch_consistency as consistency


class _Geometry:
    def __init__(self, packages):
        self.packages = tuple(packages)

    @classmethod
    def from_mapping(cls, value):
        return value


class _PatchBackground:
    def __init__(
        self,
        x,
        y,
        z,
        *,
        frequency_hz,
        materials,
        coil_materials,
        package_materials,
        seawater_material,
        ambient_temperature,
    ):
        self.x = np.asarray(x, float)
        self.y = np.asarray(y, float)
        self.z = np.asarray(z, float)
        self.frequency_hz = frequency_hz
        self.materials = materials
        self.coil_materials = tuple(coil_materials)
        self.package_materials = tuple(package_materials)
        self.seawater_material = seawater_material
        self.ambient_temperature = ambient_temperature


def _package(center, half):
    return types.SimpleNamespace(
        half_extent=np.asarray(half, float),
        pose=types.SimpleNamespace(
            rotation=np.eye(3),
            translation=np.asarray(center, float),
        ),
    )


def test_full_geometry_patch_bounds_cover_every_package():
    geometry = _Geometry(
        [
            _package([-0.25, 0.0, 0.0], [0.08, 0.05, 0.02]),
            _package([0.30, 0.10, -0.05], [0.07, 0.04, 0.03]),
        ]
    )
    parent = types.SimpleNamespace(
        x=np.linspace(-0.6, 0.6, 61),
        y=np.linspace(-0.6, 0.6, 61),
        z=np.linspace(-0.6, 0.6, 61),
    )
    module = types.SimpleNamespace(
        UnifiedUWPTGeometry=_Geometry,
        _select_axis=lambda axis, lo, hi: np.asarray(
            axis[
                max(0, np.searchsorted(axis, lo, side="right") - 1):
                min(len(axis), np.searchsorted(axis, hi, side="left") + 1)
            ],
            float,
        ),
    )
    axes, center, aabb_half, returned = consistency._full_geometry_patch_axes(
        module,
        parent,
        geometry,
        0,
        {"boundary_padding": 0.02},
    )
    assert returned is geometry
    assert np.allclose(center, [-0.25, 0.0, 0.0])
    assert np.allclose(aabb_half, [0.08, 0.05, 0.02])
    for package in geometry.packages:
        lo = package.pose.translation - package.half_extent
        hi = package.pose.translation + package.half_extent
        assert axes[0][0] <= lo[0] and axes[0][-1] >= hi[0]
        assert axes[1][0] <= lo[1] and axes[1][-1] >= hi[1]
        assert axes[2][0] <= lo[2] and axes[2][-1] >= hi[2]


def test_full_patch_background_keeps_all_parent_material_channels():
    parent = types.SimpleNamespace(
        __class__=_PatchBackground,
        frequency_hz=100e3,
        materials={"tx": {}, "rx": {}, "ptx": {}, "prx": {}, "water": {}},
        coil_materials=("tx", "rx"),
        package_materials=("ptx", "prx"),
        seawater_material="water",
        ambient_temperature=293.15,
    )
    # SimpleNamespace cannot override the actual __class__ used by constructor
    # dispatch, so use a tiny real parent instance of the desired class.
    parent = _PatchBackground(
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        frequency_hz=100e3,
        materials={"tx": {}, "rx": {}, "ptx": {}, "prx": {}, "water": {}},
        coil_materials=("tx", "rx"),
        package_materials=("ptx", "prx"),
        seawater_material="water",
        ambient_temperature=293.15,
    )
    patch = consistency._make_full_patch_background(
        None,
        parent,
        (
            np.array([-0.5, 0.0, 0.5]),
            np.array([-0.5, 0.0, 0.5]),
            np.array([-0.5, 0.0, 0.5]),
        ),
        fine_step=0.012,
    )
    assert patch.coil_materials == parent.coil_materials
    assert patch.package_materials == parent.package_materials
    assert patch.seawater_material == parent.seawater_material
