import types

import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_background import EPS0
from sdfmpneo.unified_resolved_admittance_hodge import _build_permittivity_weights


def _background():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0, 2.0])
    cell_axes = tuple((axis[:-1] + axis[1:]) * 0.5 for axis in (x, y, z))
    edge_cell_hodge = sp.csr_matrix(
        (
            np.full(4, 0.25),
            np.array([0, 1, 2, 3]),
            np.array([0, 4]),
        ),
        shape=(1, 8),
    )
    bg = types.SimpleNamespace(
        x=x,
        y=y,
        z=z,
        nx=2,
        ny=2,
        nz=2,
        n_cells=8,
        cell_axes=cell_axes,
        edge_tuples=((0, 0, 1, 1),),
        edge_lengths=np.array([1.0]),
        edge_cell_hodge=edge_cell_hodge,
        n_edges=1,
        package_materials=("pkg",),
        coil_materials=(),
        seawater_material="sea",
        materials={
            "sea": {"relative_permittivity": 80.0},
            "pkg": {"relative_permittivity": 3.0},
        },
    )

    def cell_properties(context, state, em=False):
        del state, em
        eps = np.zeros(bg.n_cells, float)
        for name, fraction in context.fractions.items():
            eps += np.asarray(fraction, float) * EPS0 * float(
                bg.materials[name]["relative_permittivity"]
            )
        zeros = np.zeros(bg.n_cells, float)
        return zeros, eps, zeros, zeros, zeros, zeros

    bg.cell_properties = cell_properties
    return bg


def _package():
    class _Pose:
        @staticmethod
        def apply(points):
            return np.asarray(points, float) + np.array([0.5, 1.0, 1.0])

        @staticmethod
        def inverse(points):
            return np.asarray(points, float) - np.array([0.5, 1.0, 1.0])

    return types.SimpleNamespace(
        half_extent=np.array([0.5, 1.0, 1.0]),
        pose=_Pose(),
    )


def test_resolved_dielectric_hodge_recovers_pure_seawater():
    bg = _background()
    bg.package_materials = ()
    context = types.SimpleNamespace(
        geometry=types.SimpleNamespace(packages=()),
        fractions={"sea": np.ones(bg.n_cells)},
    )
    exact, legacy, meta = _build_permittivity_weights(bg, context)
    expected = np.array([80.0 * EPS0])
    assert np.allclose(exact, expected)
    assert np.allclose(legacy, expected)
    assert meta["dielectric_hodge_legacy_relative_difference"] <= 1e-14


def test_resolved_dielectric_hodge_replaces_seawater_by_package_epsilon():
    bg = _background()
    package = _package()
    package_fraction = np.zeros(bg.n_cells)
    package_fraction[:4] = 1.0
    sea_fraction = 1.0 - package_fraction
    context = types.SimpleNamespace(
        geometry=types.SimpleNamespace(packages=(package,)),
        fractions={"pkg": package_fraction, "sea": sea_fraction},
    )
    exact, legacy, _meta = _build_permittivity_weights(bg, context)
    expected = np.array([3.0 * EPS0])
    assert np.allclose(exact, expected, rtol=1e-12, atol=1e-24)
    assert np.allclose(legacy, expected, rtol=1e-12, atol=1e-24)
