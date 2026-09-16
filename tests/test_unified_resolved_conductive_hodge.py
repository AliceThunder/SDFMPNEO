import types

import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_resolved_conductive_hodge import _build_conductivity_hodge


class _IdentityPose:
    @staticmethod
    def apply(points):
        return np.asarray(points, float)

    @staticmethod
    def inverse(points):
        return np.asarray(points, float)


def _background():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0, 2.0])
    nx = ny = nz = 2
    cell_axes = tuple((axis[:-1] + axis[1:]) * 0.5 for axis in (x, y, z))
    # One x-oriented edge spanning x=[0,1] at y=z=1.  Its four adjacent
    # quarter-cell dual wedges each have volume 1/4 and length 1.
    edge_cell_hodge = sp.csr_matrix(
        (
            np.full(4, 0.25),
            np.array([0, 1, 2, 3]),
            np.array([0, 4]),
        ),
        shape=(1, 8),
    )
    return types.SimpleNamespace(
        x=x,
        y=y,
        z=z,
        nx=nx,
        ny=ny,
        nz=nz,
        cell_axes=cell_axes,
        edge_tuples=((0, 0, 1, 1),),
        edge_lengths=np.array([1.0]),
        edge_cell_hodge=edge_cell_hodge,
        n_edges=1,
        package_materials=("pkg",),
        seawater_material="sea",
        ambient_temperature=293.15,
        _temperature_material=lambda material, temperature: 0.0 if material == "pkg" else 5.0,
    )


def test_exact_dual_hodge_recovers_pure_seawater_without_packages():
    bg = _background()
    bg.package_materials = ()
    context = types.SimpleNamespace(geometry=types.SimpleNamespace(packages=()))
    matrix, weights = _build_conductivity_hodge(bg, context)
    assert np.allclose(matrix.toarray(), 5.0 * bg.edge_cell_hodge.toarray())
    assert np.allclose(weights, [5.0])


def test_exact_dual_hodge_removes_fully_insulating_package_volume():
    bg = _background()
    package = types.SimpleNamespace(
        half_extent=np.array([0.5, 1.0, 1.0]),
        pose=_IdentityPose(),
    )
    # Center the package at x=0.5, y=z=1 via a tiny pose wrapper.
    class _Pose:
        @staticmethod
        def apply(points):
            return np.asarray(points, float) + np.array([0.5, 1.0, 1.0])

        @staticmethod
        def inverse(points):
            return np.asarray(points, float) - np.array([0.5, 1.0, 1.0])

    package.pose = _Pose()
    context = types.SimpleNamespace(geometry=types.SimpleNamespace(packages=(package,)))
    matrix, weights = _build_conductivity_hodge(bg, context)
    assert np.allclose(matrix.toarray(), 0.0, atol=1e-14)
    assert np.allclose(weights, 0.0, atol=1e-14)
