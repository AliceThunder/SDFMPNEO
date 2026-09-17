import types

import numpy as np

from sdfmpneo.unified_fast_terminal_dissipative_scalar import (
    _parent_cell_ids,
    _prolong_parent_cell_values,
)


def _background(x, y, z):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    return types.SimpleNamespace(
        x=x,
        y=y,
        z=z,
        nx=len(x) - 1,
        ny=len(y) - 1,
        nz=len(z) - 1,
        n_cells=(len(x) - 1) * (len(y) - 1) * (len(z) - 1),
        cell_axes=(
            0.5 * (x[:-1] + x[1:]),
            0.5 * (y[:-1] + y[1:]),
            0.5 * (z[:-1] + z[1:]),
        ),
    )


def test_refined_cells_inherit_containing_parent_material_exactly():
    parent = _background([0.0, 1.0, 3.0], [0.0, 2.0, 5.0], [0.0, 4.0, 7.0])
    patch = _background(
        [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0],
    )
    values = np.arange(parent.n_cells, dtype=float) + 10.0
    ids = _parent_cell_ids(parent, patch)
    refined = _prolong_parent_cell_values(parent, patch, values)

    assert ids.shape == (patch.n_cells,)
    assert refined.shape == (patch.n_cells,)
    np.testing.assert_array_equal(refined, values[ids])

    # Every refined cell inside the first parent x/y/z cell keeps value[0].
    nx0 = 3  # x centers below 1
    ny0 = 3  # y centers below 2
    nz0 = 3  # z centers below 4
    cube = refined.reshape(patch.nx, patch.ny, patch.nz)
    np.testing.assert_array_equal(cube[:nx0, :ny0, :nz0], np.full((nx0, ny0, nz0), values[0]))
