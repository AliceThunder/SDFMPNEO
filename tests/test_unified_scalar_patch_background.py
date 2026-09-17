import numpy as np

import sdfmpneo
from sdfmpneo import unified_global_longitudinal_reference as longitudinal
from sdfmpneo import unified_longitudinal_patch_consistency as consistency
from sdfmpneo.unified_gradient_block_maxwell import gradient_operator
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground


def _background():
    materials = {
        "c": {"electrical_conductivity": 1.0, "relative_permittivity": 1.0, "relative_permeability": 1.0,
              "thermal_conductivity": 1.0, "volumetric_heat_capacity": 1.0},
        "p": {"electrical_conductivity": 0.0, "relative_permittivity": 2.0, "relative_permeability": 1.0,
              "thermal_conductivity": 1.0, "volumetric_heat_capacity": 1.0},
        "s": {"electrical_conductivity": 5.0, "relative_permittivity": 80.0, "relative_permeability": 1.0,
              "thermal_conductivity": 1.0, "volumetric_heat_capacity": 1.0},
    }
    axis = np.linspace(-0.03, 0.03, 7)
    bg = OpenBoundaryBackground(
        axis, axis, axis,
        frequency_hz=1e5,
        materials=materials,
        coil_materials=("c",),
        package_materials=("p",),
        seawater_material="s",
    )
    bg.background_config = {"fine_step": 0.01, "self_correction": {"enabled": False}}
    bg.self_correction_config = {"enabled": False}
    return bg


def test_refined_scalar_patch_skips_maxwell_only_topology():
    parent = _background()
    coarse_axes = (parent.x[1:-1], parent.y[1:-1], parent.z[1:-1])
    refined_axes = tuple(np.sort(np.unique(np.r_[axis, 0.5 * (axis[:-1] + axis[1:])])) for axis in coarse_axes)
    patch = consistency._make_full_patch_background(
        longitudinal, parent, refined_axes, fine_step=0.005
    )
    assert patch._sdfmpneo_scalar_only_topology is True
    assert not hasattr(patch, "curl")
    assert not hasattr(patch, "face_cell_hodge")
    assert not hasattr(patch, "reconstruct")
    assert patch.edge_cell_hodge.shape == (patch.n_edges, patch.n_cells)
    G = gradient_operator(patch, gauge_fixed=False)
    assert G.shape[0] == patch.n_edges
    assert G.shape[1] == (patch.nx + 1) * (patch.ny + 1) * (patch.nz + 1)


def test_coarse_scalar_patch_keeps_full_topology():
    parent = _background()
    axes = (parent.x[1:-1], parent.y[1:-1], parent.z[1:-1])
    patch = consistency._make_full_patch_background(
        longitudinal, parent, axes, fine_step=0.01
    )
    assert hasattr(patch, "curl")
    assert hasattr(patch, "face_cell_hodge")
    assert hasattr(patch, "reconstruct")
