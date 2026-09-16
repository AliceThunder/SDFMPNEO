import numpy as np

from sdfmpneo.unified_open_boundary import OpenBoundaryBackground
from sdfmpneo import unified_global_longitudinal_reference as longitudinal
from sdfmpneo.unified_terminal_longitudinal_refinement import (
    _base_axis_steps,
    _refined_axes_for_request,
    _subdivide_axis,
)


MATERIALS = {
    "tx_copper": {
        "electrical_conductivity": 5.8e7,
        "relative_permeability": 1.0,
        "relative_permittivity": 1.0,
        "thermal_conductivity": 400.0,
        "volumetric_heat_capacity": 3.45e6,
    },
    "rx_copper": {
        "electrical_conductivity": 5.8e7,
        "relative_permeability": 1.0,
        "relative_permittivity": 1.0,
        "thermal_conductivity": 400.0,
        "volumetric_heat_capacity": 3.45e6,
    },
    "tx_package": {
        "electrical_conductivity": 0.0,
        "relative_permeability": 1.0,
        "relative_permittivity": 3.0,
        "thermal_conductivity": 0.2,
        "volumetric_heat_capacity": 1.5e6,
    },
    "rx_package": {
        "electrical_conductivity": 0.0,
        "relative_permeability": 1.0,
        "relative_permittivity": 3.0,
        "thermal_conductivity": 0.2,
        "volumetric_heat_capacity": 1.5e6,
    },
    "seawater": {
        "electrical_conductivity": 5.0,
        "relative_permeability": 1.0,
        "relative_permittivity": 80.0,
        "thermal_conductivity": 0.6,
        "volumetric_heat_capacity": 4.1e6,
    },
}


GEOMETRY = {
    "transmitter": {
        "shape": "circle",
        "turns": 1.25,
        "outer_half_size": 0.012,
        "pitch": 0.004,
        "conductor_width": 0.002,
        "conductor_thickness": 0.0006,
        "corner_radius": 0.008,
        "translation": [0.0, 0.0, -0.012],
        "angles": [0.1, -0.15, 0.2],
    },
    "receiver": {
        "shape": "circle",
        "turns": 1.25,
        "outer_half_size": 0.012,
        "pitch": 0.004,
        "conductor_width": 0.0025,
        "conductor_thickness": 0.0005,
        "corner_radius": 0.008,
        "translation": [0.0, 0.0, 0.012],
        "angles": [-0.1, 0.08, -0.2],
    },
    "package_half_extent": [0.020, 0.020, 0.004],
}


def _background():
    axis = np.arange(-0.05, 0.0500001, 0.01)
    bg = OpenBoundaryBackground(
        axis,
        axis,
        axis,
        frequency_hz=1.0e5,
        materials=MATERIALS,
        coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"),
        seawater_material="seawater",
        ambient_temperature=293.15,
    )
    bg.background_config = {
        "fine_step": 0.012,
        "mesh_check": {"relative_tolerance": 0.1},
        "self_correction": {"fine_step": 0.003, "validation_fine_step": 0.00225},
        "global_longitudinal_correction": {
            "fine_step": 0.003,
            "validation_fine_step": 0.00225,
            "terminal_cells_per_support": 3.0,
            "terminal_core_padding_factor": 1.5,
            "terminal_patch_max_cells": 600000,
        },
    }
    return bg


def test_axis_refinement_is_nested_and_local():
    coarse = np.array([0.0, 0.01, 0.02, 0.03])
    refined = _subdivide_axis(coarse, ((0.012, 0.016),), 0.001)
    for value in coarse:
        assert np.min(np.abs(refined - value)) <= 1e-14
    local = refined[(refined >= 0.012 - 1e-14) & (refined <= 0.016 + 1e-14)]
    assert local.size >= 5
    assert np.max(np.diff(local)) <= 0.001 + 1e-14
    # No unrelated 10-mm coarse interval is globally refined.
    assert not np.any((refined > 0.0200001) & (refined < 0.0299999))


def test_terminal_steps_follow_physical_support_and_validation_refines_them():
    bg = _background()
    base_steps, intervals, boxes, contact = _base_axis_steps(
        longitudinal, bg, GEOMETRY, 0, 0.003
    )
    assert contact > 0.0
    assert len(boxes) == 2
    assert len(intervals) == 3
    assert np.all(base_steps > 0.0)
    assert np.all(base_steps <= 0.003)
    # The 0.6-mm conductor thickness forces at least one axis well below 3 mm.
    assert float(np.min(base_steps)) < 0.001

    coarse_axes = (bg.x, bg.y, bg.z)
    fine_axes, fine_steps, _, _ = _refined_axes_for_request(
        longitudinal, bg, GEOMETRY, 0, coarse_axes, 0.003, 0.003
    )
    validation_axes, validation_steps, _, _ = _refined_axes_for_request(
        longitudinal, bg, GEOMETRY, 0, coarse_axes, 0.00225, 0.003
    )
    assert np.allclose(validation_steps, 0.75 * fine_steps, rtol=0.0, atol=1e-15)
    for coarse, fine, validation in zip(coarse_axes, fine_axes, validation_axes):
        for value in coarse:
            assert np.min(np.abs(fine - value)) <= 1e-14
            assert np.min(np.abs(validation - value)) <= 1e-14
        assert len(validation) >= len(fine) >= len(coarse)


def test_installed_model_uses_reactive_longitudinal_v4():
    assert longitudinal._MODEL == "global_boundary_conditioned_longitudinal_reactive_defect_v4"
    assert bool(getattr(longitudinal, "_terminal_longitudinal_refinement_installed", False))
    assert bool(getattr(longitudinal, "_reactive_longitudinal_reference_installed", False))
