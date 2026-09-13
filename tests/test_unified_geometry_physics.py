import numpy as np
import pytest

from sdfmpneo.unified_background import FixedMultiscaleBackground


MATERIALS = {
    "tx_copper": dict(
        electrical_conductivity=5.8e7,
        resistivity_temperature_coefficient=0.00393,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=1.0,
        thermal_conductivity=400.0,
        volumetric_heat_capacity=3.45e6,
    ),
    "rx_copper": dict(
        electrical_conductivity=5.8e7,
        resistivity_temperature_coefficient=0.00393,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=1.0,
        thermal_conductivity=400.0,
        volumetric_heat_capacity=3.45e6,
    ),
    "tx_package": dict(
        electrical_conductivity=0.0,
        resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=3.0,
        thermal_conductivity=0.2,
        volumetric_heat_capacity=1.5e6,
    ),
    "rx_package": dict(
        electrical_conductivity=0.0,
        resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=3.0,
        thermal_conductivity=0.2,
        volumetric_heat_capacity=1.5e6,
    ),
    "seawater": dict(
        electrical_conductivity=5.0,
        resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15,
        relative_permeability=1.0,
        relative_permittivity=80.0,
        thermal_conductivity=0.6,
        volumetric_heat_capacity=4.1e6,
    ),
}


def background():
    axis = np.linspace(-0.05, 0.05, 7)
    return FixedMultiscaleBackground(
        axis,
        axis,
        axis,
        frequency_hz=100000.0,
        materials=MATERIALS,
        coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"),
        seawater_material="seawater",
        ambient_temperature=293.15,
    )


def geometry(rx_x=0.0, rx_z=0.015, package_half=(0.018, 0.018, 0.004)):
    coil = dict(
        shape="circle",
        turns=0.5,
        outer_half_size=0.012,
        pitch=0.002,
        conductor_width=0.001,
        conductor_thickness=0.001,
        corner_radius=0.006,
        angles=[0.0, 0.0, 0.0],
    )
    return {
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.015]),
        "receiver": dict(coil, translation=[rx_x, 0.0, rx_z]),
        "package_half_extent": list(package_half),
    }


def test_geometry_rejects_package_overlap_and_coil_outside_package():
    bg = background()
    bg.validate_geometry(geometry())

    with pytest.raises(ValueError, match="overlap"):
        bg.validate_geometry(geometry(rx_z=-0.010))

    with pytest.raises(ValueError, match="contained"):
        bg.validate_geometry(geometry(package_half=(0.010, 0.010, 0.004)))


def test_small_geometry_translation_changes_source_continuously_not_by_cell_jump():
    bg = background()
    first = bg.geometry_context(geometry(rx_x=0.0), assemble_thermal=False)
    second = bg.geometry_context(geometry(rx_x=1e-4), assemble_thermal=False)

    a = first.source_shape[:, 1]
    b = second.source_shape[:, 1]
    relative = np.linalg.norm(b - a) / max(np.linalg.norm(a), np.finfo(float).tiny)
    assert relative > 1e-8
    assert relative < 0.05

    # Line heat remains a normalized conservative spatial distribution.
    assert np.isclose(np.sum(first.line_heat_weights[1]), 1.0)
    assert np.isclose(np.sum(second.line_heat_weights[1]), 1.0)
    heat_change = np.linalg.norm(second.line_heat_weights[1] - first.line_heat_weights[1])
    assert 0.0 < heat_change < 0.1
