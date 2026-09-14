import numpy as np

from sdfmpneo.unified_open_boundary import OpenBoundaryBackground


MATERIALS = {
    "tx_copper": {
        "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15,
        "relative_permeability": 1.0,
        "relative_permittivity": 1.0,
        "thermal_conductivity": 400.0,
        "volumetric_heat_capacity": 3.45e6,
    },
    "rx_copper": {
        "electrical_conductivity": 5.8e7,
        "resistivity_temperature_coefficient": 0.00393,
        "reference_temperature": 293.15,
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
        "outer_half_size": 0.015,
        "pitch": 0.002,
        "conductor_width": 0.0015,
        "conductor_thickness": 0.001,
        "corner_radius": 0.01,
        "translation": [0.0, 0.0, -0.025],
        "angles": [0.0, 0.0, 0.0],
    },
    "receiver": {
        "shape": "circle",
        "turns": 1.25,
        "outer_half_size": 0.015,
        "pitch": 0.002,
        "conductor_width": 0.0015,
        "conductor_thickness": 0.001,
        "corner_radius": 0.01,
        "translation": [0.0, 0.0, 0.025],
        "angles": [0.1, -0.05, 0.2],
    },
    "package_half_extent": [0.025, 0.025, 0.004],
}


def _background(step):
    axis = np.arange(-0.08, 0.0800001, step)
    if axis[-1] < 0.079:
        axis = np.r_[axis, 0.08]
    return OpenBoundaryBackground(
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


def test_source_polyline_and_wire_length_do_not_change_with_em_mesh():
    coarse = _background(0.02)
    fine = _background(0.016)
    c0 = coarse.geometry_context(GEOMETRY, assemble_thermal=False)
    c1 = fine.geometry_context(GEOMETRY, assemble_thermal=False)

    audit0 = tuple(c0.source_regularization)
    audit1 = tuple(c1.source_regularization)
    assert len(audit0) == len(audit1) == 2
    for a0, a1 in zip(audit0, audit1):
        assert a0["centerline_step"] == a1["centerline_step"] == 1.0e-3
        assert np.isclose(a0["path_length"], a1["path_length"], rtol=0.0, atol=1e-14)
        assert np.isclose(
            a0["terminal_separation"], a1["terminal_separation"], rtol=0.0, atol=1e-14
        )

    r0 = coarse.wire_resistances(c0, None)
    r1 = fine.wire_resistances(c1, None)
    assert np.allclose(r0, r1, rtol=1e-13, atol=1e-15)
