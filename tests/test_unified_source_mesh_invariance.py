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
        "outer_half_size": 0.012,
        "pitch": 0.004,
        "conductor_width": 0.003,
        "conductor_thickness": 0.001,
        "corner_radius": 0.008,
        "translation": [0.0, 0.0, -0.015],
        "angles": [0.0, 0.0, 0.0],
    },
    "receiver": {
        "shape": "circle",
        "turns": 1.25,
        "outer_half_size": 0.012,
        "pitch": 0.004,
        "conductor_width": 0.003,
        "conductor_thickness": 0.001,
        "corner_radius": 0.008,
        "translation": [0.0, 0.0, 0.015],
        "angles": [0.1, -0.05, 0.2],
    },
    "package_half_extent": [0.020, 0.020, 0.004],
}


def _background(step):
    axis = np.arange(-0.035, 0.0350001, step)
    if axis[-1] < 0.034:
        axis = np.r_[axis, 0.035]
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


def test_source_support_is_fixed_while_quadrature_resolves_with_mesh():
    coarse = _background(0.004)
    fine = _background(0.002)
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
        assert a0["terminal_regularization_mesh_independent"] is True
        assert a1["terminal_regularization_mesh_independent"] is True
        assert a0["cross_section_support_mesh_independent"] is True
        assert a1["cross_section_support_mesh_independent"] is True
        assert a0["terminal_charge_support_mesh_independent"] is True
        assert a1["terminal_charge_support_mesh_independent"] is True
        assert a0["terminal_profile"] == a1["terminal_profile"]
        assert a0["terminal_charge_model"] == a1["terminal_charge_model"]
        assert a0["terminal_charge_lift_model"] == a1["terminal_charge_lift_model"]
        assert np.isclose(
            a0["terminal_contact_length"], a1["terminal_contact_length"], rtol=0.0, atol=1e-15
        )
        assert np.isclose(
            a0["terminal_charge_contact_length"],
            a1["terminal_charge_contact_length"],
            rtol=0.0,
            atol=1e-15,
        )
        # Linear coordinate moments of a trilinear nodal deposition are exact;
        # therefore the declared regularized source vector must be mesh invariant
        # even though the nodal support itself is allowed to depend on grid phase.
        assert np.allclose(
            np.asarray(a0["regularized_source_vector"], float),
            np.asarray(a1["regularized_source_vector"], float),
            rtol=0.0,
            atol=2e-12,
        )
        assert np.allclose(
            np.asarray(a0["terminal_charge_vector"], float),
            np.asarray(a1["terminal_charge_vector"], float),
            rtol=0.0,
            atol=2e-12,
        )
        assert a0["cross_section_quadrature"] == "composite_gauss3"
        assert a1["cross_section_quadrature"] == "composite_gauss3"
        assert a0["cross_section_width_panels"] >= 2
        assert a0["cross_section_thickness_panels"] >= 2
        assert a1["cross_section_width_panels"] > a0["cross_section_width_panels"]
        assert a1["cross_section_quadrature_points"] > a0["cross_section_quadrature_points"]
        assert a1["source_quadrature_resolution"] < a0["source_quadrature_resolution"]
        assert a0["terminal_charge_support_nodes"] > 1
        assert a1["terminal_charge_support_nodes"] >= a0["terminal_charge_support_nodes"]
        assert a0["terminal_charge_target_relative_error"] <= 5e-11
        assert a1["terminal_charge_target_relative_error"] <= 5e-11
        assert a0["terminal_charge_lift_relative_curl"] <= 1e-12
        assert a1["terminal_charge_lift_relative_curl"] <= 1e-12
        assert a0["terminal_path_integral_relative_error"] <= 1e-12
        assert a1["terminal_path_integral_relative_error"] <= 1e-12

    r0 = coarse.wire_resistances(c0, None)
    r1 = fine.wire_resistances(c1, None)
    assert np.allclose(r0, r1, rtol=1e-13, atol=1e-15)
