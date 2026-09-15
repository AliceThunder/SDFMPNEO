import numpy as np

from sdfmpneo.unified_charge_regularized_source import terminal_charge_target
from sdfmpneo.unified_gradient_block_maxwell import (
    gradient_operator,
    source_terminal_divergence,
)
from sdfmpneo.unified_open_boundary import OpenBoundaryBackground


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
        relative_permeability=1.0,
        relative_permittivity=3.0,
        thermal_conductivity=0.2,
        volumetric_heat_capacity=1.5e6,
    ),
    "rx_package": dict(
        electrical_conductivity=0.0,
        relative_permeability=1.0,
        relative_permittivity=3.0,
        thermal_conductivity=0.2,
        volumetric_heat_capacity=1.5e6,
    ),
    "seawater": dict(
        electrical_conductivity=5.0,
        relative_permeability=1.0,
        relative_permittivity=80.0,
        thermal_conductivity=0.6,
        volumetric_heat_capacity=4.1e6,
    ),
}


GEOMETRY = {
    "transmitter": {
        "shape": "circle",
        "turns": 0.75,
        "outer_half_size": 0.012,
        "pitch": 0.003,
        "conductor_width": 0.002,
        "conductor_thickness": 0.0008,
        "corner_radius": 0.008,
        "translation": [0.0, 0.0, -0.012],
        "angles": [0.0, 0.0, 0.0],
    },
    "receiver": {
        "shape": "circle",
        "turns": 0.75,
        "outer_half_size": 0.012,
        "pitch": 0.003,
        "conductor_width": 0.002,
        "conductor_thickness": 0.0008,
        "corner_radius": 0.008,
        "translation": [0.0, 0.0, 0.012],
        "angles": [0.1, -0.05, 0.2],
    },
    "package_half_extent": [0.018, 0.018, 0.004],
}


def _background(step=0.004):
    axis = np.arange(-0.032, 0.0320001, step)
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


def test_charge_regularization_hits_physical_divergence_without_changing_curl():
    bg = _background()
    context = bg.geometry_context(GEOMETRY, assemble_thermal=False)
    G = gradient_operator(bg, gauge_fixed=False)
    for port, coil in enumerate(context.geometry.coils):
        source = np.asarray(context.source_shape[:, port], float)
        target, meta = terminal_charge_target(bg, coil)
        q, net, moment = source_terminal_divergence(bg, source)
        assert net <= 1e-12
        assert np.linalg.norm(q - target) / np.linalg.norm(target) <= 5e-11
        assert np.allclose(moment, np.asarray(meta["terminal_charge_vector"]), atol=1e-12)
        row = context.source_regularization[port]
        assert row["terminal_charge_target_relative_error"] <= 5e-11
        assert row["terminal_charge_lift_relative_curl"] <= 1e-12
        assert row["terminal_charge_curl_preserved"] is True
        # The installed source remains genuinely open; the compatible lift does
        # not turn it into a divergence-free loop.
        assert np.linalg.norm(G.T @ source) > 1e-8


def test_charge_target_moment_is_mesh_invariant_for_same_physical_contact():
    coarse = _background(0.004)
    fine = _background(0.002)
    c0 = coarse.geometry_context(GEOMETRY, assemble_thermal=False)
    c1 = fine.geometry_context(GEOMETRY, assemble_thermal=False)
    for row0, row1 in zip(c0.source_regularization, c1.source_regularization):
        assert np.allclose(
            np.asarray(row0["terminal_charge_vector"], float),
            np.asarray(row1["terminal_charge_vector"], float),
            rtol=0.0,
            atol=2e-12,
        )
        assert row0["terminal_charge_target_relative_error"] <= 5e-11
        assert row1["terminal_charge_target_relative_error"] <= 5e-11
        assert row0["terminal_charge_lift_relative_curl"] <= 1e-12
        assert row1["terminal_charge_lift_relative_curl"] <= 1e-12
