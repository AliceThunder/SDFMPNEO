import numpy as np

from sdfmpneo.unified_open_boundary import OpenBoundaryBackground
from sdfmpneo.unified_tensor_surrogate import solve_port_truth_tensors


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


def geometry():
    coil = dict(
        shape="circle",
        turns=0.5,
        outer_half_size=0.010,
        pitch=0.002,
        conductor_width=0.001,
        conductor_thickness=0.001,
        corner_radius=0.005,
        angles=[0.0, 0.0, 0.0],
    )
    return {
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.012]),
        "receiver": dict(coil, translation=[0.0, 0.0, 0.012]),
        "package_half_extent": [0.016, 0.016, 0.004],
    }


def background():
    axis = np.linspace(-0.05, 0.05, 5)
    return OpenBoundaryBackground(
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


def test_open_boundary_has_passive_nonzero_surface_power_form():
    bg = background()
    admittance = bg.boundary_admittance()
    weights = bg.outward_loss_weights()
    assert bg.boundary_model == "silver_muller_impedance"
    assert admittance.real > 0.0
    assert weights.shape == (bg.n_edges,)
    assert np.all(weights >= 0.0)
    assert np.count_nonzero(weights) > 0


def test_open_boundary_truth_has_independent_poynting_power_balance():
    bg = background()
    z, d, d_out, audit = solve_port_truth_tensors(bg, geometry())
    assert audit["independent_outward_power_available"]
    assert audit["max_linear_relative_residual"] <= 1e-8
    assert audit["reciprocity_relative_error"] <= 1e-8
    assert audit["open_boundary_power_balance_relative_error"] <= 1e-7
    assert np.min(np.linalg.eigvalsh(d)) >= -1e-9
    assert np.min(np.linalg.eigvalsh(d_out)) >= -1e-9
    assert np.allclose(
        0.5 * (z + z.conj().T),
        d + d_out,
        rtol=1e-7,
        atol=1e-9,
    )
