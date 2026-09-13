import numpy as np
import pytest

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_thermal import build_thermal_basis


MATERIALS = {
    "tx_copper": dict(
        electrical_conductivity=5.8e7, resistivity_temperature_coefficient=0.00393,
        reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=1.0,
        thermal_conductivity=400.0, volumetric_heat_capacity=3.45e6,
    ),
    "rx_copper": dict(
        electrical_conductivity=5.8e7, resistivity_temperature_coefficient=0.00393,
        reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=1.0,
        thermal_conductivity=400.0, volumetric_heat_capacity=3.45e6,
    ),
    "tx_package": dict(
        electrical_conductivity=0.0, resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=3.0,
        thermal_conductivity=0.2, volumetric_heat_capacity=1.5e6,
    ),
    "rx_package": dict(
        electrical_conductivity=0.0, resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=3.0,
        thermal_conductivity=0.2, volumetric_heat_capacity=1.5e6,
    ),
    "seawater": dict(
        electrical_conductivity=5.0, resistivity_temperature_coefficient=0.0,
        reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=80.0,
        thermal_conductivity=0.6, volumetric_heat_capacity=4.1e6,
    ),
}


def geometry(rx_x=0.0):
    coil = dict(
        shape="circle", turns=0.5, outer_half_size=0.014, pitch=0.002,
        conductor_width=0.001, conductor_thickness=0.001,
        corner_radius=0.007, angles=[0.0, 0.0, 0.0],
    )
    return {
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.012]),
        "receiver": dict(coil, translation=[rx_x, 0.0, 0.012]),
        "package_half_extent": [0.022, 0.022, 0.006],
    }


def background():
    axis = np.linspace(-0.06, 0.06, 7)
    return FixedMultiscaleBackground(
        axis, axis, axis,
        frequency_hz=100000.0,
        materials=MATERIALS,
        coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"),
        seawater_material="seawater",
        ambient_temperature=293.15,
    )


def test_background_keeps_seawater_volume_and_accepts_physical_temperature_samples():
    bg = background()
    assert bg.thermal_rank == 0
    ctx = bg.geometry_context(geometry(), assemble_thermal=False)
    total = sum(ctx.fractions.values())
    assert np.allclose(total, 1.0, rtol=0.0, atol=1e-12)
    assert np.count_nonzero(ctx.fractions["seawater"] > 0.0) > bg.n_cells // 2
    assert ctx.source_shape.shape == (bg.n_edges, 2)

    A = bg.em_operator(ctx, {"tx_copper": 30.0, "rx_copper": 50.0})
    B = bg.rhs_matrix(ctx)
    assert A.shape == (bg.n_edges, bg.n_edges)
    assert B.shape == (bg.n_edges, 2)
    assert np.all(np.isfinite(A.data))
    assert np.linalg.norm(B) > 0.0


def test_automatic_thermal_basis_enables_hard_reduced_operators():
    bg = background()
    _, report = build_thermal_basis(bg, [geometry()], target_relative_residual=0.8)
    assert report.converged
    assert bg.thermal_rank == report.basis_dimension
    assert bg.thermal_rank > 0
    ctx = bg.geometry_context(geometry())
    np.linalg.cholesky(0.5 * (ctx.thermal_mass_reduced + ctx.thermal_mass_reduced.T))
    np.linalg.cholesky(0.5 * (ctx.thermal_stiffness_reduced + ctx.thermal_stiffness_reduced.T))


def test_seawater_joule_weight_is_three_dimensional_and_nonnegative():
    bg = background()
    ctx = bg.geometry_context(geometry(), assemble_thermal=False)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(bg.n_edges, 2)) + 1j * rng.normal(size=(bg.n_edges, 2))
    ex, ey, ez, weight = bg.material_joule_cells(ctx, None, X)
    assert ex.shape == ey.shape == ez.shape == (bg.n_cells, 2)
    assert weight.shape == (bg.n_cells,)
    assert np.all(weight >= 0.0)
    assert np.count_nonzero(weight > 0.0) > bg.n_cells // 2


def test_geometry_outside_physical_background_is_not_silently_clipped():
    bg = background()
    with pytest.raises(ValueError, match="outside the fixed physical background"):
        bg.geometry_context(geometry(rx_x=0.07), assemble_thermal=False)
