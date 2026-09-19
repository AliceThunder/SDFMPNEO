import numpy as np

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_thermal import (
    _raw_block_condition,
    _stabilize_component_blocks,
    build_geometry_aware_thermal_library,
)


MATERIALS = {
    "tx_copper": dict(electrical_conductivity=5.8e7, resistivity_temperature_coefficient=0.00393,
                      reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=1.0,
                      thermal_conductivity=400.0, volumetric_heat_capacity=3.45e6),
    "rx_copper": dict(electrical_conductivity=5.8e7, resistivity_temperature_coefficient=0.00393,
                      reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=1.0,
                      thermal_conductivity=400.0, volumetric_heat_capacity=3.45e6),
    "tx_package": dict(electrical_conductivity=0.0, resistivity_temperature_coefficient=0.0,
                       reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=3.0,
                       thermal_conductivity=0.2, volumetric_heat_capacity=1.5e6),
    "rx_package": dict(electrical_conductivity=0.0, resistivity_temperature_coefficient=0.0,
                       reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=3.0,
                       thermal_conductivity=0.2, volumetric_heat_capacity=1.5e6),
    "seawater": dict(electrical_conductivity=5.0, resistivity_temperature_coefficient=0.0,
                     reference_temperature=293.15, relative_permeability=1.0, relative_permittivity=80.0,
                     thermal_conductivity=0.6, volumetric_heat_capacity=4.1e6),
}


def make_geometry(offset=0.0, yaw=0.0):
    coil = dict(shape="circle", turns=0.5, outer_half_size=0.012, pitch=0.002,
                conductor_width=0.001, conductor_thickness=0.001, corner_radius=0.006)
    return {
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.010], angles=[0.0, 0.0, 0.0]),
        "receiver": dict(coil, translation=[offset, 0.0, 0.010], angles=[0.0, 0.0, yaw]),
        "package_half_extent": [0.018, 0.018, 0.004],
    }


def make_background():
    axis = np.linspace(-0.05, 0.05, 5)
    return FixedMultiscaleBackground(
        axis, axis, axis, frequency_hz=100000.0, materials=MATERIALS,
        coil_materials=("tx_copper", "rx_copper"),
        package_materials=("tx_package", "rx_package"),
        seawater_material="seawater", ambient_temperature=293.15,
    )


def test_geometry_aware_basis_has_fixed_rank_and_moves_with_geometry():
    bg = make_background()
    reference = make_geometry(0.0)
    library, report = build_geometry_aware_thermal_library(
        bg,
        reference,
        [make_geometry(0.002)],
        validation_geometries=[make_geometry(0.001)],
        target_relative_error=0.99,
        time_scales=(0.1, 1.0),
        trajectory_times=(0.1, 1.0, 10.0),
    )
    assert report.converged
    assert report.stop_reason == "target_reached"
    assert report.basis_dimension == library.rank
    assert report.background_rank > 0
    assert len(report.local_ranks) == 2
    assert all(rank > 0 for rank in report.local_ranks)
    assert report.maximum_validation_relative_energy_error <= report.target_relative_error
    assert report.maximum_validation_trajectory_relative_error <= report.target_relative_error
    assert report.validation_geometry_count == 1
    assert report.shifts[0] == 0.0
    assert report.trajectory_times == (0.1, 1.0, 10.0)
    assert report.trajectory_diagnostics
    assert report.worst_validation_trajectory

    phi0 = library.basis_for_geometry(bg, reference)
    phi1 = library.basis_for_geometry(bg, make_geometry(0.003, 0.2))
    assert phi0.shape == phi1.shape == (bg.n_cells, library.rank)
    assert not np.allclose(phi0, phi1)
    gram0 = phi0.T @ (bg.cell_volumes[:, None] * phi0)
    gram1 = phi1.T @ (bg.cell_volumes[:, None] * phi1)
    assert np.allclose(gram0, np.eye(library.rank), rtol=1e-9, atol=1e-9)
    assert np.allclose(gram1, np.eye(library.rank), rtol=1e-9, atol=1e-9)


def test_identity_geometry_reproduces_deterministic_basis_and_bounds():
    bg = make_background()
    reference = make_geometry(0.0)
    library, _ = build_geometry_aware_thermal_library(
        bg, reference, [reference], target_relative_error=0.99, time_scales=(0.1, 1.0)
    )
    first = library.basis_for_geometry(bg, reference)
    second = library.basis_for_geometry(bg, reference)
    assert np.allclose(first, second)
    lo, hi = library.mode_bounds(bg, reference)
    assert lo.shape == hi.shape == (library.rank,)
    assert np.all(hi >= lo)


def test_held_out_diagnostics_identify_source_time_and_trajectory_output():
    bg = make_background()
    library, report = build_geometry_aware_thermal_library(
        bg,
        make_geometry(0.0),
        [make_geometry(0.002)],
        validation_geometries=[make_geometry(0.001)],
        target_relative_error=0.999,
        time_scales=(0.1, 1.0),
        trajectory_times=(0.1, 1.0),
    )
    assert library.rank > 0
    assert report.validation_diagnostics
    assert report.worst_validation_anchor
    assert report.worst_validation_anchor["source_kind"] in {
        "volume", "wire[0]", "wire[1]", "initial"
    }
    assert "relative_energy_error" in report.worst_validation_anchor
    assert report.trajectory_diagnostics
    assert report.worst_validation_trajectory["case"].startswith(("volume", "wire", "initial"))
    assert "field_mass_relative_error" in report.worst_validation_trajectory
    assert "maximum_temperature_relative_error" in report.worst_validation_trajectory
    assert "maximum_wire_average_relative_error" in report.worst_validation_trajectory



def test_component_stabilization_trims_only_greedy_tails_and_preserves_fixed_blocks():
    bg = make_background()
    reference = bg.validate_geometry(make_geometry(0.0))
    weights = np.asarray(bg.cell_volumes, float)
    n = bg.n_cells

    def normalized(vector):
        value = np.asarray(vector, float).reshape(n)
        return value / np.sqrt(np.dot(value, weights * value))

    e0 = np.zeros(n); e0[0] = 1.0
    e1 = np.zeros(n); e1[1] = 1.0
    e2 = np.zeros(n); e2[2] = 1.0
    e3 = np.zeros(n); e3[3] = 1.0
    e4 = np.zeros(n); e4[4] = 1.0
    e5 = np.zeros(n); e5[5] = 1.0

    background_modes = np.column_stack((normalized(e0), normalized(e1)))
    local0 = np.column_stack((normalized(e2), normalized(e3)))
    # The first RX mode is independent; the trailing greedy mode is almost a
    # duplicate of a background direction and should be the one trimmed.
    near_duplicate = normalized(e0 + 1e-7 * e5)
    local1 = np.column_stack((normalized(e4), near_duplicate))

    before = _raw_block_condition(
        bg,
        background_modes,
        (local0, local1),
    )
    assert before > 1e10

    stable_bg, stable_local, info = _stabilize_component_blocks(
        bg,
        reference,
        background_modes,
        (local0, local1),
        [reference],
        1e10,
    )

    assert stable_bg.shape[1] == 2
    assert stable_local[0].shape[1] == 2
    assert stable_local[1].shape[1] == 1
    assert info["trimmed_background"] == 0
    assert info["trimmed_local"] == [0, 1]
    after = _raw_block_condition(
        bg,
        stable_bg,
        stable_local,
    )
    assert after <= 1e10
