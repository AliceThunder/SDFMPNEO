import numpy as np

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_thermal import build_thermal_basis


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


def make_geometry(offset=0.0):
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
        "transmitter": dict(coil, translation=[0.0, 0.0, -0.010]),
        "receiver": dict(coil, translation=[offset, 0.0, 0.010]),
        "package_half_extent": [0.018, 0.018, 0.004],
    }


def make_background():
    axis = np.linspace(-0.05, 0.05, 5)
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


def test_thermal_rank_comes_from_multitime_energy_error_and_is_volume_orthonormal():
    bg = make_background()
    basis, report = build_thermal_basis(
        bg,
        [make_geometry(0.0), make_geometry(0.004)],
        validation_geometries=[make_geometry(0.002)],
        target_relative_error=0.9,
        time_scales=(0.1, 1.0, 10.0),
    )

    assert report.converged
    assert report.stop_reason == "target_reached"
    assert report.basis_dimension == bg.thermal_rank == basis.shape[1]
    assert report.maximum_anchor_relative_energy_error <= report.target_relative_error
    assert report.maximum_validation_relative_energy_error <= report.target_relative_error
    assert report.enrichment_geometry_count == 0
    assert report.validation_geometry_count == 1
    assert report.source_direction_count >= 2
    assert report.shifts[0] == 0.0
    assert len(report.shifts) == 4
    assert 0 < bg.thermal_rank <= bg.n_cells

    gram = basis.T @ (bg.cell_volumes[:, None] * basis)
    assert np.allclose(gram, np.eye(bg.thermal_rank), rtol=1e-10, atol=1e-10)


def test_validation_reserve_is_split_into_enrichment_and_held_out_audit():
    bg = make_background()
    reserve = [make_geometry(0.001), make_geometry(0.002), make_geometry(0.003)]
    basis, report = build_thermal_basis(
        bg,
        [make_geometry(0.0), make_geometry(0.004)],
        validation_geometries=reserve,
        target_relative_error=0.99,
        time_scales=(0.1, 1.0),
    )

    assert basis.shape[1] > 0
    assert report.enrichment_geometry_count == 2
    assert report.validation_geometry_count == 1
    assert report.equation_anchor_count > 0
    assert np.isfinite(report.maximum_enrichment_relative_energy_error)
    assert np.isfinite(report.maximum_validation_relative_energy_error)


def test_tighter_energy_error_target_cannot_require_fewer_modes_on_same_anchors():
    geometries = [make_geometry(0.0), make_geometry(0.004)]

    loose = make_background()
    _, loose_report = build_thermal_basis(
        loose,
        geometries,
        target_relative_error=0.9,
        time_scales=(0.1, 1.0),
    )

    tight = make_background()
    _, tight_report = build_thermal_basis(
        tight,
        geometries,
        target_relative_error=0.6,
        time_scales=(0.1, 1.0),
    )

    assert loose_report.converged
    assert tight_report.converged
    assert tight_report.basis_dimension >= loose_report.basis_dimension
    assert tight_report.maximum_anchor_relative_energy_error <= 0.6