import numpy as np
from types import SimpleNamespace

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_corrected_truth import _refresh_audit
from sdfmpneo.unified_online_thermal import (
    audit_online_thermal_trajectories,
    build_online_thermal_context,
    merge_thermal_time_scales,
)
from sdfmpneo.unified_tensor_surrogate import (
    decode_spatial_tensors,
    normalize_cell_joule_tensors,
    pack_spatial_tensors,
    project_spatial_dataset_to_production_cone,
    SpatialTensorDataset,
    spatial_tensor_output_dimension,
)


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


def make_geometry():
    coil = dict(
        shape="circle",
        turns=0.5,
        outer_half_size=0.012,
        pitch=0.002,
        conductor_width=0.001,
        conductor_thickness=0.001,
        corner_radius=0.006,
    )
    return {
        "transmitter": dict(
            coil,
            translation=[0.0, 0.0, -0.010],
            angles=[0.0, 0.0, 0.0],
        ),
        "receiver": dict(
            coil,
            translation=[0.002, 0.0, 0.010],
            angles=[0.0, 0.0, 0.1],
        ),
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


def test_spatial_decoder_enforces_cell_psd_and_exact_total_d():
    d = np.array(
        [
            [4.0, 0.8 + 0.3j],
            [0.8 - 0.3j, 3.0],
        ],
        complex,
    )
    raw_cells = np.array(
        [
            [[2.0, 1.5 + 0.2j], [1.5 - 0.2j, 0.2]],
            [[-0.2, -0.4j], [0.4j, 1.0]],
            [[0.5, 0.1 - 0.3j], [0.1 + 0.3j, 0.8]],
        ],
        complex,
    )
    d_projected, cells = normalize_cell_joule_tensors(
        raw_cells,
        d,
    )
    eigenvalues = np.linalg.eigvalsh(cells)
    assert np.min(eigenvalues) >= -1e-10
    assert np.allclose(
        np.sum(cells, axis=0),
        d_projected,
        rtol=1e-10,
        atol=1e-10,
    )

    z = d_projected + np.eye(2) + 1j * np.array(
        [[0.2, -0.1], [-0.1, 0.4]],
        float,
    )
    packed = pack_spatial_tensors(z, d_projected, cells)
    assert packed.size == spatial_tensor_output_dimension(2, 3)
    decoded = decode_spatial_tensors(packed, 2, 3)

    for current in (
        np.array([1.0, 0.0], complex),
        np.array([0.0, 1.0], complex),
        np.array([1.0, 1.0], complex),
        np.array([1.0, 1.0j], complex),
    ):
        q = decoded.cell_heat(current)
        assert np.min(q) >= 0.0
        assert np.isclose(
            np.sum(q),
            decoded.volume_power(current),
            rtol=1e-10,
            atol=1e-10,
        )


def test_online_thermal_rom_is_geometry_local_and_preserves_uniform_initial():
    bg = make_background()
    geometry = make_geometry()
    n = bg.n_cells

    profile = np.linspace(1.0, 2.0, n)
    profile /= np.sum(profile)
    second = np.linspace(2.0, 1.0, n)
    second /= np.sum(second)
    cell_h = np.zeros((n, 2, 2), complex)
    for k in range(n):
        v = np.array(
            [
                np.sqrt(profile[k]),
                (0.25 + 0.15j) * np.sqrt(second[k]),
            ],
            complex,
        )
        cell_h[k] = np.outer(v, v.conj())
        cell_h[k] += second[k] * np.diag([0.05, 0.4])

    context = build_online_thermal_context(
        bg,
        geometry,
        cell_h,
        time_scales=(0.1, 1.0),
        conditioning_limit=1e10,
        target_relative_error=0.99,
    )
    phi = np.asarray(context.thermal_basis, float)
    assert 1 <= phi.shape[1] < bg.n_cells
    assert context.online_thermal_report.rank == phi.shape[1]
    assert context.online_thermal_report.source_count == 6
    assert context.online_thermal_report.conditioning <= 1e10

    ones = np.ones(bg.n_cells, float)
    rhs = phi.T @ (context.thermal_mass_full @ ones)
    coefficients = np.linalg.solve(
        context.thermal_mass_reduced,
        rhs,
    )
    assert np.allclose(
        phi @ coefficients,
        ones,
        rtol=1e-9,
        atol=1e-9,
    )

    audit = audit_online_thermal_trajectories(
        bg,
        geometry,
        cell_h,
        times=(0.1, 1.0, 10.0),
        time_scales=(0.1, 1.0),
        conditioning_limit=1e10,
        target_relative_error=0.99,
    )
    assert audit["converged"]
    assert audit["rank"] < bg.n_cells
    assert audit["maximum_mass_relative_error"] <= 0.99
    assert audit["basis_time_scales"] == [0.1, 1.0, 10.0]


def test_merge_thermal_time_scales_skips_initial_and_infinite_queries():
    merged = merge_thermal_time_scales(
        (0.1, 1.0, 10.0),
        (0.0, 100.0, 1000.0, "inf"),
    )
    assert merged == (0.1, 1.0, 10.0, 100.0, 1000.0)


def test_cached_spatial_dataset_is_upgraded_to_production_passive_cone():
    d = np.array(
        [[4.0, 0.5 + 0.2j], [0.5 - 0.2j, 3.0]],
        complex,
    )
    cells = np.zeros((3, 2, 2), complex)
    cells[0] = 0.4 * d
    cells[1] = 0.35 * d
    cells[2] = 0.25 * d
    # Deliberately make Herm(Z)-D indefinite.  The production decoder must add
    # the minimum passive correction, and cached labels must be rewritten onto
    # that exact same cone before training.
    z = np.array(
        [[3.5 + 0.2j, 0.2 - 0.1j], [0.2 - 0.1j, 2.5 + 0.4j]],
        complex,
    )
    packed = pack_spatial_tensors(z, d, cells)
    dataset = SpatialTensorDataset(
        inputs=np.zeros((1, 1), float),
        outputs=np.asarray([packed], float),
        split=np.asarray(["train"]),
        audit={
            "minimum_d_vol_eigenvalue": -1.0,
            "minimum_implied_outward_eigenvalue": -1.0,
            "minimum_cell_joule_tensor_eigenvalue": -1.0,
            "maximum_spatial_joule_total_mismatch": 1.0,
            "maximum_spatial_truth_projection_correction": 0.0,
        },
        n_ports=2,
        n_cells=3,
    )

    correction = project_spatial_dataset_to_production_cone(dataset)
    decoded = decode_spatial_tensors(dataset.outputs[0], 2, 3)
    assert correction > 0.0
    assert np.min(np.linalg.eigvalsh(decoded.d_vol).real) >= -1e-12
    assert np.min(np.linalg.eigvalsh(decoded.implied_d_out).real) >= -1e-12
    assert np.min(np.linalg.eigvalsh(decoded.cell_h).real) >= -1e-12
    assert np.allclose(
        np.sum(decoded.cell_h, axis=0),
        decoded.d_vol,
        rtol=1e-10,
        atol=1e-10,
    )
    second = project_spatial_dataset_to_production_cone(dataset)
    assert second <= 1e-10


def test_corrected_audit_preserves_independent_physical_outward_measurement():
    z = np.array([[3.0 + 0.2j]], complex)
    d = np.array([[2.0]], complex)
    d_out = np.array([[1.0]], complex)
    raw_audit = {
        "minimum_physical_outward_eigenvalue": 0.75,
        "open_boundary_power_balance_relative_error": 2.5e-12,
    }
    correction = SimpleNamespace(
        audit={
            "enabled": True,
            "corrected_power_balance_relative_error": 1e-13,
            "maximum_joule_total_power_relative_error": 1e-14,
        }
    )
    refreshed = _refresh_audit(z, d, d_out, raw_audit, correction)
    assert refreshed["minimum_physical_outward_eigenvalue"] == 0.75
    assert refreshed["open_boundary_power_balance_relative_error"] == 2.5e-12
    assert refreshed["minimum_implied_outward_eigenvalue"] >= 0.0
