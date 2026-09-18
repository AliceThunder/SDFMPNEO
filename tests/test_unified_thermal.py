import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sdfmpneo.unified_background import FixedMultiscaleBackground
from sdfmpneo.unified_geometry import UnifiedUWPTGeometry
from sdfmpneo.unified_thermal import (
    _anchor_error,
    _anchor_relative_errors_grouped,
    _greedy_background_residual,
    _group_anchors,
    _transport_field,
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
    assert report.background_rank >= 0
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


def test_transported_self_volume_response_does_not_consume_fixed_background_rank():
    """A translated self-heating mode belongs to the local transported span."""
    bg = make_background()
    reference = UnifiedUWPTGeometry.from_mapping(make_geometry(0.0))
    shifted_mapping = reference.to_mapping()
    shifted_mapping["coils"][0]["translation"][0] += 0.025
    shifted_mapping["packages"][0]["translation"][0] += 0.025
    shifted = UnifiedUWPTGeometry.from_mapping(shifted_mapping)

    x, y, z = bg.cell_centers.T
    canonical = np.exp(-((x / 0.018) ** 2 + (y / 0.018) ** 2 + (z / 0.018) ** 2))
    canonical /= np.sqrt(np.dot(canonical, bg.cell_volumes * canonical))
    moved = _transport_field(
        bg,
        canonical,
        reference.coils[0].pose,
        shifted.coils[0].pose,
    )

    identity = sp.eye(bg.n_cells, format="csr")
    anchors = []
    for gi, vector in enumerate((canonical, moved)):
        anchors.append(
            {
                "A": identity,
                "b": vector.copy(),
                "u": vector.copy(),
                "denom2": float(np.dot(vector, vector)),
                "label": f"geometry[{gi}]/volume[0]/s=0",
                "source_kind": "volume",
                "shift": 0.0,
                "geometry_index": gi,
                "port_index": 0,
                "rhs_norm": float(np.linalg.norm(vector)),
            }
        )

    empty = np.empty((bg.n_cells, 0), float)
    background_modes, _steps, stop, error = _greedy_background_residual(
        bg,
        anchors,
        [reference, shifted],
        reference,
        (canonical[:, None], empty),
        5e-2,
        None,
        None,
    )
    assert stop == "target_reached"
    assert background_modes.shape[1] == 0
    assert error <= 5e-2


def test_grouped_resolvent_errors_match_scalar_evaluation():
    A = sp.diags([2.0, 3.0, 5.0, 7.0], format="csr")
    phi = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.2, 0.1],
            [0.0, 0.3],
        ],
        dtype=float,
    )
    anchors = []
    for j, b in enumerate(
        (
            np.array([1.0, 0.2, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.3, 0.1]),
            np.array([0.1, 0.0, 0.0, 1.0]),
        )
    ):
        u = np.asarray(spla.spsolve(A, b), float)
        anchors.append(
            {
                "A": A,
                "b": b,
                "u": u,
                "denom2": float(u @ (A @ u)),
                "label": f"a{j}",
                "source_kind": "volume",
                "shift": 1.0,
                "geometry_index": 0,
                "port_index": None,
                "rhs_norm": float(np.linalg.norm(b)),
            }
        )
    grouped = dict(
        (anchor["label"], error)
        for anchor, error in _anchor_relative_errors_grouped(_group_anchors(anchors), phi)
    )
    scalar = dict(
        (anchor["label"], _anchor_error(anchor, phi)[0])
        for anchor in anchors
    )
    assert grouped.keys() == scalar.keys()
    for key in grouped:
        assert grouped[key] == pytest.approx(scalar[key], rel=1e-12, abs=1e-12)
