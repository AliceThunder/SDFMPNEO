import types
import inspect
import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_background import FixedMultiscaleBackground
import sdfmpneo.unified_tensor_surrogate as tensor_truth
from sdfmpneo.unified_geometry import UnifiedUWPTGeometry
from sdfmpneo.unified_thermal import (
    _basis_condition_from_gram,
    _certified_basis_condition,
    _weighted_generator_condition,
    _transport_local_field,
    _partition_solution_states,
    _moving_local_allocations,
    _enrich_full_library_residual,
    _maxwell_port_fields,
    _raw_block_condition,
    _stabilize_component_blocks,
    GeometryAwareThermalLibrary,
    build_geometry_aware_thermal_library,
    configure_maxwell_field_cache,
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
    assert report.component_target_relative_error >= report.target_relative_error
    assert report.conditioning_trim_diagnostics
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



def test_component_stabilization_prefers_fixed_background_tail_over_moving_modes():
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

    # The actual near-dependence is between background e0 and the trailing RX
    # mode.  Stabilization deliberately sacrifices the fixed-background tail
    # first because full-library residual enrichment can regenerate fixed
    # directions, whereas a trimmed moving mode cannot follow held-out geometry.
    background_modes = np.column_stack((normalized(e0), normalized(e1)))
    local0 = np.column_stack((normalized(e2), normalized(e3)))
    near_duplicate = normalized(e0 + 1e-7 * e5)
    local1 = np.column_stack((normalized(e4), near_duplicate))

    before = _raw_block_condition(
        bg,
        background_modes,
        (local0, local1),
    )
    assert before > 1e6

    stable_bg, stable_local, info = _stabilize_component_blocks(
        bg,
        reference,
        background_modes,
        (local0, local1),
        [reference],
        1e6,
    )

    assert stable_bg.shape[1] == 1
    assert info["trimmed_background"] == 1
    # A moving tail may still need trimming after the fixed block reaches its
    # one-mode floor, but it must never be trimmed while a fixed tail remains.
    assert stable_local[0].shape[1] == 2
    assert stable_local[1].shape[1] in {1, 2}
    assert info["trimmed_local"][0] == 0
    after = _raw_block_condition(
        bg,
        stable_bg,
        stable_local,
    )
    assert after <= 1e6


def test_component_stabilization_can_trim_local_tail_after_background_floor():
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

    background_modes = np.column_stack((normalized(e0),))
    local0 = np.column_stack((normalized(e1), normalized(e2)))
    local1 = np.column_stack(
        (
            normalized(e3),
            normalized(e0 + 1e-7 * e4),
        )
    )

    stable_bg, stable_local, info = _stabilize_component_blocks(
        bg,
        reference,
        background_modes,
        (local0, local1),
        [reference],
        1e6,
    )

    assert stable_bg.shape[1] == 1
    assert info["trimmed_background"] == 0
    assert sum(info["trimmed_local"]) >= 1
    assert _raw_block_condition(
        bg,
        stable_bg,
        stable_local,
    ) <= 1e6


def test_certified_maxwell_field_cache_survives_background_rebuild(tmp_path, monkeypatch):
    geometry = UnifiedUWPTGeometry.from_mapping(make_geometry(0.0))
    context = types.SimpleNamespace(geometry=geometry)
    A = np.eye(4, dtype=complex)
    B = np.column_stack(
        (
            np.array([1.0, 2.0, -1.0, 0.5], complex),
            np.array([0.25, -0.5, 1.5, 2.0], complex),
        )
    )

    class Background:
        background_config = {
            "linear_solver": {"relative_residual_tolerance": 1e-9}
        }

        @staticmethod
        def em_operator(context, temperature):
            assert temperature is None
            return A

        @staticmethod
        def rhs_matrix(context):
            return B

    calls = {"count": 0}

    def solve_once(background, context):
        calls["count"] += 1
        return B.copy(), 0.0

    monkeypatch.setattr(tensor_truth, "_solve_port_fields", solve_once)
    cache_path = tmp_path / "thermal-fields.npz"

    first_background = Background()
    configure_maxwell_field_cache(first_background, cache_path, "physics-signature")
    first = _maxwell_port_fields(first_background, context)
    assert calls["count"] == 1
    assert cache_path.is_file()
    assert np.allclose(first, B)

    def forbidden_resolve(background, context):
        raise AssertionError("certified Maxwell field cache unexpectedly missed")

    monkeypatch.setattr(tensor_truth, "_solve_port_fields", forbidden_resolve)
    rebuilt_background = Background()
    configure_maxwell_field_cache(
        rebuilt_background,
        cache_path,
        "physics-signature",
    )
    second = _maxwell_port_fields(rebuilt_background, context)
    assert np.allclose(second, B)


def test_maxwell_field_cache_signature_mismatch_recomputes(tmp_path, monkeypatch):
    geometry = UnifiedUWPTGeometry.from_mapping(make_geometry(0.0))
    context = types.SimpleNamespace(geometry=geometry)
    A = np.eye(3, dtype=complex)
    B = np.ones((3, 2), complex)

    class Background:
        background_config = {
            "linear_solver": {"relative_residual_tolerance": 1e-9}
        }

        @staticmethod
        def em_operator(context, temperature):
            return A

        @staticmethod
        def rhs_matrix(context):
            return B

    calls = {"count": 0}

    def solve(background, context):
        calls["count"] += 1
        return B.copy(), 0.0

    monkeypatch.setattr(tensor_truth, "_solve_port_fields", solve)
    cache_path = tmp_path / "thermal-fields.npz"

    first_background = Background()
    configure_maxwell_field_cache(first_background, cache_path, "signature-a")
    _maxwell_port_fields(first_background, context)

    second_background = Background()
    configure_maxwell_field_cache(second_background, cache_path, "signature-b")
    _maxwell_port_fields(second_background, context)

    assert calls["count"] == 2



def test_geometry_aware_thermal_library_schema_v7_round_trip(tmp_path):
    bg = make_background()
    reference = bg.validate_geometry(make_geometry(0.0))
    n = bg.n_cells
    weights = np.asarray(bg.cell_volumes, float)

    def unit(index):
        value = np.zeros(n, float)
        value[index] = 1.0
        return value / np.sqrt(np.dot(value, weights * value))

    library = GeometryAwareThermalLibrary(
        reference,
        np.column_stack((unit(0), unit(1))),
        (
            np.column_stack((unit(2),)),
            np.column_stack((unit(3),)),
        ),
        (0.1, 1.0, 10.0),
        1e10,
    )
    path = tmp_path / "thermal-library.npz"
    library.save(path)
    loaded = GeometryAwareThermalLibrary.load(path)

    assert loaded.rank == library.rank
    assert loaded.block_ranks == library.block_ranks
    assert loaded.time_scales == library.time_scales
    assert loaded.conditioning_limit == library.conditioning_limit
    assert np.allclose(loaded.background_modes, library.background_modes)
    for got, expected in zip(loaded.local_modes, library.local_modes):
        assert np.allclose(got, expected)



def test_component_stabilization_does_not_trim_below_production_condition_limit():
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

    background_modes = np.column_stack((normalized(e0),))
    local0 = np.column_stack((normalized(e2),))
    # For two normalized nearly parallel directions the basis condition is
    # approximately 2/eps, while the Gram condition is its square.
    # eps=3e-7 gives a basis condition of order 1e6--1e7: far above the old
    # accidental 1e5 effective Gate but safely below the declared 1e10 Gate.
    near = normalized(e0 + 3e-7 * e1)
    local1 = np.column_stack((normalized(e3), near))

    before = _raw_block_condition(
        bg,
        background_modes,
        (local0, local1),
    )
    assert 1e6 < before < 1e10

    stable_bg, stable_local, info = _stabilize_component_blocks(
        bg,
        reference,
        background_modes,
        (local0, local1),
        [reference],
        1e10,
    )

    assert stable_bg.shape == background_modes.shape
    assert stable_local[0].shape == local0.shape
    assert stable_local[1].shape == local1.shape
    assert info["trimmed_background"] == 0
    assert info["trimmed_local"] == [0, 0]
    assert info["conditioning_stabilization_target"] == 1e10



def test_basis_condition_is_square_root_of_gram_condition():
    gram = np.diag([1.0, 1e-12])
    assert np.isclose(_basis_condition_from_gram(gram), 1e6)



def test_weighted_generator_condition_stays_resolved_beyond_gram_safe_range():
    weights = np.ones(4, float)
    first = np.array([1.0, 0.0, 0.0, 0.0])
    second = np.array([1.0, 3e-10, 0.0, 0.0])
    second /= np.linalg.norm(second)
    generator = np.column_stack((first, second))

    condition = _weighted_generator_condition(generator, weights)
    certified = _certified_basis_condition(generator, weights)
    assert np.isfinite(condition)
    assert np.isfinite(certified)
    assert 1e9 < condition < 1e10
    assert np.isclose(certified, condition, rtol=1e-10)

    gram = generator.T @ generator
    # The direct generator certificate remains meaningful even if the squared
    # Gram spectrum is at or beyond ordinary double-precision comfort.
    gram_estimate = _basis_condition_from_gram(gram)
    assert condition >= 1e9
    assert not np.isfinite(gram_estimate) or gram_estimate > 1e8



def test_rigid_local_transport_is_identity_at_reference_geometry():
    bg = make_background()
    reference = UnifiedUWPTGeometry.from_mapping(make_geometry(0.0))
    center = reference.coils[0].pose.translation
    radius2 = np.sum((bg.cell_centers - center[None, :]) ** 2, axis=1)
    field = np.exp(-radius2 / (2.0 * 0.012 ** 2))

    transported = _transport_local_field(
        bg,
        field,
        reference,
        reference,
        0,
    )

    assert np.allclose(transported, field, rtol=1e-12, atol=1e-12)


def test_rigid_local_transport_does_not_hardcode_size_deformation():
    bg = make_background()
    source = UnifiedUWPTGeometry.from_mapping(make_geometry(0.0))
    target_mapping = make_geometry(0.0)
    target_mapping["transmitter"]["outer_half_size"] = 0.024
    target_mapping["package_half_extent"] = [0.03, 0.03, 0.004]
    target = UnifiedUWPTGeometry.from_mapping(target_mapping)

    center = source.coils[0].pose.translation
    radius2 = np.sum((bg.cell_centers - center[None, :]) ** 2, axis=1)
    field = np.exp(-radius2 / (2.0 * 0.012 ** 2))
    transported = _transport_local_field(
        bg,
        field,
        source,
        target,
        0,
    )

    # Geometry-size variation lives in the normalized local span assembled from
    # multiple training snapshots; transport itself only changes pose.
    assert np.allclose(transported, field, rtol=1e-12, atol=1e-12)


def test_v14_build_path_partitions_state_and_enriches_moving_residuals():
    source = inspect.getsource(build_geometry_aware_thermal_library)
    assert "_partition_solution_states" in source
    assert "_enrich_full_library_residual" in source
    assert "canonical-source-solve" not in source
    assert "_enrich_background_against_full_library" not in source






def test_partition_solution_states_reconstructs_full_state_and_equation():
    bg = make_background()
    geometry = bg.validate_geometry(make_geometry(0.0))
    n = bg.n_cells
    A = np.eye(n)
    u = np.linspace(0.25, 1.25, n)
    b = A @ u
    anchor = {
        "A": A,
        "b": b,
        "u": u,
        "denom2": float(np.dot(u, b)),
        "label": "geometry[0]/wire[1]/s=10",
        "case_label": "wire[1]",
        "source_kind": "wire[1]",
        "shift": 10.0,
        "geometry_index": 0,
        "rhs_norm": float(np.linalg.norm(b)),
    }

    background_rows, local_rows = _partition_solution_states(
        bg,
        geometry,
        [anchor],
    )

    pieces = [row["u"] for rows in local_rows for row in rows]
    pieces.extend(row["u"] for row in background_rows)
    rhs_pieces = [row["b"] for rows in local_rows for row in rows]
    rhs_pieces.extend(row["b"] for row in background_rows)

    assert pieces
    assert np.allclose(np.sum(np.column_stack(pieces), axis=1), u)
    assert np.allclose(np.sum(np.column_stack(rhs_pieces), axis=1), b)
    assert all(
        row["source_kind"] == "state-local"
        for rows in local_rows
        for row in rows
    )
    assert all(
        row["source_kind"] in {"state-far", "initial"}
        for row in background_rows
    )



def test_residual_enrichment_routes_near_state_error_into_local_atlas():
    bg = make_background()
    reference = bg.validate_geometry(make_geometry(0.0))
    weights = np.asarray(bg.cell_volumes, float)
    allocations, _far = _moving_local_allocations(bg, reference)

    u = np.asarray(allocations[:, 0], float)
    assert np.linalg.norm(u) > 0.0
    A = sp.eye(bg.n_cells, format="csr")
    b = u.copy()
    anchor = {
        "A": A,
        "b": b,
        "u": u,
        "denom2": float(np.dot(u, b)),
        "label": "geometry[0]/synthetic-near/s=1",
        "case_label": "synthetic-near",
        "source_kind": "wire[0]",
        "shift": 1.0,
        "geometry_index": 0,
        "rhs_norm": float(np.linalg.norm(b)),
    }

    seed_index = int(np.argmin(allocations[:, 0]))
    seed = np.zeros(bg.n_cells, float)
    seed[seed_index] = 1.0
    seed /= np.sqrt(np.dot(seed, weights * seed))

    empty = np.empty((bg.n_cells, 0), float)
    (
        _bg_modes,
        local_modes,
        _steps,
        stop,
        final_error,
        diagnostics,
    ) = _enrich_full_library_residual(
        bg,
        reference,
        seed[:, None],
        (empty.copy(), empty.copy()),
        [reference],
        [[anchor]],
        1e-8,
        None,
        (0.1, 1.0, 10.0),
        1e10,
        None,
    )

    assert stop == "target_reached"
    assert final_error <= 1e-8
    assert sum(block.shape[1] for block in local_modes) > 0
    assert sum(diagnostics["residual_local_additions"]) > 0
