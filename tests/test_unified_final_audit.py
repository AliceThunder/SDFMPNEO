import numpy as np

from sdfmpneo.unified_final_audit import _audit_operating_cases, _current_vectors
from sdfmpneo.unified_spatial_final_audit import (
    _load_truth_cache,
    _save_truth_cache,
)
from sdfmpneo.unified_tensor_surrogate import SpatialDecodedTensors


def test_current_vectors_cover_hermitian_phase_directions_for_two_ports():
    vectors = _current_vectors(2)
    assert len(vectors) == 6
    e0 = np.array([1.0, 0.0], complex)
    e1 = np.array([0.0, 1.0], complex)
    expected = [e0, e1, e0 + e1, e0 - e1, e0 + 1j * e1, e0 - 1j * e1]
    assert all(any(np.allclose(value, target) for value in vectors) for target in expected)


def test_final_audit_operating_cases_are_independent_of_prediction_defaults():
    settings = {
        "TRAINING": {
            "final_audit": {
                "operating_cases": [
                    {"name": "current", "operating": [3.0, 1.0]},
                    {"name": "circuit", "drive": {
                        "voltage": [7.0, 0.0],
                        "series_impedance": [0.2, 0.3],
                    }},
                ]
            }
        },
        "PREDICTION": {"operating": [999.0, 999.0]},
    }
    cases = _audit_operating_cases(settings, settings["TRAINING"]["final_audit"])
    assert [name for name, _ in cases] == ["current", "circuit"]
    assert cases[0][1] == [3.0, 1.0]
    assert cases[1][1]["voltage"] == [7.0, 0.0]


def test_final_audit_can_fall_back_to_prediction_for_custom_minimal_settings():
    settings = {"TRAINING": {}, "PREDICTION": {"operating": [1.0, 0.0]}}
    cases = _audit_operating_cases(settings, {})
    assert cases == [("prediction_default", [1.0, 0.0])]


def test_spatial_final_audit_truth_cache_roundtrip(tmp_path):
    coil = {
        "shape": "circle",
        "turns": 1.5,
        "outer_half_size": 0.025,
        "pitch": 0.002,
        "conductor_width": 0.0015,
        "conductor_thickness": 0.001,
        "corner_radius": 0.012,
        "angles": [0.0, 0.0, 0.0],
    }
    geometry = {
        "transmitter": dict(
            coil,
            translation=[0.0, 0.0, 0.0],
        ),
        "receiver": dict(
            coil,
            translation=[0.0, 0.0, 0.035],
        ),
        "package_half_extent": [0.035, 0.035, 0.005],
    }
    z = np.array(
        [[2.0 + 0.2j, 0.1 - 0.03j], [0.1 - 0.03j, 1.8 + 0.25j]],
        complex,
    )
    d = np.array(
        [[1.2, 0.05 + 0.01j], [0.05 - 0.01j, 1.0]],
        complex,
    )
    cells = np.stack(
        [0.2 * d, 0.3 * d, 0.5 * d],
        axis=0,
    )
    d_out = 0.1 * np.eye(2, dtype=complex)
    audit = {
        "minimum_d_vol_eigenvalue": np.float64(0.9),
        "nested": {
            "complex_diagnostic": np.complex128(1.0 + 2.0j),
        },
    }
    bundle = (
        SpatialDecodedTensors(
            z,
            d,
            cells,
            d_out,
            0.0,
            0.0,
        ),
        audit,
    )
    path = tmp_path / "heldout_truth.npz"
    _save_truth_cache(
        path,
        cache_key="physical-signature",
        geometries=[geometry],
        bundles=[bundle],
    )
    restored = _load_truth_cache(
        path,
        cache_key="physical-signature",
        geometries=[geometry],
        n_ports=2,
        n_cells=3,
    )
    assert restored is not None
    truth, restored_audit = restored[0]
    assert np.allclose(truth.z_field, z)
    assert np.allclose(truth.d_vol, d)
    assert np.allclose(truth.cell_h, cells)
    assert np.allclose(truth.implied_d_out, d_out)
    assert restored_audit["minimum_d_vol_eigenvalue"] == 0.9
    assert restored_audit["nested"]["complex_diagnostic"] == {
        "real": 1.0,
        "imag": 2.0,
    }

    assert _load_truth_cache(
        path,
        cache_key="different-physics",
        geometries=[geometry],
        n_ports=2,
        n_cells=3,
    ) is None
