import numpy as np

from sdfmpneo.unified_final_audit import _audit_operating_cases, _current_vectors


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