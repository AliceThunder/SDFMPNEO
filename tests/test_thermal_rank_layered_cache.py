from pathlib import Path

import numpy as np

from sdfmpneo.thermal import ThermalSpectralModel
from sdfmpneo.training.automatic_thermal_rank import (
    _load_envelope_cache,
    _load_spectrum_cache,
    _save_envelope_cache,
    _save_spectrum_cache,
    _thermal_cache_paths,
    _thermal_envelope_cache_key,
    _thermal_selection_cache_key,
    _thermal_spectrum_cache_key,
)


def _config(tmp_path):
    mesh = tmp_path / "mesh.msh"
    mesh.write_bytes(b"mesh-v1")
    return tmp_path / "model.config.json", {
        "mesh": "mesh.msh",
        "frequency_hz": 1e5,
        "ambient_temperature": 293.15,
        "constitutive_relative_error": 1e-8,
        "materials": {
            "1": {
                "name": "material",
                "electrical_conductivity": 5.0,
                "resistivity_temperature_coefficient": 0.0,
                "reference_temperature": 293.15,
                "relative_permeability": 1.0,
                "thermal_conductivity": 2.0,
                "volumetric_heat_capacity": 3.0,
            }
        },
        "terminal_pairs": [[1, 2]],
        "port_names": ["p"],
        "current_offset": None,
        "current_matrix": None,
        "thermal_truncation": {
            "relative_tolerance": 1e-3,
            "absolute_tolerance": 0.0,
            "initial_coordinate_bound": 0.1,
            "source_bound_safety_factor": 2.0,
            "restart_state_safety_factor": 1.5,
            "temperature_probe_axes": 4,
            "cache": True,
        },
        "training": {
            "operating_lower": [0.0],
            "operating_upper": [1.0],
        },
    }


def test_tolerance_change_reuses_spectrum_and_envelope(tmp_path):
    path, config = _config(tmp_path)
    spectrum0 = _thermal_spectrum_cache_key(path, config)
    envelope0 = _thermal_envelope_cache_key(path, config, spectrum0)
    selection0 = _thermal_selection_cache_key(path, config, envelope0)

    config["thermal_truncation"]["relative_tolerance"] = 2e-3

    spectrum1 = _thermal_spectrum_cache_key(path, config)
    envelope1 = _thermal_envelope_cache_key(path, config, spectrum1)
    selection1 = _thermal_selection_cache_key(path, config, envelope1)
    assert spectrum1 == spectrum0
    assert envelope1 == envelope0
    assert selection1 != selection0


def test_electrical_change_reuses_thermal_spectrum_only(tmp_path):
    path, config = _config(tmp_path)
    spectrum0 = _thermal_spectrum_cache_key(path, config)
    envelope0 = _thermal_envelope_cache_key(path, config, spectrum0)

    config["materials"]["1"]["electrical_conductivity"] = 7.0

    spectrum1 = _thermal_spectrum_cache_key(path, config)
    envelope1 = _thermal_envelope_cache_key(path, config, spectrum1)
    assert spectrum1 == spectrum0
    assert envelope1 != envelope0


def test_thermal_change_invalidates_spectrum_and_envelope(tmp_path):
    path, config = _config(tmp_path)
    spectrum0 = _thermal_spectrum_cache_key(path, config)
    envelope0 = _thermal_envelope_cache_key(path, config, spectrum0)

    config["materials"]["1"]["thermal_conductivity"] = 4.0

    spectrum1 = _thermal_spectrum_cache_key(path, config)
    envelope1 = _thermal_envelope_cache_key(path, config, spectrum1)
    assert spectrum1 != spectrum0
    assert envelope1 != envelope0


def test_layered_cache_round_trips(tmp_path):
    path, config = _config(tmp_path)
    paths = _thermal_cache_paths(path, config["thermal_truncation"])
    skey = _thermal_spectrum_cache_key(path, config)
    ekey = _thermal_envelope_cache_key(path, config, skey)

    M = np.diag([2.0, 3.0])
    K = np.diag([4.0, 9.0])
    spectrum = ThermalSpectralModel.build(M, K)
    _save_spectrum_cache(paths["spectrum"], skey, spectrum)
    loaded = _load_spectrum_cache(paths["spectrum"], skey, M, K)
    assert loaded is not None
    assert np.allclose(loaded.lambdas, spectrum.lambdas)
    assert np.allclose(abs(loaded.Phi), abs(spectrum.Phi))

    envelope = np.array([1.0, 0.02])
    _save_envelope_cache(paths["envelope"], ekey, envelope, 5, 3)
    cached = _load_envelope_cache(paths["envelope"], ekey)
    assert cached is not None
    observed, n_operating, n_states = cached
    assert np.allclose(observed, envelope)
    assert (n_operating, n_states) == (5, 3)
    assert Path(paths["selection"]).name.endswith(".thermal_rank.cache.json")
