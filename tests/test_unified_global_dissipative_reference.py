import types

import numpy as np

import sdfmpneo.unified_global_dissipative_reference as dissipative


def _background(step=0.012):
    return types.SimpleNamespace(
        background_config={
            "fine_step": float(step),
            "global_dissipative_reference": {
                "enabled": True,
                "reference_step": 0.009,
                "validation_step": 0.00675,
                "reference_max_step": 0.06,
                "validation_max_step": 0.045,
                "relative_tolerance": 0.1,
                "max_cells": 400000,
            },
        }
    )


def _fake_module():
    module = types.SimpleNamespace()
    module._MODEL = "reactive-v4"
    module._geometry_key = lambda geometry: str(geometry)
    module._background_step = lambda background: float(background.background_config["fine_step"])
    module._resolve_settings = lambda settings, background: None

    def correction(background, geometry, *, phi=None):
        modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], 2), float)
        return {
            "delta_z": np.array([0.0 + 0.2j, 0.0 + 0.3j]),
            "delta_d_vol": np.zeros(2),
            "delta_d_out": np.zeros(2),
            "delta_modal_h": modal,
            "audit": {
                "global_scalar": {"d_vol": [3.0, 5.0]},
                "model": "reactive-v4",
            },
        }

    module._correction = correction
    module.audit_reference_convergence = lambda background, geometry: {
        "model": "reactive-v4",
        "converged": True,
        "maximum_relative_error": 0.08,
        "global_scalar": {"d_vol": [3.0, 5.0]},
        "samples": [],
    }
    module._global_scalar_potentials = lambda background, geometry: (
        None,
        None,
        {"d_vol": [3.0, 5.0]},
    )
    return module


def test_dissipative_reference_aligns_coarse_self_loss_and_preserves_reactive_part(monkeypatch):
    module = _fake_module()

    def state(module_arg, background, geometry, *, step, max_step, phi=None):
        assert np.isclose(step, 0.009)
        return {
            "fine_step": step,
            "n_cells": 10,
            "scalar_dofs": 9,
            "d_vol": np.array([6.0, 8.0]),
            "z_reaction": np.array([6.0 + 1j, 8.0 + 2j]),
            "modal_h": None,
            "maximum_scalar_relative_residual": 1e-14,
        }

    monkeypatch.setattr(dissipative, "_reference_state", state)
    dissipative.install(module)
    result = module._correction(_background(0.012), "g0", phi=None)
    assert np.allclose(result["delta_d_vol"], [3.0, 3.0])
    assert np.allclose(np.real(result["delta_z"]), [3.0, 3.0])
    assert np.allclose(np.imag(result["delta_z"]), [0.2, 0.3])
    assert result["audit"]["global_dissipative_reference_enabled"] is True


def test_reference_mesh_is_not_corrected_again(monkeypatch):
    module = _fake_module()
    calls = {"reference": 0}

    def state(*args, **kwargs):
        calls["reference"] += 1
        raise AssertionError("9-mm reference background must not be corrected again")

    monkeypatch.setattr(dissipative, "_reference_state", state)
    dissipative.install(module)
    result = module._correction(_background(0.009), "g0", phi=None)
    assert np.allclose(result["delta_d_vol"], 0.0)
    assert np.allclose(np.real(result["delta_z"]), 0.0)
    assert np.allclose(np.imag(result["delta_z"]), [0.2, 0.3])
    assert calls["reference"] == 0


def test_global_dissipative_gate_uses_reference_vs_validation_defect(monkeypatch):
    module = _fake_module()

    def state(module_arg, background, geometry, *, step, max_step, phi=None):
        if np.isclose(step, 0.009):
            values = np.array([6.0, 8.0])
        elif np.isclose(step, 0.00675):
            values = np.array([6.3, 8.4])
        else:
            raise AssertionError(step)
        return {
            "fine_step": step,
            "n_cells": 10,
            "scalar_dofs": 9,
            "d_vol": values,
            "z_reaction": values.astype(complex) + 1j,
            "modal_h": None,
            "maximum_scalar_relative_residual": 1e-14,
        }

    monkeypatch.setattr(dissipative, "_reference_state", state)
    dissipative.install(module)
    report = module.audit_reference_convergence(_background(0.012), "g0")
    assert report["converged"] is True
    assert report["global_dissipative_reference"]["converged"] is True
    assert report["maximum_global_dissipative_relative_error"] < 0.1


def test_resolve_settings_tracks_existing_full_em_refinement_factor():
    module = _fake_module()
    dissipative.install(module)
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "max_step": 0.08,
            "mesh_check": {"refinement_factor": 0.75, "relative_tolerance": 0.1},
        }
    }
    background = types.SimpleNamespace(background_config={"fine_step": 0.012})
    module._resolve_settings(settings, background)
    own = settings["BACKGROUND"]["global_dissipative_reference"]
    assert np.isclose(own["reference_step"], 0.009)
    assert np.isclose(own["validation_step"], 0.00675)
    assert np.isclose(own["reference_max_step"], 0.06)
    assert np.isclose(own["validation_max_step"], 0.045)
