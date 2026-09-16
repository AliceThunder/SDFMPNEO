import types

import numpy as np

import sdfmpneo.unified_terminal_dissipative_defect as terminal_defect


def _background():
    return types.SimpleNamespace(
        background_config={
            "fine_step": 0.012,
            "mesh_check": {"relative_tolerance": 0.1},
            "global_longitudinal_correction": {
                "fine_step": 0.003,
                "terminal_cells_per_support": 1.6,
                "terminal_core_padding_factor": 1.5,
                "terminal_patch_max_cells": 575000,
                "coarse_consistency_tolerance": 1e-8,
            },
            "global_dissipative_reference": {"enabled": False},
            "terminal_dissipative_reference": {
                "enabled": True,
                "reference_cells_per_support": 3.2,
                "validation_ratio": 0.75,
                "core_padding_factor": 1.5,
                "max_cells": 575000,
                "relative_tolerance": 0.1,
                "coarse_consistency_tolerance": 1e-8,
            },
        },
        coil_materials=("tx", "rx"),
    )


def _fake_module():
    module = types.SimpleNamespace()
    module._MODEL = "reactive-v4"
    module._config = lambda background: {"fine_step": 0.003}
    module._background_step = lambda background: float(background.background_config["fine_step"])
    module._cached_context = lambda background, geometry: None
    module._remember_context = lambda background, geometry, context: None
    module._volume_edge_diagonal = lambda background, context: (None, None, None)
    module._boundary_node_mask = lambda background: None
    module._boundary_values = lambda *args, **kwargs: None
    module._interpolate_cell_basis = lambda *args, **kwargs: None
    module._resolve_settings = lambda settings, background: None

    def correction(background, geometry, *, phi=None):
        modal = None if phi is None else np.zeros((np.asarray(phi).shape[1], 2), float)
        return {
            "delta_z": np.array([0.0 + 0.2j, 0.0 + 0.3j]),
            "delta_d_vol": np.zeros(2),
            "delta_d_out": np.zeros(2),
            "delta_modal_h": modal,
            "audit": {"model": "reactive-v4"},
        }

    module._correction = correction
    module.audit_reference_convergence = lambda background, geometry: {
        "model": "reactive-v4",
        "converged": True,
        "maximum_relative_error": 0.08,
        "samples": [],
    }
    return module


def test_terminal_charge_components_preserve_declared_unit_totals():
    q = np.array([-0.2, -0.8, 0.0, 0.3, 0.7])
    feed = terminal_defect._component_charge(q, 0)
    ret = terminal_defect._component_charge(q, 1)
    assert np.all(feed <= 0.0)
    assert np.all(ret >= 0.0)
    assert np.isclose(-np.sum(feed), 1.0)
    assert np.isclose(np.sum(ret), 1.0)
    assert np.allclose(feed + ret, q)


def test_resolve_settings_disables_falsified_uniform_reference():
    module = _fake_module()
    terminal_defect.install(module, types.SimpleNamespace())
    settings = {
        "BACKGROUND": {
            "fine_step": 0.012,
            "mesh_check": {"relative_tolerance": 0.1},
            "global_longitudinal_correction": {
                "terminal_cells_per_support": 1.6,
                "terminal_core_padding_factor": 1.5,
                "terminal_patch_max_cells": 575000,
            },
            "global_dissipative_reference": {"enabled": True},
        }
    }
    background = types.SimpleNamespace(background_config={"fine_step": 0.012})
    module._resolve_settings(settings, background)
    assert settings["BACKGROUND"]["global_dissipative_reference"]["enabled"] is False
    own = settings["BACKGROUND"]["terminal_dissipative_reference"]
    assert np.isclose(own["reference_cells_per_support"], 3.2)
    assert np.isclose(own["validation_ratio"], 0.75)
    assert background.background_config["global_dissipative_reference"]["enabled"] is False


def test_terminal_dissipative_correction_changes_only_real_self_and_loss(monkeypatch):
    module = _fake_module()

    def reference(module_arg, background, geometry, port, terminal, *, cells_per_support, phi=None):
        value = float((port + 1) * (terminal + 1))
        modal = None if phi is None else np.full(np.asarray(phi).shape[1], value)
        return {
            "delta_d_vol": value,
            "delta_modal_h": modal,
        }

    monkeypatch.setattr(terminal_defect, "_terminal_reference", reference)
    terminal_defect.install(module, types.SimpleNamespace())
    phi = np.zeros((5, 1))
    result = module._correction(_background(), "g0", phi=phi)
    # Port 1: 1 + 2 = 3. Port 2: 2 + 4 = 6.
    assert np.allclose(np.real(result["delta_z"]), [3.0, 6.0])
    assert np.allclose(np.imag(result["delta_z"]), [0.2, 0.3])
    assert np.allclose(result["delta_d_vol"], [3.0, 6.0])
    assert np.allclose(result["delta_modal_h"], [[3.0, 6.0]])
    assert result["audit"]["terminal_dissipative_reference_enabled"] is True


def test_terminal_dissipative_audit_controls_combined_convergence(monkeypatch):
    module = _fake_module()

    def reference(module_arg, background, geometry, port, terminal, *, cells_per_support, phi=None):
        is_validation = cells_per_support > 4.0
        base = 10.0 + port + terminal
        delta = base if not is_validation else base + (0.2 if port == 0 else 4.0)
        return {
            "delta_d_vol": delta,
            "coarse": {
                "d_vol": 1.0,
                "parent_restriction_relative_residual": 1e-14,
            },
            "refined": {
                "d_vol": 1.0 + delta,
                "scalar_relative_residual": 1e-13,
            },
        }

    monkeypatch.setattr(terminal_defect, "_terminal_reference", reference)
    terminal_defect.install(module, types.SimpleNamespace())
    report = module.audit_reference_convergence(_background(), "g0")
    assert report["converged"] is False
    assert report["terminal_dissipative_reference"]["converged"] is False
    assert report["maximum_terminal_dissipative_relative_error"] > 0.1
