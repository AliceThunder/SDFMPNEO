import types

import numpy as np
import scipy.sparse as sp
import pytest

import sdfmpneo.unified_longitudinal_patch_consistency as consistency


def _parent_and_patch(rhs_value=2.0):
    parent = types.SimpleNamespace(
        nx=1,
        ny=1,
        nz=1,
        x=np.array([0.0, 1.0]),
        y=np.array([0.0, 1.0]),
        z=np.array([0.0, 1.0]),
    )
    patch = types.SimpleNamespace()
    patch.geometry_context = lambda geometry, assemble_thermal=False: object()
    patch.rhs_matrix = lambda context: np.array([[complex(rhs_value)]])
    return parent, patch


def _fake_scalar_module():
    G = sp.csr_matrix(([-1.0, 1.0], ([0, 0], [0, 1])), shape=(1, 8))
    coords = np.array(
        [[i, j, k] for i in (0.0, 1.0) for j in (0.0, 1.0) for k in (0.0, 1.0)],
        float,
    )
    boundary = np.ones(8, bool)
    boundary[1] = False
    return types.SimpleNamespace(
        node_coordinates=lambda patch: coords,
        gradient_operator=lambda patch, gauge_fixed=False: G,
        _volume_edge_diagonal=lambda patch, context: (
            np.array([1.0 + 0.0j]),
            np.array([0.0]),
            np.array([0.0]),
        ),
        _boundary_node_mask=lambda patch: boundary,
    )


def test_parent_scalar_restriction_has_zero_patch_residual_when_physics_match():
    parent, patch = _parent_and_patch(rhs_value=2.0)
    module = _fake_scalar_module()
    # Flattening order is (x,y,z); node 1 is (0,0,1).  With phi_0=0,
    # phi_1=2 and unit edge mass, the single interior equation is exact.
    potential = np.zeros(8, complex)
    potential[1] = 2.0
    residual = consistency.coarse_parent_restriction_residual(
        module, parent, patch, object(), potential
    )
    assert residual <= 1e-14


def test_parent_scalar_restriction_detects_changed_patch_rhs():
    parent, patch = _parent_and_patch(rhs_value=3.0)
    module = _fake_scalar_module()
    potential = np.zeros(8, complex)
    potential[1] = 2.0
    residual = consistency.coarse_parent_restriction_residual(
        module, parent, patch, object(), potential
    )
    assert residual > 0.1


def test_install_rejects_uncertified_patch_in_gate_and_production():
    module = types.SimpleNamespace()
    module._background_step = lambda background: 0.012
    module._patch_state = lambda *args, **kwargs: {}
    module.audit_reference_convergence = lambda background, geometry: {
        "converged": True,
        "maximum_relative_error": 0.02,
        "samples": [
            {
                "coarse": {
                    "parent_restriction_relative_residual": 2.0e-7,
                }
            }
        ],
    }
    module._correction = lambda background, geometry, phi=None: {
        "delta_z": np.zeros(2, complex),
        "delta_d_vol": np.zeros(2),
        "delta_d_out": np.zeros(2),
        "delta_modal_h": None,
        "audit": {
            "enabled": True,
            "ports": [
                {
                    "coarse": {
                        "parent_restriction_relative_residual": 2.0e-7,
                    }
                }
            ],
        },
    }
    module._resolve_settings = lambda settings, background: None

    consistency.install(module)
    background = types.SimpleNamespace(
        background_config={
            "global_longitudinal_correction": {
                "coarse_consistency_tolerance": 1e-8,
            }
        }
    )

    report = module.audit_reference_convergence(background, object())
    assert not report["converged"]
    assert not report["coarse_parent_restriction_consistent"]
    assert np.isclose(
        report["maximum_coarse_parent_restriction_relative_residual"], 2e-7
    )

    with pytest.raises(RuntimeError, match="not the parent scalar restriction"):
        module._correction(background, object())


def test_resolve_settings_adds_strict_consistency_tolerance():
    module = types.SimpleNamespace()
    module._background_step = lambda background: 0.012
    module._patch_state = lambda *args, **kwargs: {}
    module.audit_reference_convergence = lambda background, geometry: {
        "converged": True,
        "samples": [],
    }
    module._correction = lambda background, geometry, phi=None: {
        "audit": {"enabled": False}
    }

    def original_resolve(settings, background):
        settings["BACKGROUND"].setdefault("global_longitudinal_correction", {})
        background.background_config.setdefault("global_longitudinal_correction", {})

    module._resolve_settings = original_resolve
    consistency.install(module)

    settings = {"BACKGROUND": {}}
    background = types.SimpleNamespace(background_config={})
    module._resolve_settings(settings, background)
    assert np.isclose(
        settings["BACKGROUND"]["global_longitudinal_correction"][
            "coarse_consistency_tolerance"
        ],
        1e-8,
    )
    assert np.isclose(
        background.background_config["global_longitudinal_correction"][
            "coarse_consistency_tolerance"
        ],
        1e-8,
    )
