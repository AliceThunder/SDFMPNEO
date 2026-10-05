import json
import types

import numpy as np

import sdfmpneo.unified_charge_regularized_source as charge_source
import sdfmpneo.unified_terminal_contact_source as terminal_source
import sdfmpneo.unified_terminal_dissipative_quadrature_fix as quadrature_fix


def test_source_and_terminal_charge_share_explicit_quadrature_override():
    coil = types.SimpleNamespace(conductor_width=0.004, conductor_thickness=0.002)
    background = types.SimpleNamespace(
        dx=np.array([0.01]),
        dy=np.array([0.012]),
        dz=np.array([0.014]),
        _sdfmpneo_source_quadrature_resolution_override=0.0005,
    )

    source_q = terminal_source._cross_section_quadrature(background, coil)
    charge_q = charge_source._cross_section_quadrature(background, coil)

    assert np.isclose(source_q[-1], 0.0005)
    assert np.isclose(charge_q[-1], 0.0005)
    assert source_q[-3:-1] == charge_q[-3:-1]
    assert np.allclose(source_q[0], charge_q[0])
    assert np.allclose(source_q[2], charge_q[2])


def test_terminal_reference_uses_validation_resolution_and_drops_internal_prepared():
    records = []

    def original_quadrature(background, coil):
        resolution = float(min(np.min(background.dx), np.min(background.dy), np.min(background.dz)))
        return (
            np.array([0.0]),
            np.array([1.0]),
            np.array([0.0]),
            np.array([1.0]),
            1,
            1,
            resolution,
        )

    fake_source = types.SimpleNamespace(
        _cross_section_quadrature=original_quadrature,
        _QUADRATURE_PANEL_TO_MESH=0.5,
        _composite_gauss_1d=terminal_source._composite_gauss_1d,
    )
    fake_charge = types.SimpleNamespace(_cross_section_quadrature=original_quadrature)

    fake_defect = types.SimpleNamespace()
    fake_defect._MODEL = "terminal-dissipative-v2"
    fake_defect._config = lambda background: {
        "validation_cells_per_support": 4.0,
    }

    def terminal_axes(module, background, geometry, port, terminal, coarse_axes, cells):
        assert np.isclose(cells, 4.0)
        axes = (
            np.array([0.0, 0.25, 0.50]),
            np.array([0.0, 0.20, 0.40]),
            np.array([0.0, 0.125, 0.25]),
        )
        return axes, np.array([0.25, 0.20, 0.125]), None, 0.01, 0.01

    fake_defect._terminal_axes = terminal_axes

    def balanced_state(*args, **kwargs):
        patch = args[2]
        records.append(
            (
                bool(kwargs.get("exact_refined", False)),
                getattr(patch, "_sdfmpneo_source_quadrature_resolution_override", None),
            )
        )
        return {"local_d_vol": 1.0}

    fake_defect._balanced_state = balanced_state

    def terminal_reference(
        module,
        background,
        geometry,
        port,
        terminal,
        *,
        cells_per_support,
        phi=None,
        prepared=None,
    ):
        coarse_patch = types.SimpleNamespace()
        refined_patch = types.SimpleNamespace()
        fake_defect._balanced_state(
            module,
            background,
            coarse_patch,
            geometry,
            port,
            None,
            None,
            phi=phi,
            fine_step=0.01,
            certify_parent=True,
            exact_refined=False,
        )
        refined = fake_defect._balanced_state(
            module,
            background,
            refined_patch,
            geometry,
            port,
            None,
            None,
            phi=phi,
            fine_step=0.005,
            certify_parent=False,
            exact_refined=True,
        )
        return {
            "delta_d_vol": 2.0,
            "refined": refined,
            "prepared": prepared,
        }

    fake_defect._terminal_reference = terminal_reference
    fake_longitudinal = types.SimpleNamespace(_MODEL="reactive+terminal-dissipative-v2")

    quadrature_fix.install(fake_source, fake_charge, fake_defect, fake_longitudinal)

    opaque_background = object()
    prepared = (
        np.array([0.0]),
        (np.array([0.0, 1.0]),) * 3,
        "full-geometry",
        opaque_background,
        {"ok": True},
    )
    result = fake_defect._terminal_reference(
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        "geometry",
        0,
        1,
        cells_per_support=3.2,
        prepared=prepared,
    )

    assert records[0] == (False, None)
    assert records[1][0] is True
    assert np.isclose(records[1][1], 0.125)
    assert "prepared" not in result
    assert np.isclose(result["source_quadrature_resolution"], 0.125)
    assert result["refined"]["source_quadrature_locked_to_validation"] is True
    # The public payload must remain JSON encodable even though prepared held an
    # opaque background object.
    json.dumps(result)
