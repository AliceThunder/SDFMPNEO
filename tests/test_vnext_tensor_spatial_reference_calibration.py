import numpy as np
import pytest

import sdfmpneo_vnext.tensor_spatial_reference as tensor_spatial_reference


class _Prepared:
    raw_dielectric_closure_error = 0.6456
    normalized_dielectric_closure_error = 2.0e-12


def _call():
    return tensor_spatial_reference.prepare_tensor_spatial_reference_adaptive(
        object(),
        object(),
        volume_axial_order=6,
        volume_radial_order=4,
        volume_azimuthal_order=16,
        background_radial_order=10,
        background_angular_order=32,
        maximum_raw_closure_error=0.35,
        maximum_quadrature_refinements=4,
    )


def test_tensor_spatial_raw_closure_uses_same_resolution_calibration(monkeypatch):
    calls = []

    def fake_prepare(*args, maximum_raw_closure_error, **kwargs):
        calls.append((float(maximum_raw_closure_error), dict(kwargs)))
        if np.isfinite(maximum_raw_closure_error):
            raise RuntimeError(
                "raw electric-environment field integration does not close the "
                "port-level dielectric loss channel: relative error=6.456e-01"
            )
        return _Prepared()

    monkeypatch.setattr(
        tensor_spatial_reference,
        "prepare_hybrid_reference_loss_field",
        fake_prepare,
    )

    calibration = _call()

    assert len(calls) == 2
    assert calls[0][0] == 0.35
    assert np.isinf(calls[1][0])
    for _, kwargs in calls:
        assert kwargs["volume_axial_order"] == 6
        assert kwargs["volume_radial_order"] == 4
        assert kwargs["volume_azimuthal_order"] == 16
        assert kwargs["background_radial_order"] == 10
        assert kwargs["background_angular_order"] == 32
    assert calibration.prepared.normalized_dielectric_closure_error < 1e-6
    assert calibration.refinements == 0


def test_tensor_spatial_calibration_does_not_mask_other_runtime_errors(monkeypatch):
    def fake_prepare(*args, **kwargs):
        raise RuntimeError("unexpected tensor field failure")

    monkeypatch.setattr(
        tensor_spatial_reference,
        "prepare_hybrid_reference_loss_field",
        fake_prepare,
    )

    with pytest.raises(RuntimeError, match="unexpected tensor field failure"):
        _call()
