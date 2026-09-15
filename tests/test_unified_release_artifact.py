import json

import numpy as np
import pytest

from sdfmpneo.unified_runtime import _require_release_artifact


_SELF_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"


def _write_metadata(path, release):
    meta = {"metadata": release}
    np.savez_compressed(path, metadata_json=np.array(json.dumps(meta, sort_keys=True)))


def _certified_release():
    return {
        "truth_preflight": {
            "certified": True,
            "local_self_correction_converged": True,
        },
        "physics_gate": {
            "certified": True,
            "self_correction_model": _SELF_MODEL,
        },
        "final_held_out_audit": {
            "certified": True,
            "production_integrator_ok": True,
            "certificate_level": "frozen_held_out_numerical_validation",
        },
    }


def test_prediction_release_gate_accepts_certified_runtime_artifact(tmp_path):
    path = tmp_path / "model.npz"
    release = _certified_release()
    _write_metadata(path, release)
    assert _require_release_artifact(path) == release


def test_prediction_release_gate_rejects_missing_integrator_audit(tmp_path):
    path = tmp_path / "model.npz"
    release = _certified_release()
    release["final_held_out_audit"]["production_integrator_ok"] = False
    _write_metadata(path, release)
    with pytest.raises(ValueError, match="production_integrator"):
        _require_release_artifact(path)


def test_prediction_release_gate_rejects_missing_self_correction_certificate(tmp_path):
    path = tmp_path / "model.npz"
    release = _certified_release()
    release["truth_preflight"]["local_self_correction_converged"] = False
    _write_metadata(path, release)
    with pytest.raises(ValueError, match="local_self_correction"):
        _require_release_artifact(path)


def test_prediction_release_gate_rejects_wrong_self_correction_model(tmp_path):
    path = tmp_path / "model.npz"
    release = _certified_release()
    release["physics_gate"]["self_correction_model"] = "canonical_local_fine_minus_coarse_self_defect_v1"
    _write_metadata(path, release)
    with pytest.raises(ValueError, match="self_correction_model"):
        _require_release_artifact(path)


def test_prediction_release_gate_rejects_library_only_artifact(tmp_path):
    path = tmp_path / "model.npz"
    _write_metadata(path, {})
    with pytest.raises(ValueError, match="not a certified production release"):
        _require_release_artifact(path)
