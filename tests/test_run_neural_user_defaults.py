import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_run():
    path = Path(__file__).resolve().parents[1] / "run.py"
    spec = importlib.util.spec_from_file_location("sdfmpneo_run_neural_defaults", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_defaults_use_cuda_and_separate_state_domain():
    run = _load_run()
    assert run.TRAINING["device"] == "cuda"
    assert "state_lower" in run.TRAINING
    assert "state_upper" in run.TRAINING
    assert run.TRAINING["state_lower"] != run.TRAINING["initial_lower"]
    assert run.PREDICTION["allow_extrapolation"] is True
    assert run.TRAINING["optimizer"]["mixed_precision"] is True
    assert run.TRAINING["optimizer"]["validation_interval"] >= 1


def test_temperature_summary_is_physical_not_modal_norm():
    run = _load_run()
    model = SimpleNamespace(
        thermal_operators=SimpleNamespace(
            thermal_basis=np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
        )
    )
    summary = run._temperature_summary(model, np.array([0.5, 0.25]))
    assert summary is not None
    assert np.isclose(summary["maximum_temperature_rise"], 0.5)
    assert np.isclose(summary["maximum_temperature"], run.PHYSICS["ambient_temperature"] + 0.5)
    assert "maximum_temperature_celsius" in summary
