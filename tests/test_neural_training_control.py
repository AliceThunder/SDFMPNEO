import json
from types import SimpleNamespace

import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.neural_control import NeuralTrainingRuntime
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.training.monitor import TrainingMonitor, TrainingStopped, write_command


def _pod():
    # thermal_rank=1, current_dimension=1 -> packed symmetric width=3
    return TensorPOD(
        mean=np.zeros(3),
        basis=np.eye(3)[:, :2],
        singular_values=np.array([2.0, 1.0]),
        thermal_rank=1,
        current_dimension=1,
        total_centered_energy=5.0,
    )


def test_monitor_pause_resume_and_stop(tmp_path):
    control = tmp_path / "control.json"
    metrics = tmp_path / "metrics.jsonl"
    write_command(control, "run")
    with TrainingMonitor(metrics, control, interval=0.01) as monitor:
        monitor.checkpoint()
        write_command(control, "stop")
        with pytest.raises(TrainingStopped):
            monitor.checkpoint()


def test_neural_checkpoint_preserves_pod_network_and_optimizer(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "training.pt"
    runtime = NeuralTrainingRuntime(None, checkpoint)
    runtime.pod = _pod()

    model = torch.nn.Linear(2, 2).double()
    model.config = SimpleNamespace(to_dict=lambda: {
        "input_dimension": 2,
        "output_dimension": 2,
        "width": 4,
        "blocks": 1,
        "activation": "silu",
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    runtime.optimizer = optimizer
    runtime._save_training_state(model)

    restored = NeuralTrainingRuntime(None, checkpoint)
    payload = restored._load_payload()
    assert payload["format_version"] == 2
    saved_pod = restored._saved_pod()
    assert saved_pod is not None
    assert np.array_equal(saved_pod.mean, runtime.pod.mean)
    assert np.array_equal(saved_pod.basis, runtime.pod.basis)
    assert saved_pod.rank == runtime.pod.rank
    assert payload["optimizer_state"] is not None
    assert set(payload["network_state"]) == set(model.state_dict())


def test_initial_pod_is_used_when_no_training_checkpoint(tmp_path):
    pod = _pod()
    runtime = NeuralTrainingRuntime(None, tmp_path / "missing.pt", initial_pod=pod)
    assert runtime._saved_pod() is pod
