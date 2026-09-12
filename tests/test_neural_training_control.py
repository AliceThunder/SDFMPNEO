from types import SimpleNamespace

import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.neural_control import NeuralTrainingRuntime
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.training.monitor import TrainingMonitor, TrainingStopped, write_command


def _pod():
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


def test_neural_checkpoint_preserves_dataset_pod_network_and_optimizer(tmp_path):
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "training.pt"
    runtime = NeuralTrainingRuntime(None, checkpoint)
    runtime.pod = _pod()
    runtime.dataset_hash = "dataset-123"

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
    assert payload["format_version"] == 3
    assert payload["dataset_hash"] == "dataset-123"
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


@pytest.mark.parametrize("version", [1, 2, 3])
def test_old_checkpoint_versions_are_readable(tmp_path, version):
    torch = pytest.importorskip("torch")
    path = tmp_path / f"v{version}.pt"
    payload = {
        "format_version": version,
        "network_config": {"input_dimension": 1, "output_dimension": 1, "width": 4, "blocks": 1, "activation": "silu"},
        "network_state": None,
        "optimizer_state": None,
    }
    if version >= 2:
        pod = _pod()
        payload["pod"] = {
            "mean": pod.mean,
            "basis": pod.basis,
            "singular_values": pod.singular_values,
            "thermal_rank": pod.thermal_rank,
            "current_dimension": pod.current_dimension,
            "total_centered_energy": pod.total_centered_energy,
        }
    if version >= 3:
        payload["dataset_hash"] = "x"
    torch.save(payload, path)
    runtime = NeuralTrainingRuntime(None, path)
    loaded = runtime._load_payload()
    assert int(loaded["format_version"]) == version
