import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.dataset import QuadraticJouleDataset
from sdfmpneo.electrothermal_tensor.network import ResidualMLPConfig
from sdfmpneo.electrothermal_tensor.pod import fit_dataset_pod
from sdfmpneo.electrothermal_tensor.symmetric import tensor_smat
from sdfmpneo.electrothermal_tensor.trainer import NeuralTrainingConfig, train_tensor_surrogate


def test_standard_neural_training_loop_runs_without_physics_solver():
    pytest.importorskip("torch")
    rng = np.random.default_rng(22)
    n = 36
    states = rng.uniform(-1.0, 1.0, size=(n, 1))
    geometry = np.empty((n, 0))
    # One thermal mode, one current: packed symmetric width=3.  The tensor map
    # is affine in state, so rank-1 POD captures it exactly after centering.
    packed = np.column_stack([
        1.0 + 0.5 * states[:, 0],
        0.2 + 0.1 * states[:, 0],
        2.0 - 0.25 * states[:, 0],
    ])[:, None, :]
    tensors = tensor_smat(packed, 2)
    dataset = QuadraticJouleDataset.from_tensors(states, geometry, tensors, split_seed=9)
    pod = fit_dataset_pod(dataset, rank=1)
    surrogate, report = train_tensor_surrogate(
        dataset,
        pod,
        operating_lower=np.array([-1.0]),
        operating_upper=np.array([1.0]),
        network_config=ResidualMLPConfig(
            input_dimension=1,
            output_dimension=1,
            width=16,
            blocks=1,
        ),
        training_config=NeuralTrainingConfig(
            epochs=6,
            batch_size=12,
            learning_rate=2e-3,
            patience=6,
            heat_loss_weight=0.1,
            dtype="float64",
            seed=3,
        ),
        device="cpu",
    )
    assert report.best_epoch >= 1
    assert np.isfinite(report.best_validation_loss)
    q = surrogate.heat_source_numpy(np.array([0.1]), np.empty(0), np.array([0.4]))
    assert q.shape == (1,)
    assert np.all(np.isfinite(q))
