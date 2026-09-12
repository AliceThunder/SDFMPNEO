import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.dataset import QuadraticJouleDataset
from sdfmpneo.electrothermal_tensor.network import ResidualMLPConfig
from sdfmpneo.electrothermal_tensor.pod import fit_dataset_pod
from sdfmpneo.electrothermal_tensor.symmetric import tensor_smat
from sdfmpneo.electrothermal_tensor.trainer import NeuralTrainingConfig, train_tensor_surrogate


def _small_dataset():
    rng = np.random.default_rng(22)
    n = 36
    states = rng.uniform(-1.0, 1.0, size=(n, 1))
    geometry = np.empty((n, 0))
    packed = np.column_stack([
        1.0 + 0.5 * states[:, 0],
        0.2 + 0.1 * states[:, 0],
        2.0 - 0.25 * states[:, 0],
    ])[:, None, :]
    tensors = tensor_smat(packed, 2)
    dataset = QuadraticJouleDataset.from_tensors(
        states,
        geometry,
        tensors,
        split_seed=9,
        metadata={
            "operating_lower": [-1.0],
            "operating_upper": [1.0],
        },
    )
    return dataset, fit_dataset_pod(dataset, rank=1)


def _training_config():
    return NeuralTrainingConfig(
        epochs=6,
        batch_size=12,
        learning_rate=2e-3,
        patience=6,
        heat_loss_weight=0.1,
        dtype="float64",
        seed=3,
    )


def _network_config():
    return ResidualMLPConfig(
        input_dimension=1,
        output_dimension=1,
        width=16,
        blocks=1,
    )


def _train(dataset, pod):
    return train_tensor_surrogate(
        dataset,
        pod,
        operating_lower=np.array([-1.0]),
        operating_upper=np.array([1.0]),
        network_config=_network_config(),
        training_config=_training_config(),
        device="cpu",
    )


def test_standard_neural_training_loop_runs_without_physics_solver():
    pytest.importorskip("torch")
    dataset, pod = _small_dataset()
    surrogate, report = _train(dataset, pod)
    assert report.best_epoch >= 1
    assert np.isfinite(report.best_validation_loss)
    q = surrogate.heat_source_numpy(np.array([0.1]), np.empty(0), np.array([0.4]))
    assert q.shape == (1,)
    assert np.all(np.isfinite(q))


def test_same_seed_reproduces_cpu_training_outputs():
    pytest.importorskip("torch")
    dataset, pod = _small_dataset()
    first, first_report = _train(dataset, pod)
    second, second_report = _train(dataset, pod)

    test_ids = dataset.indices("test")
    states = dataset.states[test_ids]
    geometry = dataset.geometries[test_ids]
    first_beta = first.predict_coefficients_batch_numpy(states, geometry)
    second_beta = second.predict_coefficients_batch_numpy(states, geometry)
    first_packed = pod.mean[None, :] + first_beta @ pod.basis.T
    second_packed = pod.mean[None, :] + second_beta @ pod.basis.T
    np.testing.assert_allclose(second_packed, first_packed, rtol=0.0, atol=0.0)

    operating = np.full((len(test_ids), 1), 0.37)
    np.testing.assert_allclose(
        second.heat_source_batch_numpy(states, geometry, operating),
        first.heat_source_batch_numpy(states, geometry, operating),
        rtol=0.0,
        atol=0.0,
    )
    assert second_report.best_epoch == first_report.best_epoch
    assert second_report.epochs_completed == first_report.epochs_completed
