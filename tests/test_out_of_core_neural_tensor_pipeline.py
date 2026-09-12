import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.dataset import QuadraticJouleDataset
from sdfmpneo.electrothermal_tensor.disk_dataset import DiskQuadraticJouleDataset
from sdfmpneo.electrothermal_tensor.generator import generate_snapshots_resumable
from sdfmpneo.electrothermal_tensor.network import ResidualMLPConfig
from sdfmpneo.electrothermal_tensor.pod import fit_dataset_pod
from sdfmpneo.electrothermal_tensor.trainer import NeuralTrainingConfig, train_tensor_surrogate
from sdfmpneo.electrothermal_tensor.validation import validate_surrogate_on_dataset


def _tensor(state, geometry):
    a0, a1 = np.asarray(state, dtype=float)
    g0 = float(np.asarray(geometry, dtype=float)[0])
    G0 = np.array([[1.0 + 0.4 * a0, 0.15 + 0.1 * g0], [0.15 + 0.1 * g0, 0.6 + 0.2 * a1]])
    G1 = np.array([[0.2 - 0.1 * a1, -0.05 + 0.08 * a0], [-0.05 + 0.08 * a0, 0.3 + 0.1 * g0]])
    return np.stack([G0, G1])


def _disk_dataset(tmp_path):
    rng = np.random.default_rng(4)
    states = rng.uniform(-0.5, 0.5, size=(48, 2))
    geometries = rng.uniform(-1.0, 1.0, size=(48, 1))
    return generate_snapshots_resumable(
        states,
        geometries,
        _tensor,
        checkpoint_path=tmp_path / "partial.npz",
        final_path=tmp_path / "dataset.npz",
        disk_backed_threshold_bytes=0,
        split_seed=7,
        metadata={
            "physical_signature": "synthetic",
            "state_lower": [-0.5, -0.5],
            "state_upper": [0.5, 0.5],
            "geometry_lower": [-1.0],
            "geometry_upper": [1.0],
            "operating_lower": [-1.0],
            "operating_upper": [1.0],
        },
    )


def test_disk_dataset_out_of_core_pod_matches_low_rank_structure(tmp_path):
    dataset = _disk_dataset(tmp_path)
    assert isinstance(dataset, DiskQuadraticJouleDataset)
    assert isinstance(dataset.outputs, np.memmap)

    pod = fit_dataset_pod(
        dataset,
        rank=4,
        exact_svd_max_bytes=0,
        out_of_core_max_rank=4,
        chunk_rows=7,
    )
    assert pod.rank == 4
    assert 0.999999 <= pod.energy_fraction() <= 1.0

    ids = dataset.indices("validation")
    block = np.asarray(dataset.outputs[ids], dtype=float)
    reconstruction = pod.decode(pod.encode(block))
    np.testing.assert_allclose(reconstruction, block, rtol=1e-8, atol=1e-9)


def test_memmapped_dataset_trains_and_batches_gate5(tmp_path):
    pytest.importorskip("torch")
    dataset = _disk_dataset(tmp_path)
    pod = fit_dataset_pod(
        dataset,
        rank=4,
        exact_svd_max_bytes=0,
        out_of_core_max_rank=4,
        chunk_rows=8,
    )
    surrogate, report = train_tensor_surrogate(
        dataset,
        pod,
        operating_lower=np.array([-1.0]),
        operating_upper=np.array([1.0]),
        network_config=ResidualMLPConfig(
            input_dimension=3,
            output_dimension=4,
            width=24,
            blocks=2,
        ),
        training_config=NeuralTrainingConfig(
            epochs=80,
            batch_size=8,
            learning_rate=5e-3,
            weight_decay=0.0,
            heat_loss_weight=0.2,
            patience=20,
            seed=11,
            dtype="float64",
            evaluation_batch_size=8,
        ),
        device="cpu",
    )
    assert report.epochs_completed >= 1
    state = np.array([0.1, -0.2])
    geometry = np.array([0.3])
    operating = np.array([0.4])
    q = surrogate.heat_source_numpy(state, geometry, operating)
    assert q.shape == (2,)
    assert np.all(np.isfinite(q))

    validation = validate_surrogate_on_dataset(
        surrogate,
        dataset,
        operating_lower=np.array([-1.0]),
        operating_upper=np.array([1.0]),
        split="test",
        operating_samples_per_state=3,
        seed=17,
        batch_size=3,
    )
    assert validation.sample_count == len(dataset.indices("test"))
    assert np.isfinite(validation.packed_relative_rms)
    assert np.isfinite(validation.heat_relative_rms)


def test_standard_dataset_loader_dispatches_to_verified_disk_backend(tmp_path):
    dataset = _disk_dataset(tmp_path)
    loaded = QuadraticJouleDataset.load(dataset.directory)
    assert isinstance(loaded, DiskQuadraticJouleDataset)
    assert isinstance(loaded.outputs, np.memmap)
    assert loaded.manifest().dataset_hash == dataset.manifest().dataset_hash


def test_disk_dataset_integrity_check_rejects_modified_output(tmp_path):
    dataset = _disk_dataset(tmp_path)
    store = dataset.directory
    del dataset
    output = np.load(store / "outputs.npy", mmap_mode="r+", allow_pickle=False)
    output[0, 0] += 1.0
    output.flush()
    del output
    with pytest.raises(ValueError, match="file hash mismatch"):
        DiskQuadraticJouleDataset.load(store, verify=True)
