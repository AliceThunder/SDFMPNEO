import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.disk_dataset import DiskQuadraticJouleDataset
from sdfmpneo.electrothermal_tensor.generator import generate_snapshots_resumable


def _tensor(state):
    value = float(state[0])
    return np.array([[[1.0 + value, 0.2], [0.2, 2.0 - 0.5 * value]]])


def test_snapshot_generation_resumes_without_recomputing_completed_rows(tmp_path):
    states = np.arange(12.0)[:, None] / 10.0
    geometry = np.empty((len(states), 0))
    checkpoint = tmp_path / "partial.npz"
    sidecar = tmp_path / "partial.npz.packed.npy"
    calls = []

    def failing_factory(state, _geometry):
        index = int(round(float(state[0]) * 10.0))
        calls.append(index)
        if index == 5:
            raise RuntimeError("controlled interruption")
        return _tensor(state)

    with pytest.raises(RuntimeError):
        generate_snapshots_resumable(
            states,
            geometry,
            failing_factory,
            checkpoint_path=checkpoint,
            checkpoint_every=2,
            metadata={"physical_signature": "physics-v1"},
        )
    assert checkpoint.exists()
    assert sidecar.exists()
    with np.load(checkpoint, allow_pickle=False) as data:
        completed_before = np.asarray(data["completed"], dtype=bool)
        assert int(data["format_version"]) == 3
        assert "tensors" not in data.files
        assert int(data["output_width"]) == 3
        assert str(data["physical_signature"]) == "physics-v1"
    packed = np.load(sidecar, mmap_mode="r", allow_pickle=False)
    assert packed.shape == (len(states), 3)
    assert packed.dtype == np.float64
    assert np.count_nonzero(completed_before) > 0

    resumed_calls = []

    def good_factory(state, _geometry):
        index = int(round(float(state[0]) * 10.0))
        resumed_calls.append(index)
        return _tensor(state)

    dataset = generate_snapshots_resumable(
        states,
        geometry,
        good_factory,
        checkpoint_path=checkpoint,
        checkpoint_every=3,
        metadata={"physical_signature": "physics-v1"},
    )
    completed_ids = set(np.flatnonzero(completed_before).tolist())
    assert completed_ids.isdisjoint(resumed_calls)
    assert dataset.n_samples == len(states)


def test_snapshot_resume_rejects_changed_physical_signature(tmp_path):
    states = np.arange(8.0)[:, None] / 10.0
    geometry = np.empty((len(states), 0))
    checkpoint = tmp_path / "partial.npz"

    def failing_factory(state, _geometry):
        if float(state[0]) >= 0.3:
            raise RuntimeError("stop")
        return _tensor(state)

    with pytest.raises(RuntimeError):
        generate_snapshots_resumable(
            states,
            geometry,
            failing_factory,
            checkpoint_path=checkpoint,
            checkpoint_every=1,
            metadata={"physical_signature": "physics-v1"},
        )

    with pytest.raises(ValueError, match="physical signature differs"):
        generate_snapshots_resumable(
            states,
            geometry,
            lambda state, geometry: _tensor(state),
            checkpoint_path=checkpoint,
            metadata={"physical_signature": "physics-v2"},
        )


def test_successful_small_frozen_dataset_removes_working_checkpoint(tmp_path):
    states = np.arange(10.0)[:, None] / 10.0
    geometry = np.empty((len(states), 0))
    checkpoint = tmp_path / "partial.npz"
    sidecar = tmp_path / "partial.npz.packed.npy"
    final = tmp_path / "frozen.npz"

    dataset = generate_snapshots_resumable(
        states,
        geometry,
        lambda state, _geometry: _tensor(state),
        checkpoint_path=checkpoint,
        checkpoint_every=2,
        metadata={"physical_signature": "physics-v1"},
        final_path=final,
    )
    assert dataset.n_samples == len(states)
    assert final.exists()
    assert final.with_suffix(".json").exists()
    assert not checkpoint.exists()
    assert not sidecar.exists()


def test_large_policy_freezes_to_verified_disk_store_without_duplicate_output(tmp_path):
    states = np.arange(12.0)[:, None] / 10.0
    geometry = np.empty((len(states), 0))
    checkpoint = tmp_path / "partial.npz"
    final = tmp_path / "frozen.npz"

    dataset = generate_snapshots_resumable(
        states,
        geometry,
        lambda state, _geometry: _tensor(state),
        checkpoint_path=checkpoint,
        metadata={"physical_signature": "physics-v1"},
        final_path=final,
        disk_backed_threshold_bytes=0,
    )
    assert isinstance(dataset, DiskQuadraticJouleDataset)
    store = tmp_path / "frozen.store"
    assert store.is_dir()
    assert (store / "outputs.npy").exists()
    assert (store / "manifest.json").exists()
    assert not final.exists()
    assert not checkpoint.exists()
    assert not (tmp_path / "partial.npz.packed.npy").exists()

    loaded = DiskQuadraticJouleDataset.load(store, verify=True)
    assert loaded.n_samples == len(states)
    np.testing.assert_allclose(loaded.outputs, dataset.outputs)
    assert loaded.manifest().dataset_hash == dataset.manifest().dataset_hash
