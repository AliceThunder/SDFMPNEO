import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.generator import generate_snapshots_resumable


def _tensor(state):
    value = float(state[0])
    return np.array([[[1.0 + value, 0.2], [0.2, 2.0 - 0.5 * value]]])


def test_snapshot_generation_resumes_without_recomputing_completed_rows(tmp_path):
    states = np.arange(12.0)[:, None] / 10.0
    geometry = np.empty((len(states), 0))
    checkpoint = tmp_path / "partial.npz"
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
        )
    assert checkpoint.exists()
    with np.load(checkpoint, allow_pickle=False) as data:
        completed_before = np.asarray(data["completed"], dtype=bool)
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
    )
    completed_ids = set(np.flatnonzero(completed_before).tolist())
    assert completed_ids.isdisjoint(resumed_calls)
    assert dataset.n_samples == len(states)
