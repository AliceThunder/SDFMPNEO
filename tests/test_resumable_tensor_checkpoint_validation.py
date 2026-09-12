import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.generator import generate_snapshots_resumable


def _tensor(state):
    value = float(state[0])
    return np.array([[[1.0 + value, 0.2], [0.2, 2.0 - 0.5 * value]]])


def _interrupting_checkpoint(tmp_path):
    states = np.arange(8.0)[:, None] / 10.0
    geometry = np.empty((len(states), 0))
    checkpoint = tmp_path / "partial.npz"

    def factory(state, _geometry):
        index = int(round(float(state[0]) * 10.0))
        if index == 4:
            raise RuntimeError("controlled interruption")
        return _tensor(state)

    with pytest.raises(RuntimeError):
        generate_snapshots_resumable(
            states,
            geometry,
            factory,
            checkpoint_path=checkpoint,
            checkpoint_every=2,
            metadata={"physical_signature": "physics-v1"},
        )
    return states, geometry, checkpoint


def test_resume_archives_changed_physical_signature_and_regenerates(tmp_path):
    states, geometry, checkpoint = _interrupting_checkpoint(tmp_path)
    calls = []

    def factory(state, _geometry):
        calls.append(float(state[0]))
        return _tensor(state)

    dataset = generate_snapshots_resumable(
        states,
        geometry,
        factory,
        checkpoint_path=checkpoint,
        metadata={"physical_signature": "physics-v2"},
    )
    assert dataset.n_samples == len(states)
    assert len(calls) == len(states)
    assert any(tmp_path.glob("partial.stale-*.npz"))
    assert any(tmp_path.glob("partial.npz.packed.stale-*.npy"))


def test_resume_archives_corrupted_packed_sidecar_and_regenerates(tmp_path):
    states, geometry, checkpoint = _interrupting_checkpoint(tmp_path)
    sidecar = checkpoint.with_name(checkpoint.name + ".packed.npy")

    wrong = np.lib.format.open_memmap(
        sidecar,
        mode="w+",
        dtype=np.float64,
        shape=(len(states), 4),
    )
    wrong[:] = 0.0
    wrong.flush()
    del wrong

    calls = []

    def factory(state, _geometry):
        calls.append(float(state[0]))
        return _tensor(state)

    dataset = generate_snapshots_resumable(
        states,
        geometry,
        factory,
        checkpoint_path=checkpoint,
        metadata={"physical_signature": "physics-v1"},
    )
    assert dataset.n_samples == len(states)
    assert len(calls) == len(states)
    assert any(tmp_path.glob("partial.stale-*.npz"))
    assert any(tmp_path.glob("partial.npz.packed.stale-*.npy"))
