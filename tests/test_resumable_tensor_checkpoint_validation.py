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


def test_resume_rejects_changed_physical_signature(tmp_path):
    states, geometry, checkpoint = _interrupting_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="physical signature differs"):
        generate_snapshots_resumable(
            states,
            geometry,
            lambda state, _geometry: _tensor(state),
            checkpoint_path=checkpoint,
            metadata={"physical_signature": "physics-v2"},
        )


def test_resume_rejects_corrupted_tensor_sidecar_shape(tmp_path):
    states, geometry, checkpoint = _interrupting_checkpoint(tmp_path)
    sidecar = checkpoint.with_name(checkpoint.name + ".tensors.npy")

    # Replace the NPY sidecar with a valid NPY file of the wrong shape.  The
    # checkpoint bitmap/hashes remain untouched, so resume must detect the
    # sidecar header mismatch before evaluating any new physics snapshot.
    wrong = np.lib.format.open_memmap(
        sidecar,
        mode="w+",
        dtype=np.float64,
        shape=(len(states), 1, 3, 3),
    )
    wrong[:] = 0.0
    wrong.flush()
    del wrong

    calls = []

    def factory(state, _geometry):
        calls.append(float(state[0]))
        return _tensor(state)

    with pytest.raises(ValueError, match="sidecar shape/dtype is malformed"):
        generate_snapshots_resumable(
            states,
            geometry,
            factory,
            checkpoint_path=checkpoint,
            metadata={"physical_signature": "physics-v1"},
        )
    assert calls == []
