import numpy as np

from sdfmpneo.electrothermal_tensor.dataset import QuadraticJouleDataset, frozen_split_indices
from sdfmpneo.electrothermal_tensor.pod import fit_dataset_pod
from sdfmpneo.electrothermal_tensor.symmetric import smat, svec, tensor_smat, tensor_svec


def test_svec_is_frobenius_isometry():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(5, 4, 4))
    B = rng.normal(size=(5, 4, 4))
    A = 0.5 * (A + np.swapaxes(A, -1, -2))
    B = 0.5 * (B + np.swapaxes(B, -1, -2))
    np.testing.assert_allclose(
        np.sum(svec(A) * svec(B), axis=-1),
        np.einsum("bij,bij->b", A, B),
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_allclose(smat(svec(A), 4), A, rtol=0.0, atol=0.0)


def test_dataset_roundtrip_and_pod_uses_training_split(tmp_path):
    rng = np.random.default_rng(8)
    n, r, p = 40, 3, 4
    states = rng.normal(size=(n, r))
    geometries = rng.uniform(-1, 1, size=(n, 2))

    # Synthetic tensors live in a two-dimensional affine subspace in svec
    # coordinates, so a rank-2 POD must reconstruct them to roundoff.
    n_sym = p * (p + 1) // 2
    mean = rng.normal(size=(r, n_sym))
    directions = rng.normal(size=(2, r, n_sym))
    coeff = rng.normal(size=(n, 2))
    packed = mean[None] + np.einsum("nk,krs->nrs", coeff, directions)
    tensors = tensor_smat(packed, p)

    dataset = QuadraticJouleDataset.from_tensors(
        states,
        geometries,
        tensors,
        split_seed=17,
        metadata={"physics_revision": "unit-test"},
    )
    first_split = dataset.split.copy()
    np.testing.assert_array_equal(
        first_split,
        frozen_split_indices(n, seed=17),
    )
    np.testing.assert_allclose(
        dataset.outputs.reshape(n, r, n_sym),
        tensor_svec(tensors),
        rtol=0.0,
        atol=0.0,
    )

    npz, _ = dataset.save(tmp_path / "snapshots")
    loaded = QuadraticJouleDataset.load(npz)
    np.testing.assert_array_equal(loaded.split, dataset.split)
    np.testing.assert_allclose(loaded.outputs, dataset.outputs)
    assert loaded.manifest().dataset_hash == dataset.manifest().dataset_hash

    pod = fit_dataset_pod(loaded, rank=2)
    train_ids = loaded.indices("train")
    np.testing.assert_allclose(
        pod.decode(pod.encode(loaded.outputs[train_ids])),
        loaded.outputs[train_ids],
        rtol=2e-12,
        atol=2e-12,
    )
    assert pod.rank == 2
