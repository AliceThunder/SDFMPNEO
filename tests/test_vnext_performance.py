import numpy as np
import pytest


torch = pytest.importorskip("torch")

from sdfmpneo_vnext.device import resolve_torch_device
from sdfmpneo_vnext.hybrid_neural import (
    HybridNormalizer,
    HybridPhysicsFactoredResidualNet,
)
from sdfmpneo_vnext.hybrid_spatial_neural import HybridSpatialLossShapeNet
from sdfmpneo_vnext.performance import forward_structured_batch
from sdfmpneo_vnext.spatial_performance import (
    _background_raw_batched,
    _conductor_raw_batched,
    _package_raw_batched,
)
from sdfmpneo_vnext.tensor_artifact_io import load_tensor_port_artifact
from sdfmpneo_vnext.tensor_features import (
    TENSOR_COIL_FEATURE_DIM,
    TENSOR_CROSS_FEATURE_DIM,
    TENSOR_PACKAGE_FEATURE_DIM,
    TENSOR_PAIR_FEATURE_DIM,
)
from sdfmpneo_vnext.tensor_neural import TensorHybridNeuralResidualArtifact


def _spd(batch, n, *, dtype):
    raw = torch.randn(batch, n, n, dtype=dtype)
    eye = torch.eye(n, dtype=dtype)[None]
    return raw @ raw.transpose(-1, -2) + 0.25 * eye


def test_true_batched_port_forward_matches_scalar_forward():
    torch.manual_seed(7)
    dtype = torch.float64
    batch = 3
    n_coils = 2
    n_packages = 1
    model = HybridPhysicsFactoredResidualNet(
        coil_dim=6,
        coil_pair_dim=4,
        package_dim=5,
        cross_dim=3,
        package_pair_dim=4,
        hidden_dim=12,
        factor_rank=2,
        depth=1,
    ).to(dtype=dtype)
    model.eval()

    coil = torch.randn(batch, n_coils, 6, dtype=dtype)
    coil_pair = torch.randn(batch, n_coils, n_coils, 4, dtype=dtype)
    package = torch.randn(batch, n_packages, 5, dtype=dtype)
    cross = torch.randn(batch, n_coils, n_packages, 3, dtype=dtype)
    package_pair = torch.randn(batch, n_packages, n_packages, 4, dtype=dtype)
    baseline_r = _spd(batch, n_coils, dtype=dtype)
    raw_x = torch.randn(batch, n_coils, n_coils, dtype=dtype)
    baseline_x = 0.5 * (raw_x + raw_x.transpose(-1, -2))
    dielectric_gate = torch.tensor([1.0, 0.0, 1.0], dtype=dtype)
    reactance_gate = torch.tensor([1.0, 1.0, 0.0], dtype=dtype)

    batched = forward_structured_batch(
        model,
        coil,
        coil_pair,
        package,
        cross,
        package_pair,
        baseline_r,
        baseline_x,
        resistance_scale=0.7,
        reactance_scale=0.9,
        dielectric_loss_gate=dielectric_gate,
        reactance_gate=reactance_gate,
    )

    for index in range(batch):
        scalar = model.forward_structured(
            coil[index],
            coil_pair[index],
            package[index],
            cross[index],
            package_pair[index],
            baseline_r[index],
            baseline_x[index],
            resistance_scale=0.7,
            reactance_scale=0.9,
            dielectric_loss_gate=float(dielectric_gate[index]),
            reactance_gate=float(reactance_gate[index]),
        )
        for batch_value, scalar_value in zip(batched, scalar):
            assert torch.allclose(
                batch_value[index],
                scalar_value,
                rtol=2e-11,
                atol=2e-11,
            )


def test_batched_spatial_heads_match_scalar_heads():
    torch.manual_seed(11)
    dtype = torch.float64
    batch = 2
    n_coils = 2
    n_packages = 2
    hidden = 8
    pair_dim = 4
    cross_dim = 3
    model = HybridSpatialLossShapeNet(
        hidden,
        pair_dim,
        cross_dim,
        field_hidden_dim=10,
        factor_rank=2,
        depth=1,
    ).to(dtype=dtype)
    model.eval()

    coil_latent = torch.randn(batch, n_coils, hidden, dtype=dtype)
    package_latent = torch.randn(batch, n_packages, hidden, dtype=dtype)
    coil_pair = torch.randn(batch, n_coils, n_coils, pair_dim, dtype=dtype)
    coil_package = torch.randn(batch, n_coils, n_packages, cross_dim, dtype=dtype)

    batch_index = np.asarray([0, 0, 1, 1], dtype=int)
    coil_index = np.asarray([0, 1, 1, 0], dtype=int)
    conductor_coordinates = np.random.default_rng(3).normal(size=(4, 4))
    conductor_batch = _conductor_raw_batched(
        model.conductor,
        coil_latent,
        coil_pair,
        batch_index,
        coil_index,
        conductor_coordinates,
    )
    for point in range(4):
        scalar = model.conductor.raw_matrices(
            coil_latent[batch_index[point]],
            coil_pair[batch_index[point]],
            np.asarray([coil_index[point]]),
            conductor_coordinates[point : point + 1],
        )[0]
        assert torch.allclose(conductor_batch[point], scalar, rtol=2e-11, atol=2e-11)

    package_index = np.asarray([0, 1, 1, 0], dtype=int)
    package_coordinates = np.random.default_rng(4).normal(size=(4, 5))
    package_batch = _package_raw_batched(
        model.package,
        coil_latent,
        package_latent,
        coil_package,
        batch_index,
        package_index,
        package_coordinates,
    )
    for point in range(4):
        scalar = model.package.raw_matrices(
            coil_latent[batch_index[point]],
            package_latent[batch_index[point]],
            coil_package[batch_index[point]],
            np.asarray([package_index[point]]),
            package_coordinates[point : point + 1],
        )[0]
        assert torch.allclose(package_batch[point], scalar, rtol=2e-11, atol=2e-11)

    coil_coordinates = np.random.default_rng(5).normal(size=(4, n_coils, 5))
    coil_coordinates[:, :, 3] = np.abs(coil_coordinates[:, :, 3]) + 0.2
    background_package_coordinates = np.random.default_rng(6).normal(
        size=(4, n_packages, 5)
    )
    background_batch = _background_raw_batched(
        model.background,
        coil_latent,
        package_latent,
        batch_index,
        coil_coordinates,
        background_package_coordinates,
    )
    for point in range(4):
        scalar = model.background.raw_matrices(
            coil_latent[batch_index[point]],
            package_latent[batch_index[point]],
            coil_coordinates[point : point + 1],
            background_package_coordinates[point : point + 1],
        )[0]
        assert torch.allclose(background_batch[point], scalar, rtol=2e-11, atol=2e-11)


def test_tensor_port_loader_preserves_float64_dtype(tmp_path):
    def zeros(size):
        return np.zeros(size, dtype=float)

    def ones(size):
        return np.ones(size, dtype=float)

    normalizer = HybridNormalizer(
        coil_node_mean=zeros(TENSOR_COIL_FEATURE_DIM),
        coil_node_scale=ones(TENSOR_COIL_FEATURE_DIM),
        coil_pair_mean=zeros(TENSOR_PAIR_FEATURE_DIM),
        coil_pair_scale=ones(TENSOR_PAIR_FEATURE_DIM),
        package_mean=zeros(TENSOR_PACKAGE_FEATURE_DIM),
        package_scale=ones(TENSOR_PACKAGE_FEATURE_DIM),
        coil_package_mean=zeros(TENSOR_CROSS_FEATURE_DIM),
        coil_package_scale=ones(TENSOR_CROSS_FEATURE_DIM),
        package_pair_mean=zeros(TENSOR_PAIR_FEATURE_DIM),
        package_pair_scale=ones(TENSOR_PAIR_FEATURE_DIM),
        resistance_scale=1.0,
        reactance_scale=1.0,
    )
    model = HybridPhysicsFactoredResidualNet(
        coil_dim=TENSOR_COIL_FEATURE_DIM,
        coil_pair_dim=TENSOR_PAIR_FEATURE_DIM,
        package_dim=TENSOR_PACKAGE_FEATURE_DIM,
        cross_dim=TENSOR_CROSS_FEATURE_DIM,
        package_pair_dim=TENSOR_PAIR_FEATURE_DIM,
        hidden_dim=8,
        factor_rank=2,
        depth=1,
    ).double()
    artifact = TensorHybridNeuralResidualArtifact(
        model,
        normalizer,
        baseline_segments=8,
        material_domain={},
        device="cpu",
    )
    path = tmp_path / "tensor-double.pt"
    artifact.save(path)
    loaded = load_tensor_port_artifact(path, device="cpu")
    assert next(loaded.model.parameters()).dtype == torch.float64
    for key, value in artifact.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])


def test_auto_device_resolves_to_supported_backend():
    assert resolve_torch_device("cpu") == "cpu"
    resolved = resolve_torch_device("auto")
    assert resolved == "cpu" or resolved == "mps" or resolved.startswith("cuda")
