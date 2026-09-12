import numpy as np
import pytest

from sdfmpneo.electrothermal_tensor.physical_layer import (
    decode_heat_source_numpy,
    decode_heat_source_torch,
)
from sdfmpneo.electrothermal_tensor.pod import TensorPOD
from sdfmpneo.electrothermal_tensor.symmetric import tensor_smat


def _pod():
    # r=2 thermal modes, one real current => p=2, n_sym=3, total width=6.
    mean = np.array([1.0, 0.2, 2.0, -0.4, 0.3, 0.8])
    direction = np.arange(1.0, 7.0)
    direction /= np.linalg.norm(direction)
    return TensorPOD(
        mean=mean,
        basis=direction[:, None],
        singular_values=np.array([2.0]),
        thermal_rank=2,
        current_dimension=1,
    )


def test_numpy_quadratic_decoder_matches_dense_tensor():
    pod = _pod()
    beta = np.array([0.7])
    u = np.array([1.3])
    packed = pod.decode(beta).reshape(2, 3)
    dense = tensor_smat(packed, 2)
    zeta = np.array([1.0, u[0]])
    expected = np.einsum("i,kij,j->k", zeta, dense, zeta)
    np.testing.assert_allclose(
        decode_heat_source_numpy(beta, pod, u),
        expected,
        rtol=2e-15,
        atol=2e-15,
    )


def test_torch_quadratic_decoder_matches_numpy_and_has_gradient():
    torch = pytest.importorskip("torch")
    pod = _pod()
    beta = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
    mean = torch.tensor(pod.mean, dtype=torch.float64)
    basis = torch.tensor(pod.basis, dtype=torch.float64)
    u = torch.tensor([0.6], dtype=torch.float64)
    value = decode_heat_source_torch(beta, mean, basis, pod.thermal_rank, u)
    expected = decode_heat_source_numpy(np.array([0.4]), pod, np.array([0.6]))
    np.testing.assert_allclose(value.detach().numpy(), expected, rtol=2e-14, atol=2e-14)
    value.sum().backward()
    assert beta.grad is not None
    assert torch.all(torch.isfinite(beta.grad))
