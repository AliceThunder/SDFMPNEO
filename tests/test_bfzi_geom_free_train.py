"""Local regression tests for BFZI-free differentiable physical operators."""
import numpy as np
import torch

from sdfmpneo_bfzi.geom_free_train import _cu, _v6, _z6
from sdfmpneo_bfzi.geom_memory import geom_port


def test_complex_port_symmetric_roundtrip():
    z=torch.tensor([1.,2.,3.,4.,5.,6.])
    got=_z6(_v6(z))
    assert torch.equal(got,z)


def test_exact_copper_forward_and_gradient():
    g=np.array([.08885,.08885,.04,.04,.9,.9,10.,10.,
                .000790569415,.000790569415,.04,
                20e-6,10e-6,20e-6,40.,40.])
    port=geom_port(g)
    temp=torch.tensor([300.,320.],dtype=torch.float64,requires_grad=True)
    out=_cu(port,temp)
    np.testing.assert_allclose(out.detach().numpy(),
                               port.copper(temp.detach().numpy()),
                               rtol=1e-12,atol=1e-12)
    grad=torch.autograd.grad(out.real.sum(),temp)[0].numpy()
    t=temp.detach().numpy()
    eps=.01
    expected=((port.copper(t+eps)-port.copper(t-eps))/(2*eps)).real
    np.testing.assert_allclose(grad,expected,rtol=1e-8,atol=1e-10)
