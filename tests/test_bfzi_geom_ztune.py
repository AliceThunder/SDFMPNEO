"""Nonlinear sea-impedance residual preserves cold physical port."""
import torch

from sdfmpneo_bfzi.geom_ztune import ZFix


def test_zero_cold_and_smooth_residual():
    torch.manual_seed(7)
    net=ZFix(ng=45,sa=torch.ones(32).numpy())
    g=torch.randn(4,45)
    a=torch.zeros(4,32)
    assert torch.count_nonzero(net(g,a))==0
    with torch.no_grad():
        net.net[-1].weight.fill_(.01)
    y=net(g,torch.rand(4,32))
    assert y.shape==(4,6)
    assert torch.isfinite(y).all()
    assert torch.count_nonzero(net(g,a))==0
