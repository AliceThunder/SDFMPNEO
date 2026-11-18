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

def test_nested_geometry_free_tangent_checkpoint(tmp_path):
    import numpy as np
    from examples.bfzi_z_tangent import IDX,Tangent
    from sdfmpneo_bfzi.geom_ztune import restore

    base=tmp_path/'base.pt'
    first=tmp_path/'first.pt'
    second=tmp_path/'second.pt'
    tan=Tangent(ng=45,nmode=len(IDX))
    sa=np.ones(32)
    sy=np.ones(14)
    torch.save(dict(state=tan.state_dict(),ng=45,idx=IDX,sa=torch.as_tensor(sa),sy=torch.as_tensor(sy)),base)
    one=ZFix(ng=45,sa=sa)
    torch.save(dict(state=one.state_dict(),ng=45,sa=torch.as_tensor(sa),sy=torch.as_tensor(sy),
                    base=str(base)),first)
    two=ZFix(ng=45,sa=sa)
    torch.save(dict(state=two.state_dict(),ng=45,sa=torch.as_tensor(sa),sy=torch.as_tensor(sy),
                    base=str(first),base_mode='ztune'),second)
    predict=restore(second)
    g=torch.zeros((2,45))
    a=torch.zeros((2,32))
    out=predict(g,a)
    assert out.shape==(2,6)
    assert torch.isfinite(out).all()
    assert torch.allclose(out,torch.zeros_like(out))
