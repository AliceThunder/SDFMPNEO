import torch

from examples.bfzi_geom_multi import rank48_extra
from sdfmpneo_bfzi.geom_neural import GeomFracET


def test_rank48_extra_only_keeps_original_32_output_rows_frozen():
    net=GeomFracET(ng=45,rank=48,hidden=16,nrate=4,
                   soe_quad='trap',heat_dt=1.875)
    old_w=net.heat_head[2].weight.detach().clone()
    old_b=net.heat_head[2].bias.detach().clone()
    rank48_extra(net)
    assert not net.heat_head[0].weight.requires_grad
    assert net.heat_head[2].weight.requires_grad
    x=torch.randn(5,net.heat_head[0].in_features)
    net.heat_head(x).sum().backward()
    grad=net.heat_head[2].weight.grad
    bias=net.heat_head[2].bias.grad
    assert torch.count_nonzero(grad[:32])==0
    assert torch.count_nonzero(grad[32:])>0
    assert torch.count_nonzero(bias[:32])==0
    assert torch.count_nonzero(bias[32:])>0
    assert torch.equal(net.heat_head[2].weight.detach()[:32],old_w[:32])
    assert torch.equal(net.heat_head[2].bias.detach()[:32],old_b[:32])
