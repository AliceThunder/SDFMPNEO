"""Causal residual electromagnetic feedback at zero/cold and heated state."""
import numpy as np
import torch
from sdfmpneo_bfzi.geom_dyn import DynFix


def test_zero_heat_and_finite_hot_dynamic_z():
    model=DynFix(np.zeros(18),np.ones(18),np.ones(32),
                 np.full(6,.001))
    g=torch.randn(3,18)
    a=torch.zeros(3,32)
    t=torch.tensor([[0.],[10.],[600.]])
    v=torch.tensor([[0.],[80.],[100.]])
    assert torch.count_nonzero(model(g,a,t,v))==0
    with torch.no_grad():
        model.net[-1].weight.fill_(.1)
    a[:,1]=torch.tensor([1.,2.,3.])
    assert model(g,a,t,v).shape==(3,6)
    assert torch.isfinite(model(g,a,t,v)).all()
    assert torch.count_nonzero(model(g,torch.zeros_like(a),t,v))==0
