import numpy as np
import pytest
from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph, evaluate_parametric_stable
from sdfmpneo.analytic.parametric_realization import evaluate_parametric_stable_with_jacobians
from sdfmpneo.training.research import ResearchTrainingConfig


def test_stationary_limit_extreme_finite_times_and_weight_sensitivities():
    graph=ParametricAnalyticEvolutionGraph([.001,.001],['u'])
    graph.add_product_response('first',0,['u'],.2)
    graph.add_product_response('second',1,['first','u'],.3)
    for t in [0.,1e-8,1.,1e3,1e5,1e12,1e300,np.finfo(float).max,np.inf]:
        a,da,ja,jda=evaluate_parametric_stable_with_jacobians(
            graph,t,a0=[0.,0.],operating=[2.],weight_derivatives=True)
        e=np.exp(-.001*t)
        te=0. if e==0 else t*e
        expected=np.array([400*(1-e),240000*(1-e)-240*te])
        assert np.allclose(a,expected,rtol=1e-10,atol=1e-8)
        assert np.all(np.isfinite(da))
        assert np.all(np.isfinite(ja))
        if t>=1e12:
            assert np.allclose(da,0,atol=1e-10)
            assert np.allclose(ja,[[2000,0],[1200000,800000]],rtol=1e-10)
            assert np.allclose(jda,0,atol=1e-10)
    with pytest.raises(ValueError):
        evaluate_parametric_stable(graph,np.nan,a0=[0.,0.],operating=[2.])


def test_mixed_time_sampling_keeps_early_late_and_infinite_checks_independent():
    c=ResearchTrainingConfig((0.,),(1.,),(0.,),(1.,),1e5,1e-5,
             sample_count=64,validation_count=64,time_sampling='mixed_log',include_steady_state=True)
    p,c1,c2=c.points(),c.points(True),c.points(True,seed=3)
    assert np.any(p[:,-1]==0)
    assert np.count_nonzero((p[:,-1]>0)&(p[:,-1]<.01))>5
    assert np.count_nonzero(np.isfinite(p[:,-1])&(p[:,-1]>1e4))>10
    assert np.isposinf(p[:,-1]).sum()==64
    assert np.isposinf(c2[:,-1]).sum()==64
    assert not np.array_equal(c1,c2)
