import time
from types import SimpleNamespace
import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from sdfmpneo.training.research import ResearchTrainingConfig, make_fixed_network, train_research_network


def test_depth_one_parameter_initial_operating_jacobians():
    net = FixedAnalyticResponseNetwork(
        [1.3, 2.1], ["u"], max_response_time=2.0,
        depth=1, channels_per_mode=1, linear_rank=2, hidden_rank=1,
        quadratic_rank=2, square_rank=2, cross_rank=1, state_rank=1,
    )
    a0 = np.array([0.2, -0.1]); u = np.array([0.3]); t = 0.6
    _, _, ja, jda = net.evaluate_parameter_jacobian(t, a0=a0, operating=u)
    rng = np.random.default_rng(3)
    for idx in rng.choice(net.parameter_count, size=min(20, net.parameter_count), replace=False):
        eps = 1e-7
        p = net.parameters.copy(); p[idx] += eps
        ap, dap = net.with_parameters(p).evaluate(t, a0=a0, operating=u)
        p[idx] -= 2 * eps
        am, dam = net.with_parameters(p).evaluate(t, a0=a0, operating=u)
        assert np.allclose(ja[:, idx], (ap-am)/(2*eps), rtol=2e-5, atol=2e-8)
        assert np.allclose(jda[:, idx], (dap-dam)/(2*eps), rtol=2e-5, atol=2e-8)
    _, _, ji, jdi = net.evaluate_initial_jacobian(t, a0=a0, operating=u)
    for i in range(2):
        eps=1e-7; e=np.zeros(2); e[i]=eps
        ap,dap=net.evaluate(t,a0=a0+e,operating=u); am,dam=net.evaluate(t,a0=a0-e,operating=u)
        assert np.allclose(ji[:,i],(ap-am)/(2*eps),rtol=2e-5,atol=2e-8)
        assert np.allclose(jdi[:,i],(dap-dam)/(2*eps),rtol=2e-5,atol=2e-8)
    _, _, jo, jdo = net.evaluate_operating_jacobian(t, a0=a0, operating=u)
    eps=1e-7
    ap,dap=net.evaluate(t,a0=a0,operating=u+eps); am,dam=net.evaluate(t,a0=a0,operating=u-eps)
    assert np.allclose(jo[:,0],(ap-am)/(2*eps),rtol=2e-5,atol=2e-8)
    assert np.allclose(jdo[:,0],(dap-dam)/(2*eps),rtol=2e-5,atol=2e-8)


def test_near_equal_poles_are_stable():
    net = FixedAnalyticResponseNetwork(
        [1.0, 1.0 + 1e-12], [], max_response_time=100.0,
        depth=1, channels_per_mode=1, linear_rank=1, hidden_rank=1,
        quadratic_rank=1, square_rank=1, cross_rank=1, state_rank=1,
    )
    response, _ = net._response_kernel(50.0, net.lambdas)
    expected = 50.0 * np.exp(-50.0)
    assert np.all(np.isfinite(response))
    assert np.allclose(response, expected, rtol=2e-9, atol=0.0)


def test_modewise_square_restores_nonlinear_initial_vector_field():
    net = FixedAnalyticResponseNetwork(
        [2.0], [], max_response_time=1.0,
        depth=1, channels_per_mode=1, linear_rank=1, hidden_rank=1,
        quadratic_rank=1, square_rank=1, cross_rank=1, state_rank=1,
    )
    p = np.zeros(net.parameter_count)
    p[net._indices("square_in")[0,0]] = 1.0
    p[net._indices("square_out")[0,0]] = 0.15
    p[net._indices("square_gate")[0]] = 1.0
    p[net._indices("channel_gate")[0,0]] = 1.0
    net = net.with_parameters(p)
    for a0 in (-0.4, 0.2, 1.3):
        a, da = net.evaluate(0.0, a0=[a0], operating=[])
        assert np.allclose(a, [a0])
        assert np.allclose(da, [-2*a0 + 0.15*a0*a0])


def test_high_rank_default_capacity_is_linear_and_fast_enough_for_smoke():
    class Field:
        thermal_model = SimpleNamespace(lambdas=np.linspace(0.01, 5.0, 198))
    cfg = ResearchTrainingConfig(
        initial_lower=(-0.1,)*198, initial_upper=(0.1,)*198,
        operating_lower=(-1.0,)*12, operating_upper=(1.0,)*12,
        max_response_time=100.0, residual_tolerance=1e-5,
        sample_count=2, validation_count=2,
        semigroup_sample_count=0, semigroup_validation_count=0,
        prune_rounds=0,
    )
    net = make_fixed_network(Field(), cfg, operating_names=[f"u{i}" for i in range(12)])
    assert net.depth == 1
    assert net.channels_per_mode == 1
    assert len(net.response_nodes) == 198
    assert net.parameter_count < 10000
    rng=np.random.default_rng(0); a0=rng.normal(scale=0.05,size=198); u=rng.uniform(-1,1,12)
    start=time.perf_counter(); net.evaluate_parameter_jacobian(10.0,a0=a0,operating=u); elapsed=time.perf_counter()-start
    assert elapsed < 2.0


class _LinearField:
    thermal_model = SimpleNamespace(lambdas=np.array([2.0]))
    def vector_field(self, a, operating): return -2.0*np.asarray(a)+np.asarray([4.0])
    def evaluate(self, a, operating):
        return SimpleNamespace(vector_field=self.vector_field(a, operating), vector_field_jacobian=np.array([[-2.0]]))


def test_joint_hard_point_gn_still_converges():
    cfg=ResearchTrainingConfig(
        initial_lower=(-0.2,),initial_upper=(2.2,),operating_lower=(),operating_upper=(),
        max_response_time=2.0,residual_tolerance=2e-7,sample_count=8,validation_count=8,
        semigroup_sample_count=4,semigroup_validation_count=4,
        max_network_depth=1,max_channels_per_mode=1,max_linear_rank=1,max_hidden_rank=1,
        max_quadratic_rank=1,max_square_rank=1,max_cross_rank=1,max_state_rank=1,
        max_iterations=20,max_validation_epochs=1,prune_rounds=0,
    )
    net=make_fixed_network(_LinearField(),cfg,operating_names=[])
    _, report=train_research_network(_LinearField(),cfg,network=net)
    assert report.numerical_tolerance_met
    assert report.maximum_validation_physics_residual <= cfg.residual_tolerance
    assert report.maximum_validation_semigroup_rate_defect <= cfg.residual_tolerance


class _StaticQuadraticField:
    thermal_model = SimpleNamespace(lambdas=np.array([2.0]))
    def vector_field(self, a, operating):
        u = float(np.asarray(operating)[0])
        return -2.0 * np.asarray(a) + np.asarray([4.0 + 0.4 * u * u])
    def evaluate(self, a, operating):
        return SimpleNamespace(
            vector_field=self.vector_field(a, operating),
            vector_field_jacobian=np.array([[-2.0]]),
        )


def test_joint_gn_handles_quadratic_operating_source():
    cfg = ResearchTrainingConfig(
        initial_lower=(-0.2,), initial_upper=(2.2,),
        operating_lower=(-1.0,), operating_upper=(1.0,),
        max_response_time=2.0, residual_tolerance=2e-6,
        sample_count=12, validation_count=16,
        semigroup_sample_count=6, semigroup_validation_count=8,
        max_network_depth=1, max_channels_per_mode=1,
        max_linear_rank=2, max_hidden_rank=1,
        max_quadratic_rank=3, max_square_rank=1,
        max_cross_rank=1, max_state_rank=1,
        jacobian_point_budget=12, semigroup_jacobian_point_budget=6,
        max_iterations=30, max_validation_epochs=2, prune_rounds=0,
    )
    net = make_fixed_network(_StaticQuadraticField(), cfg, operating_names=["u"])
    _, report = train_research_network(_StaticQuadraticField(), cfg, network=net)
    assert report.numerical_tolerance_met
    assert report.maximum_validation_physics_residual <= cfg.residual_tolerance
    assert report.maximum_validation_semigroup_rate_defect <= cfg.residual_tolerance
