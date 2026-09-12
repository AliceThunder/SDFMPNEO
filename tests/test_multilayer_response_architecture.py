import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from sdfmpneo.training.automatic_thermal_rank import select_rank_from_modal_response_envelope
from sdfmpneo.training.source_prefit_factorization import fit_source_factors
from sdfmpneo.training.thermal_horizon import finite_horizon_rank_diagnostic


def test_source_prefit_learns_low_rank_linear_directions_not_random_features():
    net = FixedAnalyticResponseNetwork(
        [0.5, 1.0, 2.0], ["u"], max_response_time=2.0,
        depth=3, channels_per_mode=1,
        layer_widths=(3, 2, 1),
        linear_rank=2, quadratic_rank=1, square_rank=1,
        layer_hidden_ranks=(1, 1), layer_cross_ranks=(1, 1),
        layer_state_ranks=(1, 1),
    )
    rng = np.random.default_rng(12)
    samples = rng.uniform(-0.8, 0.8, size=(32, 4))
    left = rng.normal(size=(4, 2))
    right = rng.normal(size=(3, 2))
    bias = np.array([0.3, -0.2, 0.1])
    desired = bias[None, :] + samples @ (left @ right.T)

    fitted, residual = fit_source_factors(net, samples, desired)
    assert np.max(np.linalg.norm(residual, axis=1)) < 1e-9
    assert len(fitted.response_nodes) == 3
    assert fitted.structure_summary()["active_channels_by_layer"] == [3, 0, 0]


def test_finite_horizon_diagnostic_approaches_steady_selection():
    lambdas = np.array([0.05, 0.5, 5.0, 20.0])
    steady = np.array([1.0, 0.2, 0.03, 0.005])
    kwargs = dict(
        relative_tolerance=0.02,
        absolute_tolerance=0.0,
        safety_factor=1.0,
        boundary_fraction=0.25,
    )
    steady_rank, _ = select_rank_from_modal_response_envelope(steady, **kwargs)
    result = finite_horizon_rank_diagnostic(
        lambdas, steady, horizons=(0.1, 10.0, 1000.0), **kwargs
    )
    assert set(result) == {"0.1s", "10s", "1000s"}
    assert all(1 <= item["rank"] <= len(lambdas) for item in result.values())
    assert result["1000s"]["rank"] == steady_rank
    assert result["0.1s"]["total_response_norm"] < result["1000s"]["total_response_norm"]


def test_inactive_funnel_layers_do_not_change_forward_solution():
    net = FixedAnalyticResponseNetwork(
        [0.7, 1.3], ["u"], max_response_time=3.0,
        depth=3, channels_per_mode=1,
        layer_widths=(2, 1, 1),
        linear_rank=1, quadratic_rank=1, square_rank=1,
        layer_hidden_ranks=(1, 1), layer_cross_ranks=(1, 1),
        layer_state_ranks=(1, 1),
    )
    # Layer 2/3 amplitudes initialize at exactly zero.  Compare the multilayer
    # fast route against a depth-one network carrying the same first-layer block.
    one = FixedAnalyticResponseNetwork(
        [0.7, 1.3], ["u"], max_response_time=3.0,
        depth=1, channels_per_mode=1,
        linear_rank=1, quadratic_rank=1, square_rank=1,
        hidden_rank=1, cross_rank=1, state_rank=1,
        input_center=net.input_center, input_scale=net.input_scale,
    )
    p = one.parameters.copy()
    for name in (
        "bias_0", "input_linear_in", "input_linear_out", "quadratic_u",
        "quadratic_v", "quadratic_out", "square_in", "square_out",
        "quadratic_gate", "square_gate",
    ):
        p[one._indices(name)] = net._array(name)
    p[one._indices("channel_gate")[0]] = net._array("channel_gate")[0, :2]
    one = one.with_parameters(p)
    a0 = np.array([0.1, -0.05]); u = np.array([0.4])
    a_multi, da_multi = net.evaluate(1.2, a0=a0, operating=u)
    a_one, da_one = one.evaluate(1.2, a0=a0, operating=u)
    np.testing.assert_allclose(a_multi, a_one, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(da_multi, da_one, rtol=1e-12, atol=1e-12)
