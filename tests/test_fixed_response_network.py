import numpy as np
import pytest

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork


def _one_mode():
    network = FixedAnalyticResponseNetwork([2.0], [], depth=1, channels_per_mode=1, quadratic_rank=1, cross_rank=1, state_rank=1)
    theta = np.zeros(network.parameter_count)
    theta[network._indices("bias_0")[0]] = 4.0
    theta[network._indices("channel_gate")[0, 0]] = 1.0
    return network.with_parameters(theta)


def test_explicit_first_layer_bias_has_exact_analytic_response():
    network = _one_mode()
    a0 = np.array([3.0])
    a, da = network.evaluate(0.7, a0=a0, operating=[])
    decay = np.exp(-2.0 * 0.7)
    expected = 3.0 * decay + 2.0 * (1.0 - decay)
    assert np.allclose(a, [expected])
    assert np.allclose(da, [-2.0 * 3.0 * decay + 4.0 * decay])
    steady, dsteady = network.evaluate(float("inf"), a0=a0, operating=[])
    assert np.allclose(steady, [2.0])
    assert np.allclose(dsteady, [0.0])


def test_parameter_jacobian_matches_bias_finite_difference():
    network = _one_mode()
    a, da, ja, jda = network.evaluate_parameter_jacobian(0.3, a0=[0.2], operating=[])
    index = int(network._indices("bias_0")[0])
    eps = 1e-7
    theta = network.parameters.copy(); theta[index] += eps
    ap, dap = network.with_parameters(theta).evaluate(0.3, a0=[0.2], operating=[])
    theta[index] -= 2 * eps
    am, dam = network.with_parameters(theta).evaluate(0.3, a0=[0.2], operating=[])
    assert np.allclose(ja[:, index], (ap - am) / (2 * eps), rtol=2e-6, atol=2e-8)
    assert np.allclose(jda[:, index], (dap - dam) / (2 * eps), rtol=2e-6, atol=2e-8)


def test_extreme_time_and_infinity_are_stable():
    network = _one_mode()
    for time in (1e300, float("inf")):
        a, da = network.evaluate(time, a0=[0.2], operating=[])
        assert np.all(np.isfinite(a))
        assert np.all(np.isfinite(da))
        assert np.allclose(a, [2.0])
        assert np.allclose(da, [0.0])


def test_metadata_accepts_only_the_current_format():
    network = _one_mode()
    metadata = network.to_metadata()
    loaded = FixedAnalyticResponseNetwork.from_metadata(metadata, network.parameters)
    assert np.allclose(loaded.parameters, network.parameters)
    bad = dict(metadata); bad["format_version"] -= 1
    with pytest.raises(ValueError, match="unsupported fixed analytic response network format"):
        FixedAnalyticResponseNetwork.from_metadata(bad, network.parameters)
    assert not hasattr(network, "legacy_parameter_count")
