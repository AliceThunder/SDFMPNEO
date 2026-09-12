import numpy as np

from sdfmpneo.analytic.fixed_response_network import FixedAnalyticResponseNetwork
from sdfmpneo.training.fixed_layer_cache_runtime import (
    _semigroup_endpoint_points,
    clear_layer_program_cache,
    evaluate_layer_semigroup as compiled_layer_semigroup,
    layer_program,
)
from sdfmpneo.training.fixed_physics_runtime import (
    evaluate_layer_semigroup as direct_layer_semigroup,
    evaluate_semigroup_batch,
)
from sdfmpneo.training.fixed_trial_runtime import _compiled_semigroup_records


def _deep_network():
    network = FixedAnalyticResponseNetwork(
        [1.0, 2.0], [], max_response_time=2.0,
        input_center=[0.0, 0.0], input_scale=[1.0, 1.0],
        depth=2, channels_per_mode=1,
        linear_rank=1, hidden_rank=1, quadratic_rank=1, square_rank=1,
        cross_rank=1, state_rank=1,
        layer_widths=(2, 1),
        layer_hidden_ranks=(1,), layer_cross_ranks=(1,), layer_state_ranks=(1,),
    )
    theta = network.parameters.copy()
    theta[network._indices("bias_0")] = np.array([0.08, -0.03])
    theta[network._indices("input_linear_out")] = np.array([[0.06], [-0.04]])
    theta[network._indices("bias_1")[0]] = 0.035
    theta[network._indices("hidden_linear_out_1")[0, 0]] = -0.025
    return network.with_parameters(theta)


def _rows():
    return np.asarray([
        [0.25, -0.15, 0.4, 0.7],
        [-0.10, 0.35, 0.65, 0.55],
        [0.40, 0.05, 0.9, 0.8],
    ])


def test_compiled_restart_layer_jacobian_matches_direct_path():
    network = _deep_network()
    rows = _rows()
    clear_layer_program_cache()
    compiled, ids = compiled_layer_semigroup(network, rows, 1)
    direct, direct_ids = direct_layer_semigroup(network, rows, 1)
    np.testing.assert_array_equal(ids, direct_ids)
    for fast, reference in zip(compiled, direct):
        np.testing.assert_allclose(
            fast.residual, reference.residual, rtol=2e-12, atol=2e-13
        )
        np.testing.assert_allclose(
            fast.parameter_jacobian,
            reference.parameter_jacobian,
            rtol=0.0,
            atol=8e-12,
        )


def test_compiled_restart_candidate_matches_full_exact_semigroup():
    parent = _deep_network()
    rows = _rows()
    ids = np.asarray(parent.layer_amplitude_parameter_indices(1), dtype=int)
    theta = parent.parameters.copy()
    theta[ids] += np.linspace(-8e-4, 1.2e-3, len(ids))
    trial = parent.with_parameters(theta)

    direct_points, first_points = _semigroup_endpoint_points(parent, rows)
    clear_layer_program_cache()
    program = layer_program(
        parent, np.vstack([direct_points, first_points]), 1
    )
    assert program is not None
    fast = _compiled_semigroup_records(trial, rows, program)
    exact = evaluate_semigroup_batch(trial, rows)
    for left, right in zip(fast, exact):
        np.testing.assert_allclose(
            left.residual, right.residual, rtol=2e-12, atol=2e-13
        )
        np.testing.assert_allclose(
            left.raw_defect, right.raw_defect, rtol=2e-12, atol=2e-13
        )
