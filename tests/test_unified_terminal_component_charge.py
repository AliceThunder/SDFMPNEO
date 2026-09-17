import types

import numpy as np

from sdfmpneo.unified_terminal_component_charge import hybrid_terminal_charge_from_targets


def _node_id(nx, ny, nz, i, j, k):
    del nx
    return (int(i) * int(ny) + int(j)) * int(nz) + int(k)


def _background(axis):
    axis = np.asarray(axis, float)
    return types.SimpleNamespace(x=axis, y=axis, z=axis)


def test_feed_refinement_keeps_return_coarse_nodal_load_exactly():
    coarse = _background([0.0, 1.0, 2.0])
    fine = _background([0.0, 0.5, 1.0, 1.5, 2.0])
    qc = np.zeros(27, float)
    qc[_node_id(3, 3, 3, 0, 1, 1)] = -1.0
    qc[_node_id(3, 3, 3, 2, 1, 1)] = 1.0
    qf = np.zeros(125, float)
    qf[_node_id(5, 5, 5, 0, 2, 2)] = -0.4
    qf[_node_id(5, 5, 5, 1, 2, 2)] = -0.6
    # This fine return cloud must be ignored when feed alone is certified.
    qf[_node_id(5, 5, 5, 4, 2, 2)] = 0.3
    qf[_node_id(5, 5, 5, 3, 2, 2)] = 0.7

    hybrid, meta = hybrid_terminal_charge_from_targets(coarse, fine, qc, qf, 0)
    positive = np.maximum(hybrid, 0.0)
    negative = np.minimum(hybrid, 0.0)
    expected_return = np.zeros_like(hybrid)
    expected_return[_node_id(5, 5, 5, 4, 2, 2)] = 1.0

    assert np.allclose(positive, expected_return, rtol=0.0, atol=0.0)
    assert np.isclose(np.sum(negative), -1.0)
    assert np.isclose(np.sum(hybrid), 0.0, atol=5e-14)
    assert meta["selected_terminal_name"] == "feed"
    assert meta["terminal_component_lock"] is True


def test_return_refinement_keeps_feed_coarse_nodal_load_exactly():
    coarse = _background([0.0, 1.0, 2.0])
    fine = _background([0.0, 0.5, 1.0, 1.5, 2.0])
    qc = np.zeros(27, float)
    qc[_node_id(3, 3, 3, 0, 1, 1)] = -1.0
    qc[_node_id(3, 3, 3, 2, 1, 1)] = 1.0
    qf = np.zeros(125, float)
    # This fine feed cloud must be ignored when return alone is certified.
    qf[_node_id(5, 5, 5, 0, 2, 2)] = -0.25
    qf[_node_id(5, 5, 5, 1, 2, 2)] = -0.75
    qf[_node_id(5, 5, 5, 4, 2, 2)] = 0.45
    qf[_node_id(5, 5, 5, 3, 2, 2)] = 0.55

    hybrid, meta = hybrid_terminal_charge_from_targets(coarse, fine, qc, qf, 1)
    positive = np.maximum(hybrid, 0.0)
    negative = np.minimum(hybrid, 0.0)
    expected_feed = np.zeros_like(hybrid)
    expected_feed[_node_id(5, 5, 5, 0, 2, 2)] = -1.0

    assert np.allclose(negative, expected_feed, rtol=0.0, atol=0.0)
    assert np.isclose(np.sum(positive), 1.0)
    assert np.isclose(np.sum(hybrid), 0.0, atol=5e-14)
    assert meta["selected_terminal_name"] == "return"
    assert meta["terminal_component_lock"] is True
