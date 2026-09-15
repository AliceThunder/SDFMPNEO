import numpy as np

from sdfmpneo.unified_compensated_field import CompensatedComplexField, field_parts
from sdfmpneo.unified_refined_gradient_projection import _compensated_gradient_field


class _Background:
    nx = 2
    ny = 0
    nz = 0
    n_edges = 2
    edge_tuples = (
        (0, 0, 0, 0),
        (0, 1, 0, 0),
    )


def test_compensated_gradient_preserves_small_edge_difference_between_huge_potentials():
    # Gauge-fixed scalar coordinates are nodes 1 and 2.  Their high parts are
    # identical and therefore lose the physical difference in ordinary binary64;
    # the low expansion carries the 3 and 8 volt corrections.
    scalar = CompensatedComplexField(
        np.array([1.0e20 + 0j, 1.0e20 + 0j]),
        np.array([3.0 + 0j, 8.0 + 0j]),
    )
    edge = _compensated_gradient_field(_Background(), scalar)
    high, low = field_parts(edge)

    # Edge 0 runs from the gauge node 0 to node 1.  Edge 1 is node 1 -> node 2
    # and must retain the five-unit low-order difference exactly.
    assert high.shape == (2,)
    assert np.isclose(high[1] + low[1], 5.0)
    assert np.isclose((high[0] - 1.0e20) + low[0], 3.0)
