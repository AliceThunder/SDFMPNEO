from types import SimpleNamespace

import numpy as np

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.long_time import (
    _uncached_realization_action,
    clear_realization_propagator_cache,
    realization_action,
)
from sdfmpneo.em.fast_reduced import _ReducedNedelecAssembler
from sdfmpneo.spatial.barycentric_polynomial import (
    assemble_polynomial_weighted_nedelec_mass,
    polynomial_p1,
)
from sdfmpneo.spatial.tetra3d import TetrahedralComplex3D
from sdfmpneo.training import research as research_training


def _one_tetrahedron():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return TetrahedralComplex3D.build(vertices, np.array([[0, 1, 2, 3]]))


def test_direct_reduced_nedelec_assembly_matches_full_projection():
    mesh = _one_tetrahedron()
    polynomial = polynomial_p1(np.array([2.0, 2.5, 3.0, 4.0]))
    rng = np.random.default_rng(13)
    fields = rng.normal(size=(mesh.n_edges, 3)) + 1j * rng.normal(size=(mesh.n_edges, 3))

    full = assemble_polynomial_weighted_nedelec_mass(mesh, [polynomial])
    reference = fields.conj().T @ (full @ fields)
    direct = _ReducedNedelecAssembler(mesh, fields).assemble([polynomial])

    assert np.allclose(direct, reference, rtol=2e-13, atol=2e-13)


def test_cached_realization_action_matches_original_finite_and_infinite():
    A = np.array(
        [[0.0, 0.0, 0.0], [0.4, -0.7, 0.0], [-0.2, 0.3, -1.2]],
        dtype=complex,
    )
    B = np.array([[1.5, -0.2], [0.1, 0.7], [-0.4, 0.3]], dtype=complex)
    clear_realization_propagator_cache()
    for time in (0.25, 1000.0, np.inf):
        expected = _uncached_realization_action(A, B, float(time))
        first = realization_action(A, B, time)
        second = realization_action(A, B, time)
        assert np.allclose(first, expected, rtol=2e-13, atol=2e-13)
        assert np.allclose(second, expected, rtol=2e-13, atol=2e-13)


class _LinearField:
    n_operating = 1
    thermal_model = SimpleNamespace(lambdas=np.array([0.4]))

    def evaluate(self, a, u):
        a = np.asarray(a, float)
        forcing = np.array([0.25 * float(u[0])])
        return SimpleNamespace(
            vector_field=-0.4 * a + forcing,
            vector_field_jacobian=np.array([[-0.4]]),
        )

    def vector_field(self, a, u):
        return self.evaluate(a, u).vector_field


def test_parallel_collocation_evaluation_matches_single_thread(monkeypatch):
    graph = ParametricAnalyticEvolutionGraph(np.array([0.4]), ["u0"])
    graph.add_product_response("response_0", 0, ("u0",), 0.15)
    points = np.array(
        [
            [0.1, 0.2, 0.0],
            [-0.2, 0.7, 0.05],
            [0.05, -0.3, 2.0],
            [0.0, 0.9, 10.0],
            [0.08, 0.4, np.inf],
        ]
    )
    field = _LinearField()

    monkeypatch.setenv("SDFMPNEO_POINT_WORKERS", "1")
    serial = research_training._evaluate(graph, field, points, jacobian=True)
    monkeypatch.setenv("SDFMPNEO_POINT_WORKERS", "4")
    parallel = research_training._evaluate(graph, field, points, jacobian=True)

    for left, right in zip(serial, parallel):
        assert np.array_equal(left.a, right.a)
        assert np.array_equal(left.residual, right.residual)
        assert np.array_equal(left.J, right.J)
        assert np.array_equal(left.initial, right.initial)
        assert np.array_equal(left.u, right.u)
        assert left.time == right.time
