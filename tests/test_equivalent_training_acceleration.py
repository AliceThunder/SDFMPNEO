from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from sdfmpneo.analytic import ParametricAnalyticEvolutionGraph
from sdfmpneo.analytic.long_time import (
    _uncached_realization_action,
    clear_realization_propagator_cache,
    realization_action,
)
from sdfmpneo.analytic.realization import AnalyticRealization, clear_realization_structure_cache
from sdfmpneo.em.fast_reduced import (
    SparseEnergyReducedEMModel as FastSparseEnergyReducedEMModel,
    _ReducedNedelecAssembler,
)
from sdfmpneo.em.sparse_reduced import SparseEnergyReducedEMModel as BaseSparseEnergyReducedEMModel
from sdfmpneo.spatial.barycentric_polynomial import (
    assemble_polynomial_weighted_nedelec_mass,
    polynomial_multiply,
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


class _ToyTetraProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.omega = 2.7
        self.n_thermal = 1
        self.n_A = 1
        self.n_scalar = 1
        self.n_em = 2
        self.a_basis = sp.csr_matrix(np.array([[1.0], [0.2], [-0.1], [0.4], [0.3], [-0.2]]))
        self.grad_c = sp.csr_matrix(np.array([[0.1], [0.7], [0.2], [-0.3], [0.5], [0.6]]))
        self.magnetic_stiffness = sp.diags(np.linspace(1.0, 2.0, mesh.n_edges), format="csr")
        self.b = np.array([1.0 + 0.2j, -0.3 + 0.1j])
        self._base = np.array([2.0, 2.2, 2.4, 2.6])
        self._derivative = np.array([0.15, -0.05, 0.08, 0.12])
        self._test = np.array([0.7, 1.0, 0.9, 1.2])

    def _weighted_polynomials(self, a, *, derivative_mode=None, test_mode=None):
        state = np.asarray(a, float)
        if state.shape != (1,):
            raise ValueError("thermal state dimension mismatch")
        if derivative_mode not in (None, 0):
            raise ValueError("derivative mode out of range")
        if test_mode not in (None, 0):
            raise ValueError("test mode out of range")
        nodal = self._base + state[0] * self._derivative if derivative_mode is None else self._derivative
        poly = polynomial_p1(nodal)
        if test_mode is not None:
            poly = polynomial_multiply(poly, polynomial_p1(self._test))
        return [poly], []

    def _coordinate_map(self):
        return sp.hstack([self.a_basis, self.grad_c], format="csr", dtype=complex)

    def operator_sparse(self, a):
        polynomials, _ = self._weighted_polynomials(a)
        S = assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials)
        Q = self._coordinate_map()
        magnetic = np.zeros((2, 2), dtype=complex)
        magnetic[0, 0] = (self.a_basis.conj().T @ (self.magnetic_stiffness @ self.a_basis))[0, 0]
        return sp.csr_matrix(magnetic) + 1j * self.omega * (Q.conj().T @ (S @ Q))

    def operator_derivative_sparse(self, a, mode):
        polynomials, _ = self._weighted_polynomials(a, derivative_mode=mode)
        S = assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials)
        Q = self._coordinate_map()
        return 1j * self.omega * (Q.conj().T @ (S @ Q))

    def loss_operator_sparse(self, output_mode, a):
        polynomials, _ = self._weighted_polynomials(a, test_mode=output_mode)
        W = assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials)
        L = (-1j * self.omega) * self._coordinate_map()
        return 0.5 * (L.conj().T @ (W @ L))

    def loss_operator_derivative_sparse(self, output_mode, state_mode, a):
        polynomials, _ = self._weighted_polynomials(
            a, derivative_mode=state_mode, test_mode=output_mode
        )
        W = assemble_polynomial_weighted_nedelec_mass(self.mesh, polynomials)
        L = (-1j * self.omega) * self._coordinate_map()
        return 0.5 * (L.conj().T @ (W @ L))


def test_fast_reduced_em_matches_full_sparse_projection():
    problem = _ToyTetraProblem(_one_tetrahedron())
    basis = np.array([[1.0, 0.15j], [0.2 - 0.1j, 0.9]], dtype=complex)
    metric = sp.eye(problem.n_em, dtype=complex, format="csr")
    base = BaseSparseEnergyReducedEMModel(problem, basis, reference_energy_metric=metric)
    fast = FastSparseEnergyReducedEMModel(problem, basis, reference_energy_metric=metric)
    state = np.array([0.23])
    rhs = np.array([0.7 - 0.2j, -0.4 + 0.5j])

    assert np.allclose(fast.operator_reduced(state), base.operator_reduced(state), rtol=3e-13, atol=3e-13)
    assert np.allclose(fast.heat_source_for_rhs(state, rhs), base.heat_source_for_rhs(state, rhs), rtol=3e-13, atol=3e-13)
    q_fast, J_fast = fast.heat_source_and_jacobian_for_rhs(state, rhs)
    q_base, J_base = base.heat_source_and_jacobian_for_rhs(state, rhs)
    assert np.allclose(q_fast, q_base, rtol=5e-13, atol=5e-13)
    assert np.allclose(J_fast, J_base, rtol=1e-12, atol=1e-12)


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


def test_structure_cache_reuses_math_without_sharing_mutable_arrays():
    clear_realization_structure_cache()
    left = AnalyticRealization.decay(0.4, 1.7)
    right = AnalyticRealization.constant(-0.3)
    first = left.product(right).response(0.9)
    second = left.product(right).response(0.9)
    assert np.array_equal(first.A, second.A)
    assert np.array_equal(first.b, second.b)
    assert np.array_equal(first.c, second.c)
    assert first.A is not second.A
    first.A[0, 0] += 1.0
    assert not np.array_equal(first.A, second.A)


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
