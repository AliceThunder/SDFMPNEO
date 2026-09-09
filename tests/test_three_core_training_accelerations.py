from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import time

import numpy as np
import scipy.sparse as sp

from sdfmpneo.analytic.long_time import (
    _OBSERVATION_CACHE,
    clear_realization_observation_cache,
    realization_action,
    realization_observation_action,
)
from sdfmpneo.analytic.realization import AnalyticRealization
from sdfmpneo.em.fast_reduced import SparseEnergyReducedEMModel as FastReducedEM
from sdfmpneo.em.sparse_reduced import SparseEnergyReducedEMModel as BaseReducedEM
from sdfmpneo.spatial.barycentric_polynomial import (
    assemble_polynomial_weighted_nedelec_mass,
    polynomial_multiply,
    polynomial_p1,
)
from sdfmpneo.spatial.tetra3d import TetrahedralComplex3D
from sdfmpneo.training.geometry_context_runtime import install_concurrent_geometry_context_cache


def test_observation_action_matches_full_propagation_and_reuses_rows():
    realization = AnalyticRealization.constant(1.7).response(0.45).response(0.8)
    extra = np.linspace(-0.3, 0.4, realization.dimension).astype(complex)
    first_B = np.column_stack([realization.b, extra])
    second_B = np.column_stack([0.7 * realization.b, -1.3 * extra])

    clear_realization_observation_cache()
    for t in (0.0, 0.2, 12.0, 1e5, np.inf):
        state = realization_action(realization.A, first_B, t)
        expected_value = realization.c @ state
        expected_slope = (
            np.zeros(first_B.shape[1], dtype=complex)
            if np.isposinf(t)
            else realization.c @ (realization.A @ state)
        )
        value, slope = realization_observation_action(
            realization.A, first_B, realization.c, t
        )
        assert np.allclose(value, expected_value, rtol=3e-13, atol=3e-13)
        assert np.allclose(slope, expected_slope, rtol=3e-13, atol=3e-13)

        cached = len(_OBSERVATION_CACHE)
        value2, slope2 = realization_observation_action(
            realization.A, second_B, realization.c, t
        )
        state2 = realization_action(realization.A, second_B, t)
        assert len(_OBSERVATION_CACHE) == cached
        assert np.allclose(value2, realization.c @ state2, rtol=3e-13, atol=3e-13)
        expected_slope2 = (
            np.zeros(second_B.shape[1], dtype=complex)
            if np.isposinf(t)
            else realization.c @ (realization.A @ state2)
        )
        assert np.allclose(slope2, expected_slope2, rtol=3e-13, atol=3e-13)


def _one_tetrahedron():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    return TetrahedralComplex3D.build(vertices, np.array([[0, 1, 2, 3]]))


class _FusedToyProblem:
    def __init__(self, mesh):
        self.mesh = mesh
        self.omega = 2.7
        self.n_thermal = 1
        self.n_A = 1
        self.n_scalar = 1
        self.n_em = 2
        self.a_basis = sp.csr_matrix(
            np.array([[1.0], [0.2], [-0.1], [0.4], [0.3], [-0.2]])
        )
        self.grad_c = sp.csr_matrix(
            np.array([[0.1], [0.7], [0.2], [-0.3], [0.5], [0.6]])
        )
        self.magnetic_stiffness = sp.diags(
            np.linspace(1.0, 2.0, mesh.n_edges), format="csr"
        )
        self.b = np.array([1.0 + 0.2j, -0.3 + 0.1j])
        self._base = np.array([2.0, 2.2, 2.4, 2.6])
        self._derivative = np.array([0.15, -0.05, 0.08, 0.12])
        self._test = np.array([0.7, 1.0, 0.9, 1.2])
        self.thermal_test_local = self._test.reshape(1, 1, 4)

    def _weighted_polynomials(self, a, *, derivative_mode=None, test_mode=None):
        state = np.asarray(a, float)
        if state.shape != (1,):
            raise ValueError("thermal state dimension mismatch")
        if derivative_mode not in (None, 0):
            raise ValueError("derivative mode out of range")
        if test_mode not in (None, 0):
            raise ValueError("test mode out of range")
        nodal = (
            self._base + state[0] * self._derivative
            if derivative_mode is None else self._derivative
        )
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
        magnetic[0, 0] = (
            self.a_basis.conj().T @ (self.magnetic_stiffness @ self.a_basis)
        )[0, 0]
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


def test_fused_reduced_heat_and_jacobian_match_sparse_reference():
    problem = _FusedToyProblem(_one_tetrahedron())
    basis = np.array([[1.0, 0.15j], [0.2 - 0.1j, 0.9]], dtype=complex)
    metric = sp.eye(problem.n_em, dtype=complex, format="csr")
    base = BaseReducedEM(problem, basis, reference_energy_metric=metric)
    fast = FastReducedEM(problem, basis, reference_energy_metric=metric)
    assert fast._fused_reduced

    state = np.array([0.23])
    rhs = np.array([0.7 - 0.2j, -0.4 + 0.5j])
    q_fast, J_fast = fast.heat_source_and_jacobian_for_rhs(state, rhs)
    q_base, J_base = base.heat_source_and_jacobian_for_rhs(state, rhs)
    assert np.allclose(
        fast.heat_source_for_rhs(state, rhs), base.heat_source_for_rhs(state, rhs),
        rtol=5e-13, atol=5e-13,
    )
    assert np.allclose(q_fast, q_base, rtol=8e-13, atol=8e-13)
    assert np.allclose(J_fast, J_base, rtol=2e-12, atol=2e-12)


def test_geometry_context_coalesces_same_key_and_builds_different_keys_concurrently():
    class FakeGeometryModel:
        def __init__(self):
            self.cache_size = 8
            self._cache = OrderedDict()
            self.stats = {"active": 0, "maximum": 0, "counts": {}}
            self.stats_lock = Lock()

        def geometry_vector(self, geometry):
            values = np.asarray(geometry, dtype=float)
            if values.shape != (1,):
                raise ValueError("bad geometry")
            return values

        def context(self, geometry):
            g = self.geometry_vector(geometry)
            key = tuple(float(value) for value in g)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            with self.stats_lock:
                self.stats["active"] += 1
                self.stats["maximum"] = max(
                    self.stats["maximum"], self.stats["active"]
                )
                self.stats["counts"][key] = self.stats["counts"].get(key, 0) + 1
            time.sleep(0.04)
            result = object()
            with self.stats_lock:
                self.stats["active"] -= 1
            self._cache[key] = result
            if len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return result

    install_concurrent_geometry_context_cache(FakeGeometryModel)
    model = FakeGeometryModel()

    with ThreadPoolExecutor(max_workers=2) as pool:
        left, right = list(pool.map(model.context, ([0.1], [0.9])))
    assert left is not right
    assert model.stats["maximum"] >= 2

    model2 = FakeGeometryModel()
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(model2.context, ([0.4], [0.4], [0.4], [0.4])))
    assert all(value is values[0] for value in values)
    assert model2.stats["counts"][(0.4,)] == 1
