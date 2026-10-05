import numpy as np
import scipy.sparse as sp

from sdfmpneo.electrothermal_tensor import (
    quadratic_heat_source,
    quadratic_heat_source_batch,
    quadratic_joule_tensor,
)
from sdfmpneo.em.sparse_reduced import SparseEnergyReducedEMModel
from sdfmpneo.training import AffineOperatingRHSMap


class _ComplexLinearEMProblem:
    n_em = 3
    n_thermal = 2
    b = np.zeros(3, dtype=complex)

    def operator_sparse(self, state):
        a = np.asarray(state, dtype=float)
        if a.shape != (2,):
            raise ValueError("thermal state dimension mismatch")
        diagonal = np.array(
            [
                2.0 + 0.10 * a[0] + 0.30j,
                3.0 + 0.20 * a[1] + 0.40j,
                4.0 + 0.05 * (a[0] - a[1]) + 0.20j,
            ],
            dtype=complex,
        )
        A = np.diag(diagonal)
        A[0, 1] = 0.08 - 0.03j
        A[1, 0] = 0.02 + 0.01j
        return sp.csr_matrix(A)

    def loss_operator_sparse(self, mode, state):
        a = np.asarray(state, dtype=float)
        if mode == 0:
            H = np.array(
                [
                    [1.0 + 0.02 * a[0], 0.06 + 0.03j, 0.0],
                    [0.06 - 0.03j, 0.7, 0.02j],
                    [0.0, -0.02j, 0.4 + 0.01 * a[1]],
                ],
                dtype=complex,
            )
        elif mode == 1:
            # A signed thermal test function may produce an indefinite modal
            # loss operator even though physical Joule density is non-negative.
            H = np.array(
                [
                    [0.5, 0.03 - 0.01j, 0.0],
                    [0.03 + 0.01j, -0.8, 0.02],
                    [0.0, 0.02, 0.15],
                ],
                dtype=complex,
            )
        else:
            raise ValueError("mode out of range")
        return sp.csr_matrix(H)


def _model_and_rhs():
    problem = _ComplexLinearEMProblem()
    basis = np.eye(problem.n_em, dtype=complex)
    em = SparseEnergyReducedEMModel(
        problem,
        basis,
        reference_energy_metric=sp.eye(problem.n_em, dtype=complex, format="csr"),
    )
    rhs = AffineOperatingRHSMap(
        np.array([1.0 + 0.2j, 0.3 - 0.1j, -0.2 + 0.4j]),
        np.array(
            [
                [0.7 + 0.1j, -0.2 + 0.5j],
                [0.1 - 0.3j, 0.8 + 0.2j],
                [0.4 + 0.2j, 0.2 - 0.1j],
            ]
        ),
    )
    return em, rhs


def test_quadratic_joule_tensor_exactly_reconstructs_affine_current_heat():
    em, rhs_map = _model_and_rhs()
    state = np.array([0.25, -0.15])
    tensor = quadratic_joule_tensor(em, state, rhs_map)

    assert tensor.shape == (2, 3, 3)
    assert np.allclose(tensor, np.swapaxes(tensor, 1, 2), rtol=0.0, atol=2e-15)

    operating = np.array(
        [
            [-1.0, -0.4],
            [-0.7, 0.9],
            [-0.1, 0.2],
            [0.0, 0.0],
            [0.4, -0.8],
            [0.9, 0.5],
            [1.0, 1.0],
        ]
    )
    expected = np.vstack(
        [
            em.heat_source_for_rhs(state, rhs_map.evaluate(u))
            for u in operating
        ]
    )
    direct = np.vstack([quadratic_heat_source(tensor, u) for u in operating])
    batched = quadratic_heat_source_batch(tensor, operating)

    np.testing.assert_allclose(direct, expected, rtol=4e-13, atol=4e-13)
    np.testing.assert_allclose(batched, expected, rtol=4e-13, atol=4e-13)


def test_modal_quadratic_tensor_is_not_artificially_projected_to_psd():
    em, rhs_map = _model_and_rhs()
    tensor = quadratic_joule_tensor(em, np.array([0.1, 0.2]), rhs_map)

    eigenvalues = np.linalg.eigvalsh(tensor[1])
    assert np.min(eigenvalues) < 0.0

    # The signed modal projection can therefore be negative for a legitimate
    # real operating vector; this is not a violation of Joule-density positivity.
    grid = np.linspace(-2.0, 2.0, 17)
    values = [
        quadratic_heat_source(tensor, np.array([u0, u1]))[1]
        for u0 in grid
        for u1 in grid
    ]
    assert np.min(values) < 0.0
