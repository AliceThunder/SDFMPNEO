import numpy as np
import scipy.linalg

from sdfmpneo.electrothermal_tensor import quadratic_heat_source, quadratic_joule_tensor
from sdfmpneo.training import AffineOperatingRHSMap


class _Problem:
    n_em = 4
    n_thermal = 2

    def loss_operator_sparse(self, mode, state):
        raise AssertionError("direct-reduced Joule tensor path must not assemble full loss operators")


class _DirectReducedEM:
    def __init__(self):
        self.problem = _Problem()
        self.V = np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.3 + 0.1j, -0.2],
                [0.1j, 0.25 - 0.05j],
            ],
            dtype=complex,
        )
        self._direct_reduced = True
        self._fused_reduced = False

    def operator_reduced(self, state):
        a = np.asarray(state, dtype=float)
        return np.array(
            [
                [2.0 + 0.2 * a[0] + 0.3j, 0.08 - 0.02j],
                [0.03 + 0.01j, 2.8 + 0.15 * a[1] + 0.2j],
            ],
            dtype=complex,
        )

    def _loss_operator_reduced(self, mode, state, derivative_mode=None):
        assert derivative_mode is None
        a = np.asarray(state, dtype=float)
        if mode == 0:
            return np.array(
                [
                    [0.7 + 0.02 * a[0], 0.05 + 0.02j],
                    [0.05 - 0.02j, 0.4],
                ],
                dtype=complex,
            )
        if mode == 1:
            return np.array(
                [
                    [0.2, -0.03j],
                    [0.03j, -0.35 + 0.01 * a[1]],
                ],
                dtype=complex,
            )
        raise ValueError("mode out of range")

    def heat_source_for_rhs(self, state, rhs):
        source = np.asarray(rhs, dtype=complex)
        reduced_rhs = self.V.conj().T @ source
        coeff = scipy.linalg.solve(self.operator_reduced(state), reduced_rhs, assume_a="gen")
        return np.array(
            [
                np.real(np.vdot(coeff, self._loss_operator_reduced(j, state) @ coeff))
                for j in range(self.problem.n_thermal)
            ],
            dtype=float,
        )


def test_quadratic_tensor_uses_direct_reduced_loss_operators_without_full_em_reconstruction():
    em = _DirectReducedEM()
    rhs = AffineOperatingRHSMap(
        np.array([0.9 + 0.2j, -0.2 + 0.1j, 0.3, -0.1j]),
        np.array(
            [
                [0.5, -0.2 + 0.1j],
                [0.1 - 0.3j, 0.7],
                [0.25 + 0.15j, -0.1],
                [-0.05, 0.3 + 0.2j],
            ],
            dtype=complex,
        ),
    )
    state = np.array([0.2, -0.15])
    tensor = quadratic_joule_tensor(em, state, rhs)

    for operating in (
        np.array([0.0, 0.0]),
        np.array([0.8, -0.4]),
        np.array([-1.2, 0.6]),
    ):
        expected = em.heat_source_for_rhs(state, rhs.evaluate(operating))
        actual = quadratic_heat_source(tensor, operating)
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=5e-13)
