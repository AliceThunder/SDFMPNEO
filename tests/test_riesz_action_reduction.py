import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sdfmpneo.em import (
    CertifiedPCGRieszAction,
    DiagonalGershgorinEnergyPreconditioner,
    SparseEnergyResidualGreedyEMReducer,
)


class DiagonalThermalToyProblem:
    n_em = 3
    n_thermal = 1

    def __init__(self):
        self.b = np.array([1.0, 0.7, 1.3], dtype=complex)

    def operator_sparse(self, a):
        state = np.asarray(a, dtype=float)
        if state.shape != (1,):
            raise ValueError("thermal state dimension mismatch")
        s = float(state[0])
        K = sp.diags([2.0, 3.0, 4.0], format="csr")
        D = sp.diags(
            [1.0 + 0.20 * s, 1.5 + 0.10 * s, 2.0 + 0.05 * s],
            format="csr",
        )
        return (K + 1j * D).tocsr()


def certified_pcg_factory(H):
    return CertifiedPCGRieszAction(
        H,
        DiagonalGershgorinEnergyPreconditioner.build(H),
    )


def test_reducer_and_residual_certificate_work_with_sparse_lu_forbidden(monkeypatch):
    def forbidden_splu(*_args, **_kwargs):
        raise AssertionError("exact sparse LU Riesz inverse was used")

    monkeypatch.setattr(spla, "splu", forbidden_splu)

    problem = DiagonalThermalToyProblem()
    reducer = SparseEnergyResidualGreedyEMReducer(
        problem,
        riesz_action_factory=certified_pcg_factory,
    )
    requested = 1e-9
    model = reducer.build(
        [np.array([-0.5]), np.array([0.0]), np.array([0.5])],
        requested_energy_state_error=requested,
    )

    assert model.reduction_certificate.certified
    assert model.reduction_certificate.maximum_energy_state_error_bound <= requested

    for state in [np.array([-0.5]), np.array([0.0]), np.array([0.5])]:
        cert = model.residual_certificate(state)
        assert cert.residual_dual_energy_norm_lower_bound >= 0.0
        assert (
            cert.residual_dual_energy_norm_lower_bound
            <= cert.residual_dual_energy_norm_upper_bound
        )
        assert cert.energy_state_error_bound <= requested
