import numpy as np
import scipy.sparse as sp

from sdfmpneo.certification import certify_multiport_impedance
from sdfmpneo.em import CompatibleAphiDiscretization, ImpressedCurrentPortSet


class _Grid:
    def __init__(self, grad):
        self.grad = sp.csr_matrix(grad)
        self.n_edges = self.grad.shape[0]
        self.n_nodes = self.grad.shape[1]


def make_problem_and_ports():
    C = np.array([
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
    ])
    G = np.ones((3, 1))
    R = np.array([
        [1.0, 0.0],
        [-1.0, 1.0],
        [0.0, -1.0],
    ])
    Nu = np.diag([2.0, 1.5])
    S0 = np.diag([1.0, 0.8, 0.6])
    omega = 3.0
    J1 = np.array([1.0, -1.0, 0.0])
    J2 = np.array([0.0, 1.0, -1.0])

    disc = CompatibleAphiDiscretization(
        curl=C,
        grad_c=G,
        a_basis=R,
        reluctivity_hodge=Nu,
        conductivity0=S0,
        conductivity_state=np.zeros((1, 3, 3)),
        source_current=J1,
        omega=omega,
        riesz_metric=np.eye(3),
        thermal_loss_hodge0=np.array([S0]),
        thermal_loss_hodge_state=np.zeros((1, 1, 3, 3)),
    )
    problem = disc.to_parametric_problem()
    ports = ImpressedCurrentPortSet.build(
        _Grid(G),
        a_basis=R,
        n_scalar=1,
        omega=omega,
        edge_currents=np.column_stack([J1, J2]),
        names=("tx", "rx"),
    )
    return problem, ports


def test_actual_full_vs_reduced_impedance_error_is_below_certificate():
    problem, ports = make_problem_and_ports()
    a = np.zeros(problem.n_thermal)

    # Deliberately incomplete two-coordinate basis in a three-coordinate field
    # space, so the certificate is tested on a nonzero ROM error.
    V = np.eye(problem.n_em, dtype=complex)[:, :2]
    full = ports.evaluate(problem, a)
    reduced = ports.evaluate(problem, a, reduced_basis=V)
    cert = certify_multiport_impedance(problem, ports, a, V)

    actual_z_error = np.abs(full.impedance - reduced.impedance)
    assert np.all(actual_z_error <= cert.impedance_abs_error_bound)
    assert np.all(np.abs(full.resistance - reduced.resistance) <= cert.resistance_abs_error_bound)
    assert np.all(np.abs(full.inductance - reduced.inductance) <= cert.inductance_abs_error_bound)
    assert cert.electromagnetic_stability > 0.0
    assert np.any(cert.residual_dual_norms > 0.0)
