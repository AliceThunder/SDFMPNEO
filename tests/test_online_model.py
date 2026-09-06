import numpy as np
import scipy.sparse as sp

from sdfmpneo import AnalyticEvolutionGraph, ExecutableSDFMPNEOModel, ThermalSpectralModel
from sdfmpneo.em import (
    ImpressedCurrentPortSet,
    ParametricEMProblem,
    ResidualGreedyEMReducer,
)


class _Grid:
    def __init__(self):
        self.grad = sp.csr_matrix((2, 1))
        self.n_edges = 2
        self.n_nodes = 1


def test_online_model_uses_same_port_drive_for_heat_source_and_residual():
    thermal = ThermalSpectralModel.build(np.array([[1.0]]), np.array([[2.0]]))
    graph = AnalyticEvolutionGraph(thermal.lambdas, np.array([0.1]))

    omega = 5.0
    K = np.array([[3.0, 0.4], [0.4, 2.0]])
    D = np.array([[0.5, 0.1], [0.1, 0.4]])
    A0 = K + 1j * omega * D
    A_state = np.zeros((1, 2, 2), dtype=complex)
    b = np.array([1.0, 0.0], dtype=complex)
    H_metric = np.eye(2)
    H_loss = np.array([0.5 * omega**2 * D], dtype=complex)
    problem = ParametricEMProblem(A0, A_state, b, H_metric, H_loss)

    ports = ImpressedCurrentPortSet.build(
        _Grid(),
        a_basis=np.eye(2),
        n_scalar=0,
        omega=omega,
        edge_currents=np.eye(2),
        names=("tx", "rx"),
    )
    reduced = ResidualGreedyEMReducer(problem).build_multi_rhs(
        [np.array([0.0])],
        ports.coordinate_rhs,
        tolerance=1e-12,
    )

    drive = np.array([1.0 + 0.1j, -0.2 + 0.05j])
    model = ExecutableSDFMPNEOModel(
        evolution=graph,
        thermal_model=thermal,
        electromagnetic_model=reduced,
        ports=ports,
        drive_currents=drive,
    )

    result = model.evaluate(0.3)
    rhs = ports.rhs_for_currents(drive)
    expected_heat = reduced.heat_source_for_rhs(result.thermal_coordinates, rhs)

    assert np.allclose(result.reduced_heat_source, expected_heat)
    assert result.temperature_field.shape == (1,)
    assert result.impedance is not None
    assert result.impedance.impedance.shape == (2, 2)
    assert result.drive_rhs_residual_dual_norm < 1e-11
    assert np.isclose(result.physical_residual_norm, np.linalg.norm(result.physical_residual))
