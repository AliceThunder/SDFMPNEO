import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_basis import build_residual_basis


class _FakeBackground:
    n_edges = 4

    def __init__(self, rhs_by_geometry):
        self.rhs_by_geometry = rhs_by_geometry

    def geometry_context(self, geometry, assemble_thermal=False):
        assert assemble_thermal is False
        return int(geometry)

    def em_operator(self, context, state):
        return sp.eye(self.n_edges, dtype=complex, format="csr")

    def rhs_matrix(self, context):
        return np.asarray(self.rhs_by_geometry[context], complex)


def test_basis_rank_is_automatically_determined_by_anchor_residual():
    e0 = np.array([[1.0], [0.0], [0.0], [0.0]], complex)
    e1 = np.array([[0.0], [1.0], [0.0], [0.0]], complex)
    bg = _FakeBackground({0: e0, 1: e1})

    V, report = build_residual_basis(
        bg,
        geometry_samples=[0, 1],
        state_samples=[np.zeros(1), np.zeros(1)],
        target_relative_residual=1e-12,
    )

    assert report.converged
    assert report.stop_reason == "target_reached"
    assert report.basis_dimension == 2
    assert V.shape == (4, 2)
    assert report.maximum_anchor_relative_residual <= 1e-12


def test_redundant_anchors_do_not_force_an_arbitrary_rank():
    e0 = np.array([[1.0], [0.0], [0.0], [0.0]], complex)
    bg = _FakeBackground({0: e0, 1: 2.0 * e0})

    V, report = build_residual_basis(
        bg,
        geometry_samples=[0, 1],
        state_samples=[np.zeros(1), np.zeros(1)],
        target_relative_residual=1e-12,
    )

    assert report.converged
    assert report.basis_dimension == 1
    assert V.shape == (4, 1)
    assert report.maximum_anchor_relative_residual <= 1e-12
