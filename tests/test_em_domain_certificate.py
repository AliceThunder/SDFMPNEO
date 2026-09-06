import numpy as np

from sdfmpneo.certification import certify_affine_reduced_residual_domain
from sdfmpneo.em import ParametricEMProblem, ReducedEMModel


def make_problem(A0, A_state, b):
    n_thermal = A_state.shape[0]
    n_em = A0.shape[0]
    return ParametricEMProblem(
        A0=np.asarray(A0, dtype=complex),
        A_state=np.asarray(A_state, dtype=complex),
        b=np.asarray(b, dtype=complex),
        H_metric=np.eye(n_em),
        H_loss=np.zeros((n_thermal, n_em, n_em), dtype=complex),
    )


def test_full_coordinate_basis_certifies_entire_continuous_box_exactly():
    problem = make_problem(
        np.array([[2.0, 0.2], [0.2, 3.0]]),
        np.array([[[0.1, 0.0], [0.0, -0.05]]]),
        np.array([1.0, 0.5]),
    )
    model = ReducedEMModel(problem, np.eye(2, dtype=complex))
    cert = certify_affine_reduced_residual_domain(
        model,
        lower=np.array([-10.0]),
        upper=np.array([10.0]),
        tolerance=1e-12,
        work_budget=1,
    )
    assert cert.certified
    assert cert.global_upper_bound == 0.0


def test_continuous_domain_finds_a_center_violation_without_grid_sampling_claim():
    problem = make_problem(
        np.eye(2),
        np.zeros((1, 2, 2)),
        np.array([1.0, 1.0]),
    )
    model = ReducedEMModel(problem, np.array([[1.0], [0.0]], dtype=complex))
    cert = certify_affine_reduced_residual_domain(
        model,
        lower=np.array([-1.0]),
        upper=np.array([1.0]),
        tolerance=0.5,
        work_budget=4,
    )
    assert cert.status == "violated"
    assert cert.violating_state is not None
    assert cert.observed_lower_bound > 0.5


def test_work_budget_exhaustion_is_indeterminate_never_false_certified():
    # The one-dimensional reduced space is exactly invariant, so the true
    # residual is zero. The derivative triangle bound is intentionally loose on
    # a wide box; one box evaluation cannot prove the result and must return
    # indeterminate rather than silently accepting it.
    problem = make_problem(
        np.diag([2.0, 3.0]),
        np.array([[[1.0, 0.0], [0.0, 0.0]]]),
        np.array([1.0, 0.0]),
    )
    model = ReducedEMModel(problem, np.array([[1.0], [0.0]], dtype=complex))
    cert = certify_affine_reduced_residual_domain(
        model,
        lower=np.array([-0.5]),
        upper=np.array([0.5]),
        tolerance=0.1,
        work_budget=1,
    )
    assert cert.status == "indeterminate"
    assert not cert.certified
    assert cert.unresolved_boxes > 0
