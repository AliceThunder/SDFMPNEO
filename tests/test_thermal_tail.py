import numpy as np

from sdfmpneo.thermal import ThermalSpectralModel


def test_certified_rank_is_selected_from_tail_bound_not_fixed_count():
    M = np.eye(3)
    K = np.diag([1.0, 4.0, 9.0])
    model = ThermalSpectralModel.build(M, K)

    initial = np.array([1.0, 0.1, 0.01])
    selection = model.select_certified_rank(
        initial_field=initial,
        source_dual_bound=0.2,
        requested_state_tolerance=0.03,
    )

    # r=1: initial tail ~= 0.1005, not certified.
    # r=2: max(0.01, 0.2/9) = 0.02222..., certified.
    assert selection.model.rank == 2
    assert selection.certificate.certified
    assert np.isclose(selection.certificate.initial_tail_norm, 0.01)
    assert np.isclose(selection.certificate.steady_forcing_tail_bound, 0.2 / 9.0)
    assert np.isclose(selection.certificate.uniform_projection_tail_bound, 0.2 / 9.0)


def test_time_dependent_tail_bound_interpolates_between_initial_and_forced_limits():
    model = ThermalSpectralModel.build(np.eye(3), np.diag([1.0, 4.0, 9.0]))
    cert = model.projection_tail_certificate(
        2,
        initial_field=np.array([0.0, 0.0, 0.01]),
        source_dual_bound=0.18,
        requested_state_tolerance=0.03,
    )

    assert np.isclose(cert.bound_at_time(0.0), 0.01)
    assert cert.bound_at_time(1.0) > 0.01
    assert cert.bound_at_time(1.0) < 0.18 / 9.0
    assert np.isclose(cert.uniform_projection_tail_bound, 0.02)


def test_output_tolerance_is_backpropagated_to_state_tolerance():
    model = ThermalSpectralModel.build(np.eye(3), np.diag([1.0, 4.0, 9.0]))
    selection = model.select_certified_rank_for_output(
        initial_field=np.array([1.0, 0.1, 0.01]),
        source_dual_bound=0.2,
        requested_output_tolerance=0.06,
        output_lipschitz=2.0,
    )
    assert selection.model.rank == 2
    assert np.isclose(selection.certificate.requested_state_tolerance, 0.03)


def test_full_rank_has_zero_projection_tail():
    model = ThermalSpectralModel.build(np.eye(2), np.diag([2.0, 5.0]))
    cert = model.projection_tail_certificate(
        2,
        initial_field=np.array([3.0, -4.0]),
        source_dual_bound=100.0,
        requested_state_tolerance=1e-12,
    )
    assert cert.certified
    assert cert.uniform_projection_tail_bound == 0.0
    assert cert.bound_at_time(100.0) == 0.0
