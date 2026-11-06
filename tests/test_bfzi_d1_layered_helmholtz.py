"""D1 method tests: physical limits, interfaces and power; no paper-table fit gate."""
# Optional historical paper comparison is kept as an example function below,
# but is not run by pytest as an algorithm-acceptance condition.
import numpy as np
from numpy.testing import assert_allclose
from sdfmpneo_bfzi.layered_helmholtz import (
    LayeredCoaxialCase, interface_green, incremental_impedance, paper_case,
    propagation, MU0, field_point, spectral_joule_power,
)


def test_homogeneous_green_limit():
    case = LayeredCoaxialCase((.08,), (.09,), .12, 85000., (4., 4., 4.))
    for k in (0.1, 10., 200., 1000.):
        u = propagation(k, 2 * np.pi * case.frequency_hz, 4., 1.,
                        include_displacement=False)
        g = interface_green(k, case)
        want = [MU0/(2*u), MU0/(2*u)*np.exp(-u*.12), MU0/(2*u)]
        assert_allclose(g, want, rtol=2e-11)


def test_vacuum_and_two_port_properties():
    dry = LayeredCoaxialCase((.085,), (.09,), .1, 100000., (0., 0., 0.))
    assert np.max(np.abs(incremental_impedance(dry, k_max=1200))) < 1e-12
    wet = LayeredCoaxialCase((.085,), (.09,), .1, 100000., (4., 4., 4.))
    z = incremental_impedance(wet, k_max=1200)
    assert_allclose(z, z.T, rtol=0, atol=1e-16)
    assert np.min(np.linalg.eigvalsh(z.real)) > -1e-8


def test_cutoff_and_layers():
    for scenario in ("infinite", "semifinite", "finite"):
        case = paper_case(scenario, turns_tx=2, turns_rx=2)
        z1 = incremental_impedance(case, k_max=1800)
        z2 = incremental_impedance(case, k_max=3200)
        assert_allclose(z1, z2, rtol=4e-4, atol=1e-6)
        assert z1[0, 0].real >= 0 and z1[1, 1].real >= 0


def example_published_table_2_infinite_and_semifinite():
    # Published analytical (R_tx,R_rx,R_mutual) [ohm], not fitted data.
    reference = {
        (85000., "infinite"): (.261, .334, .209),
        (85000., "semifinite"): (.133, .301, .152),
        (400000., "infinite"): (5.082, 6.505, 3.854),
        (400000., "semifinite"): (2.668, 6.127, 2.974),
    }
    for (freq, scenario), target in reference.items():
        case = paper_case(scenario, frequency_hz=freq)
        z = incremental_impedance(case, k_max=5000, include_displacement=True)
        values = (z[0,0].real, z[1,1].real, z[0,1].real)
        assert_allclose(values, target, rtol=0.004, atol=0.001)


def test_full_wave_and_invalid_geometry():
    case = paper_case("finite", turns_tx=1, turns_rx=1)
    assert np.all(np.isfinite(incremental_impedance(
        case, k_max=1200, include_displacement=True)))
    try:
        LayeredCoaxialCase((-.1,), (.1,), .1, 100000., (0., 4., 0.))
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid source radius accepted")


def test_independent_on_axis_biot_savart_and_field_superposition():
    dry = LayeredCoaxialCase((.1,), (.08,), .12, 100000., (0., 0., 0.))
    z = -.07
    field = field_point(dry, 0., z, k_max=4000)
    expected_bz = MU0 * .1**2 / (2 * (.1**2 + z**2)**1.5)
    assert_allclose(field["B_z_T"], expected_bz, rtol=1e-5)
    assert abs(field["B_r_T"]) < 1e-12
    assert field["q_W_m3"] == 0.0

    wet = LayeredCoaxialCase((.1,), (.08,), .12, 100000., (4., 4., 4.))
    point = (0.12, 0.04)
    f0 = field_point(wet, *point, (1., 0.), k_max=1600)
    f1 = field_point(wet, *point, (0., 1.), k_max=1600)
    combined = field_point(wet, *point, (1., 1j), k_max=1600)
    for key in ("A_phi_T_m", "E_phi_V_m", "B_r_T", "B_z_T", "J_phi_A_m2"):
        assert_allclose(combined[key], f0[key] + 1j*f1[key], rtol=2e-8, atol=1e-12)
    assert_allclose(combined["q_W_m3"],
                    abs(combined["J_phi_A_m2"])**2 / (2*4.),
                    rtol=1e-13)


def test_layered_seawater_parseval_power_matches_complex_port_power():
    # Internal spectral-model identity, not an independent 3-D A-phi benchmark.
    for scenario in ("infinite", "semifinite", "finite"):
        case = paper_case(scenario, frequency_hz=400000.,
                          turns_tx=2, turns_rx=2)
        z = incremental_impedance(case, k_max=2400,
                                  include_displacement=False)
        for currents in ((1.,0.), (0.,1.), (1.,1j), (1.,-1.)):
            i = np.asarray(currents, dtype=complex)
            port_watts = .5 * np.vdot(i, z @ i).real
            volume_watts = spectral_joule_power(case, currents,
                                                k_max=2400)
            assert port_watts > 0
            assert_allclose(volume_watts, port_watts, rtol=2e-5,
                            atol=1e-10)


def test_two_independent_sheet_source_jump_conditions():
    # Validate source normalization, interface DtN and independent port
    # directions without referring to fitted paper curves or G1 matrices.
    from sdfmpneo_bfzi.layered_helmholtz import propagation, interface_green
    for scenario in ("infinite", "semifinite", "finite"):
        case = paper_case(scenario, frequency_hz=400000.)
        d = case.gap_m
        omega = 2 * np.pi * case.frequency_hz
        for k in (.05, 10., 200., 1500.):
            u1, u2, u3 = (
                propagation(k, omega, s, er, include_displacement=True)
                for s, er in zip(case.conductivity_s_m, case.eps_r)
            )
            g00, g0d, gdd = interface_green(k, case, include_displacement=True)
            for st, sr in ((1.+0j, 0j), (0j, 1.+0j), (1.+2j, -.7+.3j)):
                a0, ad = g00*st + g0d*sr, g0d*st + gdd*sr
                t = np.exp(-u2*d)
                denominator = -np.expm1(-2*u2*d)
                slab_deriv_0 = (-u2*(1+t*t)*a0 + 2*u2*t*ad)/denominator
                slab_deriv_d = (-2*u2*t*a0 + u2*(1+t*t)*ad)/denominator
                jump0 = (u1*a0 - slab_deriv_0)/MU0
                jumpd = (slab_deriv_d + u3*ad)/MU0
                assert_allclose(jump0, st, rtol=2e-12, atol=2e-12)
                assert_allclose(jumpd, sr, rtol=2e-12, atol=2e-12)
