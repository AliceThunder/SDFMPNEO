"""Focused D1 spectral reference tests; they do not certify D1 or D2."""
import numpy as np
from numpy.testing import assert_allclose
from sdfmpneo_bfzi.layered_helmholtz import (
    LayeredCoaxialCase, interface_green, incremental_impedance, paper_case,
    propagation, MU0,
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


def test_published_table_2_infinite_and_semifinite():
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
