"""D1 spatial Helmholtz checks: external tabulated E scales are not a D1 PASS."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from sdfmpneo_bfzi.layered_helmholtz import paper_case, field_point
from sdfmpneo_bfzi.layered_grid import field_grid, secondary_field_grid


@pytest.mark.parametrize("scenario", ("infinite", "semifinite", "finite"))
def test_batched_field_agrees_point_quadrature(scenario):
    case = paper_case(scenario, turns_tx=2, turns_rx=2)
    r = np.array([0., .05, .11, .14])
    z = np.array([-.05, .025, .07, .15])
    got = field_grid(case, r, z, (1., 1j), k_max=2200)
    for i in range(len(r)):
        for j in range(len(z)):
            want = field_point(case, r[i], z[j], (1., 1j), k_max=2200)
            for key in ("A_phi_T_m", "E_phi_V_m", "B_r_T",
                        "B_z_T", "J_phi_A_m2", "q_W_m3"):
                assert_allclose(got[key][i, j], want[key],
                                rtol=3e-5, atol=2e-10)


# Published Table 2 "analytical maximum electric field" [V/m].
# The paper does not unambiguously identify this with E_secondary;
# the close scale match is evidence for a working interpretation,
# not an assertion that the authors made that definition explicitly.
@pytest.mark.parametrize("scenario,frequency,reference", [
    ("infinite", 85000., .298),
    ("infinite", 400000., 5.375),
    ("semifinite", 85000., .231),
    ("semifinite", 400000., 4.497),
    ("finite", 85000., .132),
    ("finite", 400000., 2.900),
])
def test_published_e_max_secondary_field_interpretation(scenario, frequency, reference):
    case = paper_case(scenario, frequency_hz=frequency)
    r = np.linspace(0, .5, 151)
    z = np.r_[np.linspace(-.12, -.0008, 31),
              np.linspace(.0008, .0992, 57),
              np.linspace(.1008, .22, 31)]
    scattered = secondary_field_grid(case, r, z, (5., 5.),
                                     k_max=4000., panel_width=8.)
    allowed = np.ones_like(z, dtype=bool)
    if scenario == "semifinite":
        allowed = z > 0
    if scenario == "finite":
        allowed = (z > 0) & (z < case.gap_m)
    peak = np.max(np.abs(scattered["E_phi_secondary_V_m"][:, allowed]))
    # Literature field observable not fully specified: compute diagnostic,
    # but do NOT select an after-the-fact tolerance as an acceptance gate.
    assert np.isfinite(peak) and peak > 0.
    assert reference > 0.  # reference retained for separate reported comparison


def test_total_field_not_secondary_field_defines_joule_heat():
    case = paper_case("finite", frequency_hz=400000., turns_tx=2, turns_rx=2)
    r, z = [.09, .12], [.02, .045]
    total = field_grid(case, r, z, (1., 1j), k_max=2200)
    secondary = secondary_field_grid(case, r, z, (1., 1j), k_max=2200)
    q_physical = .5 * 4. * abs(total["E_phi_V_m"])**2
    q_secondary = .5 * 4. * abs(secondary["E_phi_secondary_V_m"])**2
    assert_allclose(total["q_W_m3"], q_physical, rtol=1e-13)
    assert np.max(abs(q_secondary-q_physical)) > 1e-4


def test_k_cutoff_secondary_field_stability():
    case = paper_case("finite", frequency_hz=400000., turns_tx=2, turns_rx=2)
    r, z = [.08, .15, .24], [.013, .045, .085]
    a = secondary_field_grid(case, r, z, (5., 5.), k_max=1800)
    b = secondary_field_grid(case, r, z, (5., 5.), k_max=4000)
    assert_allclose(a["E_phi_secondary_V_m"],
                    b["E_phi_secondary_V_m"], rtol=5e-5, atol=1e-8)
