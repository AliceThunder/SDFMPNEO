"""P1: original Coulomb kernel against an exactly solvable solenoidal mode."""
import numpy as np

from examples.bfzi_p1_coulomb import (
    ball_rule, vortex_modes, vortex_green_potential,
    exact_vortex_green, trial,
)
from sdfmpneo_bfzi.source import MU0


def test_p1_ball_vortex_exact_analytic_potential_and_energy():
    # W_i=e_i cross x, div W_i=0 and x dot W_i=0 on the sphere.
    # ΔU_i=-4πW_i with exterior dipole matching.
    radius=0.7
    x,w=ball_rule(radius,5,6,8)
    modes=vortex_modes(x)
    np.testing.assert_allclose(
        np.einsum("nic,ic->ni",modes,x),
        0.0,rtol=0,atol=2e-16,
    )
    potential=vortex_green_potential(x,radius)
    energy=MU0/(4*np.pi)*np.einsum(
        "n,nmc,nkc->mk",w,modes,potential,optimize=True,
    )
    np.testing.assert_allclose(
        energy,exact_vortex_green(radius),
        rtol=2e-13,atol=1e-22,
    )


def test_p1_original_coulomb_split_is_finite_on_exact_vortex_mode():
    # This is only a coarse smoke test: the independent P1 log must
    # compare real x/y refinement and physical cutoff dependence.
    result=trial(
        1.0,(3,5,8),(4,6,10),
        (6,8,12),0.4,
    )
    assert np.isfinite(result["norm_error"])
    assert result["diag_error"]<0.25
    assert result["offdiag_error"]<0.25
    assert result["L11"]>0.0
