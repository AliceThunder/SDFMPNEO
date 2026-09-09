from decimal import Decimal, localcontext

import numpy as np

from sdfmpneo.certification.analytic_residual_domain import _time_factor_sup


def _high_precision_time_factor(power, rho, time):
    with localcontext() as ctx:
        ctx.prec = 120
        t = Decimal.from_float(float(time))
        r = Decimal.from_float(float(rho))
        return (t ** int(power)) * (-r * t).exp()


def test_time_factor_is_outward_for_long_time_rounding_case():
    # This finite-time case was found by a randomized high-precision regression.
    # The former binary64 implementation followed by one nextafter still lay
    # below the true real-valued factor by about 5.2e-16 relative.
    power = 6
    rho = 0.0002720745335044535
    lower = 75322.29040143164
    upper = 92701.13370048316
    observed = _time_factor_sup(power, rho, lower, upper)
    expected = _high_precision_time_factor(power, rho, lower)
    assert Decimal.from_float(observed) >= expected


def test_time_factor_outward_random_finite_domain_cases():
    rng = np.random.default_rng(19)
    for _ in range(256):
        power = int(rng.integers(0, 8))
        rho = float(10.0 ** rng.uniform(-8.0, 0.0))
        lower = float(rng.uniform(0.0, 1.0e5))
        upper = float(rng.uniform(lower, 1.0e5))
        candidates = [lower, upper]
        if power > 0:
            stationary = power / rho
            if lower <= stationary <= upper:
                candidates.append(stationary)
        with localcontext() as ctx:
            ctx.prec = 120
            r = Decimal.from_float(rho)
            expected = max(
                (Decimal.from_float(t) ** power) * (-r * Decimal.from_float(t)).exp()
                for t in candidates
            )
        observed = _time_factor_sup(power, rho, lower, upper)
        assert Decimal.from_float(observed) >= expected
