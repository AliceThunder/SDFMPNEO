import numpy as np
import scipy.sparse as sp

from sdfmpneo.unified_accurate_residual import accurate_residual_vector
from sdfmpneo.unified_compensated_field import (
    as_compensated_field,
    compensated_add,
    field_abs2,
    field_is_finite,
)


def test_compensated_update_preserves_sub_ulp_maxwell_correction():
    # Around 1e16, adding 1.0 to a binary64 value is rounded away.  The
    # high/low representation must retain it, because a cancellation-sensitive
    # Maxwell row can still see that unit correction.
    A = sp.csr_matrix(
        np.array(
            [
                [1.0, -1.0],
                [0.0, 1.0],
            ],
            dtype=complex,
        )
    )
    high = np.array([1.0e16, 1.0e16], dtype=complex)
    rhs = np.array([1.0, 1.0e16], dtype=complex)
    delta = np.array([1.0, 0.0], dtype=complex)

    assert (high + delta)[0] == high[0]
    field = compensated_add(as_compensated_field(high), delta)
    assert field.high[0] == high[0]
    assert field.low[0].real == 1.0
    assert field_is_finite(field)

    residual, stats = accurate_residual_vector(A, field, rhs, target_relative=1e-14)
    assert np.linalg.norm(residual) <= 1e-14
    assert stats["accumulation_mode"] in {
        "extended-longdouble-csr",
        "compensated-double-double-csr",
    }


def test_compensated_abs2_includes_high_low_cross_term():
    high = np.array([3.0 + 4.0j, -2.0 + 1.0j])
    low = np.array([1e-12 - 2e-12j, -3e-12 + 4e-12j])
    field = as_compensated_field(high)
    field = compensated_add(field, low)
    expected = (
        high.real * high.real
        + high.imag * high.imag
        + 2.0 * (high.real * low.real + high.imag * low.imag)
        + low.real * low.real
        + low.imag * low.imag
    )
    assert np.allclose(field_abs2(field), expected, rtol=1e-15, atol=0.0)
