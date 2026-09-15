"""High/low complex field representation for the final local Maxwell refinement.

The production Maxwell solve remains complex128.  Only when the finest local
validation reaches the floating-point update floor do we retain the last small
corrections as an error-free high/low expansion instead of immediately rounding
``x + delta`` back to one complex128 vector.

This is a representation of the same physical edge field, not a different
operator or tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _two_sum(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    s = a + b
    bp = s - a
    e = (a - (s - bp)) + (b - bp)
    return s, e


def _renormalized_add(high, low, increment):
    """Return a two-term expansion of high + low + increment."""
    s, e = _two_sum(high, increment)
    t, f = _two_sum(low, e)
    h, g = _two_sum(s, t)
    l = g + f
    h2, l2 = _two_sum(h, l)
    return h2, l2


@dataclass(frozen=True)
class CompensatedComplexField:
    high: np.ndarray
    low: np.ndarray

    def __post_init__(self):
        high = np.asarray(self.high, dtype=np.complex128).reshape(-1)
        low = np.asarray(self.low, dtype=np.complex128).reshape(-1)
        if high.shape != low.shape:
            raise ValueError("compensated field high/low dimensions differ")
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)

    @property
    def shape(self):
        return self.high.shape

    def collapsed(self):
        return np.asarray(self.high + self.low, dtype=np.complex128).reshape(-1)

    def __array__(self, dtype=None, copy=None):
        value = self.collapsed()
        if dtype is not None:
            value = value.astype(dtype, copy=False)
        if copy is True:
            return value.copy()
        return value


def as_compensated_field(value, *, copy=True):
    if isinstance(value, CompensatedComplexField):
        if not copy:
            return value
        return CompensatedComplexField(value.high.copy(), value.low.copy())
    high = np.asarray(value, dtype=np.complex128).reshape(-1)
    if copy:
        high = high.copy()
    return CompensatedComplexField(high, np.zeros_like(high))


def field_parts(value):
    if isinstance(value, CompensatedComplexField):
        return value.high, value.low
    high = np.asarray(value, dtype=np.complex128).reshape(-1)
    return high, np.zeros_like(high)


def collapsed_field(value):
    if isinstance(value, CompensatedComplexField):
        return value.collapsed()
    return np.asarray(value, dtype=np.complex128).reshape(-1)


def compensated_add(value, increment, *, scale=1.0):
    field = as_compensated_field(value, copy=False)
    delta = np.asarray(increment, dtype=np.complex128).reshape(-1)
    if delta.shape != field.shape:
        raise ValueError("compensated field increment dimension mismatch")
    factor = float(scale)
    if not np.isfinite(factor):
        raise ValueError("compensated field scale must be finite")
    delta = factor * delta

    hr, lr = _renormalized_add(field.high.real, field.low.real, delta.real)
    hi, li = _renormalized_add(field.high.imag, field.low.imag, delta.imag)
    high = np.asarray(hr + 1j * hi, dtype=np.complex128)
    low = np.asarray(lr + 1j * li, dtype=np.complex128)
    return CompensatedComplexField(high, low)


def field_is_finite(value):
    high, low = field_parts(value)
    return bool(np.all(np.isfinite(high)) and np.all(np.isfinite(low)))


def field_norm(value):
    high, low = field_parts(value)
    # The low term is many orders smaller in the intended use.  Summing the two
    # norms avoids throwing it away while keeping this diagnostic inexpensive.
    return float(np.hypot(np.linalg.norm(high), np.linalg.norm(low)))


def field_abs2(value):
    high, low = field_parts(value)
    return np.asarray(
        high.real * high.real
        + high.imag * high.imag
        + 2.0 * (high.real * low.real + high.imag * low.imag)
        + low.real * low.real
        + low.imag * low.imag,
        dtype=float,
    )


def field_linear_dot(coefficients, value):
    coeff = np.asarray(coefficients)
    high, low = field_parts(value)
    return complex(coeff @ high + coeff @ low)


__all__ = [
    "CompensatedComplexField",
    "as_compensated_field",
    "collapsed_field",
    "compensated_add",
    "field_abs2",
    "field_is_finite",
    "field_linear_dot",
    "field_norm",
    "field_parts",
]
