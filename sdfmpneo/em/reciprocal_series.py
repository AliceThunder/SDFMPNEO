from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sdfmpneo.spatial.barycentric_polynomial import (
    Polynomial,
    polynomial_add,
    polynomial_constant,
    polynomial_multiply,
    polynomial_p1,
    polynomial_scale,
)


@dataclass(frozen=True)
class ReciprocalSeriesCertificate:
    contraction_factor: float
    inverse_order: int
    inverse_square_order: int
    inverse_relative_bound: float
    inverse_square_relative_bound: float
    requested_relative_error: float

    @property
    def certified(self) -> bool:
        return (
            self.inverse_relative_bound <= self.requested_relative_error
            and self.inverse_square_relative_bound <= self.requested_relative_error
        )


def _inverse_relative_bound(q: float, order: int) -> float:
    return float(q ** (order + 1))


def _inverse_square_relative_bound(q: float, order: int) -> float:
    if q == 0.0:
        return 0.0
    return float(
        q ** (order + 1)
        * ((order + 2) + (order + 1) * q)
    )


def _minimal_order(q: float, tolerance: float, *, inverse_square: bool) -> int:
    if not 0.0 <= q < 1.0:
        raise ValueError("contraction factor must satisfy 0 <= q < 1")
    if not 0.0 < tolerance < 1.0:
        raise ValueError("relative error budget must satisfy 0 < tolerance < 1")
    order = 0
    bound = (
        _inverse_square_relative_bound(q, order)
        if inverse_square
        else _inverse_relative_bound(q, order)
    )
    while bound > tolerance:
        order += 1
        bound = (
            _inverse_square_relative_bound(q, order)
            if inverse_square
            else _inverse_relative_bound(q, order)
        )
    return order


def _series_polynomial(
    denominator_nodal: np.ndarray,
    *,
    inverse_square: bool,
    order: int,
) -> Polynomial:
    d = np.asarray(denominator_nodal, dtype=float)
    if d.shape != (4,):
        raise ValueError("denominator_nodal must have shape (4,)")
    if np.any(d <= 0.0):
        raise ValueError("reciprocal series requires a strictly positive denominator")

    d_min = float(np.min(d))
    d_max = float(np.max(d))
    center = 0.5 * (d_min + d_max)
    z = d / center - 1.0
    z_poly = polynomial_p1(z)

    total = polynomial_constant(0.0)
    power = polynomial_constant(1.0)
    for n in range(order + 1):
        coefficient = (-1.0) ** n
        if inverse_square:
            coefficient *= n + 1
        total = polynomial_add(total, polynomial_scale(power, coefficient))
        power = polynomial_multiply(power, z_poly)

    scale = 1.0 / (center * center) if inverse_square else 1.0 / center
    return polynomial_scale(total, scale)


def certified_reciprocal_polynomials(
    denominator_nodal: np.ndarray,
    *,
    requested_relative_error: float,
) -> tuple[Polynomial, Polynomial, ReciprocalSeriesCertificate]:
    """Return certified polynomial representations of 1/d and 1/d^2.

    With center=(d_max+d_min)/2 and z=d/center-1, positivity of d gives
    |z|<=q=(d_max-d_min)/(d_max+d_min)<1. The geometric series then has the
    exact relative remainder bounds

        err(1/d)   <= q^(N+1),
        err(1/d^2) <= q^(N+1)[(N+2)+(N+1)q].

    The minimal orders satisfying the declared error budget are therefore chosen
    deterministically. No near-uniformity threshold or empirical series order is
    used.
    """

    d = np.asarray(denominator_nodal, dtype=float)
    if d.shape != (4,):
        raise ValueError("denominator_nodal must have shape (4,)")
    if np.any(d <= 0.0):
        raise ValueError("denominator must remain strictly positive")
    if not 0.0 < requested_relative_error < 1.0:
        raise ValueError("requested_relative_error must satisfy 0 < error < 1")

    d_min = float(np.min(d))
    d_max = float(np.max(d))
    q = 0.0 if d_max == d_min else (d_max - d_min) / (d_max + d_min)
    inverse_order = _minimal_order(q, requested_relative_error, inverse_square=False)
    square_order = _minimal_order(q, requested_relative_error, inverse_square=True)
    inverse = _series_polynomial(d, inverse_square=False, order=inverse_order)
    inverse_square = _series_polynomial(d, inverse_square=True, order=square_order)
    certificate = ReciprocalSeriesCertificate(
        contraction_factor=float(q),
        inverse_order=inverse_order,
        inverse_square_order=square_order,
        inverse_relative_bound=_inverse_relative_bound(q, inverse_order),
        inverse_square_relative_bound=_inverse_square_relative_bound(q, square_order),
        requested_relative_error=float(requested_relative_error),
    )
    return inverse, inverse_square, certificate
