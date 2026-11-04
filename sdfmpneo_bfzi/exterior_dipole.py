"""Exact unbounded MQS magnetic-dipole benchmark outside an insulating sphere.

This is the axisymmetric toroidal l=1 sector. It supplies an exact
unbounded-domain reference and a complex transparent radial boundary
operator WITHOUT pretending the resulting Robin condition is valid for
general BFZI coils, two boxes or arbitrary vector spherical harmonics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import spherical_in, spherical_kn

from .source import MU0


@dataclass(frozen=True)
class ExteriorDipole:
    cavity_radius: float
    frequency_hz: float
    conductivity: float
    moment: float = 1.0

    def __post_init__(self):
        if (self.cavity_radius <= 0.0 or self.frequency_hz <= 0.0 or
                self.conductivity <= 0.0 or not np.isfinite(
                    [self.cavity_radius, self.frequency_hz,
                     self.conductivity, self.moment]
                ).all()):
            raise ValueError("positive radius/frequency/conductivity required")

    @property
    def omega(self):
        return 2*math.pi*self.frequency_hz

    @property
    def gamma(self):
        # exp(+i omega t): exterior field obeys (Delta-gamma**2) A=0.
        return np.sqrt(1j*self.omega*MU0*self.conductivity)

    @property
    def skin_depth(self):
        return math.sqrt(2.0/(self.omega*MU0*self.conductivity))

    def _coefficients(self, outer_radius=None):
        """Interior M/r² + D*r; exterior C*k1(gamma*r)+E*i1(gamma*r)."""
        a = self.cavity_radius
        g = self.gamma
        moment_coeff = MU0*self.moment/(4*math.pi)
        ratio = 0j
        if outer_radius is not None:
            if not outer_radius > a:
                raise ValueError("artificial sphere must enclose insulating cavity")
            ratio = spherical_kn(1,g*outer_radius)/spherical_in(
                1,g*outer_radius
            )
        ka = spherical_kn(1,g*a)
        kia = spherical_in(1,g*a)
        h = ka-ratio*kia
        dh = g*(
            spherical_kn(1,g*a,derivative=True)
            -ratio*spherical_in(1,g*a,derivative=True)
        )
        c = (3.0*moment_coeff/a**3)/(h/a-dh)
        d = (c*h-moment_coeff/a**2)/a
        return c, -ratio*c, d

    def azimuthal_potential(self, radius, *, outer_radius=None):
        """Coefficient f(r) in A_phi(r,theta) = f(r) sin(theta)."""
        r = np.asarray(radius, dtype=float)
        if not np.isfinite(r).all() or np.any(r < self.cavity_radius):
            raise ValueError("evaluate the conducting exterior, r>=cavity_radius")
        if outer_radius is not None and np.any(r>outer_radius):
            raise ValueError("finite-domain solution stops at artificial sphere")
        c,e,_ = self._coefficients(outer_radius)
        return (
            c*spherical_kn(1,self.gamma*r)
            + e*spherical_in(1,self.gamma*r)
        )

    def current_phi(self, radius, theta, *, outer_radius=None):
        """Complex peak azimuthal seawater J_phi, A/m²."""
        return (
            -1j*self.omega*self.conductivity
            * self.azimuthal_potential(radius,outer_radius=outer_radius)
            * np.sin(theta)
        )

    def joule_heat(self, radius, theta, *, outer_radius=None):
        j = self.current_phi(radius,theta,outer_radius=outer_radius)
        return np.abs(j)**2/(2*self.conductivity)

    def transparent_log_derivative(self, outer_radius):
        """Exact f'(R)/f(R) for the *unbounded* toroidal l=1 tail.

        Only the decaying spherical modified Bessel k1 is admissible.
        In general BFZI geometries a full vector-harmonic or boundary
        integral map is needed; a scalar constant Robin condition is NOT
        the correct boundary condition for arbitrary fields.
        """
        if outer_radius <= self.cavity_radius:
            raise ValueError("transparent interface must enclose the cavity")
        z = self.gamma*outer_radius
        return (
            self.gamma*spherical_kn(1,z,derivative=True)
            / spherical_kn(1,z)
        )
