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
from scipy.integrate import quad
from scipy.special import gammaln, lpmv, spherical_in, spherical_kn

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


def _complex_spherical_harmonic(degree, order, mu, phi):
    """Orthonormal complex Y_lm(mu,phi), with Condon-Shortley phase."""
    m = abs(order)
    amplitude = np.sqrt(
        (2*degree+1)/(4*math.pi)
        * np.exp(gammaln(degree-m+1)-gammaln(degree+m+1))
    )
    positive = amplitude*lpmv(m, degree, mu)*np.exp(1j*m*phi)
    return positive if order >= 0 else (-1)**m*np.conj(positive)


@dataclass(frozen=True)
class ExteriorSphericalCurrent:
    """Exact outgoing homogeneous-seawater current tail in a spherical basis.

    Outside any sphere containing both BFZI enclosures and all physical
    sources, each Cartesian J component satisfies
        (Delta - gamma**2) J_i = 0,  gamma**2 = i*omega*mu0*sigma.
    Thus each scalar spherical-harmonic coefficient decays as
    k_l(gamma*r)/k_l(gamma*R). The exterior DtN is NONLOCAL in angle,
    diagonal only after harmonic projection, and applies to any boundary
    trace compatible with div(J)=0. The l=1 dipole is one special sector.

    This class computes exact homogeneous exterior continuation, radial
    DtN and tail Joule loss GIVEN a compatible current trace. It is not
    yet a coupled two-box interior solve: projecting a boundary trace
    from a finite truncated solution alone does not fix its truncation
    error, nor does arbitrary Dirichlet data guarantee div(J)=0.
    """

    radius: float
    frequency_hz: float
    conductivity: float
    max_degree: int = 4

    def __post_init__(self):
        if (not np.isfinite([self.radius, self.frequency_hz,
                             self.conductivity]).all()
                or min(self.radius, self.frequency_hz,
                       self.conductivity) <= 0):
            raise ValueError("positive finite exterior sphere and material required")
        if (not isinstance(self.max_degree, int)
                or isinstance(self.max_degree, bool)
                or not 1 <= self.max_degree <= 32):
            raise ValueError("max_degree must be an integer from 1 through 32")

    @property
    def gamma(self):
        return np.sqrt(1j*2*math.pi*self.frequency_hz*MU0*self.conductivity)

    @property
    def modes(self):
        return tuple(
            (l, m) for l in range(self.max_degree+1) for m in range(-l, l+1)
        )

    def angular_nodes(self, polar_order=None, azimuthal_order=None):
        """Gauss(mu) x periodic(phi), exact harmonic products to max_degree."""
        polar = self.max_degree+2 if polar_order is None else polar_order
        azimuth = 2*self.max_degree+3 if azimuthal_order is None else azimuthal_order
        if (not isinstance(polar, int) or not isinstance(azimuth, int)
                or polar < self.max_degree+1
                or azimuth < 2*self.max_degree+1):
            raise ValueError("angular grid cannot resolve requested harmonics")
        mu, wm = np.polynomial.legendre.leggauss(polar)
        phi = 2*math.pi*(np.arange(azimuth)+0.5)/azimuth
        mu_flat = np.repeat(mu, azimuth)
        phi_flat = np.tile(phi, polar)
        weights = np.repeat(wm, azimuth)*(2*math.pi/azimuth)
        return mu_flat, phi_flat, weights

    def _harmonics(self, mu, phi):
        return np.column_stack([
            _complex_spherical_harmonic(l, m, mu, phi)
            for l, m in self.modes
        ])

    def project(self, boundary_current, polar_order=None, azimuthal_order=None):
        """Project J(R,theta,phi) Cartesian samples, return (coeff, L2 error).

        Data must use angular_nodes(polar_order,azimuthal_order), in that
        order, and must represent a physically divergence-free trace for
        the reconstructed field to be a Maxwell current.
        """
        mu, phi, w = self.angular_nodes(polar_order, azimuthal_order)
        current = np.asarray(boundary_current, complex)
        if (current.shape != (len(mu), 3)
                or not np.isfinite(current).all()):
            raise ValueError("expected finite (n_angular_nodes,3) current trace")
        y = self._harmonics(mu, phi)
        coeff = y.conj().T @ (w[:, None]*current)
        error = np.sqrt(
            np.sum(w[:, None]*np.abs(current-y@coeff)**2)
            /max(np.sum(w[:, None]*np.abs(current)**2), 1e-60)
        )
        return coeff, float(error)

    def log_derivatives(self):
        """Exact exterior radial DtN eigenvalues for every (l,m) block."""
        g = self.gamma
        z = g*self.radius
        return np.array([
            g*spherical_kn(l,z,derivative=True)/spherical_kn(l,z)
            for l, _ in self.modes
        ])

    def evaluate(self, coefficients, radius, mu, phi):
        """Evaluate the unbounded current tail at r>=matching radius."""
        coeff = np.asarray(coefficients, complex)
        if coeff.shape != (len(self.modes), 3):
            raise ValueError("incorrect spherical current coefficient shape")
        r, u, a = np.broadcast_arrays(
            np.asarray(radius, float), np.asarray(mu, float),
            np.asarray(phi, float),
        )
        if (not np.isfinite(r).all() or not np.isfinite(u).all()
                or not np.isfinite(a).all() or np.any(r < self.radius)
                or np.any(np.abs(u)>1)):
            raise ValueError("exterior query requires r>=R and -1<=mu<=1")
        harmonics = self._harmonics(u.ravel(), a.ravel())
        modes = self.modes
        radial = np.column_stack([
            spherical_kn(l, self.gamma*r.ravel())
            /spherical_kn(l, self.gamma*self.radius)
            for l, _ in modes
        ])
        result = (harmonics*radial) @ coeff
        return result.reshape(r.shape+(3,))

    def boundary_radial_derivative(self, coefficients, mu, phi):
        """Nonlocal exterior dJ/dr induced by supplied boundary coefficients."""
        coeff = np.asarray(coefficients, complex)
        if coeff.shape != (len(self.modes), 3):
            raise ValueError("incorrect spherical current coefficient shape")
        u, a = np.broadcast_arrays(np.asarray(mu, float), np.asarray(phi, float))
        result = self._harmonics(u.ravel(),a.ravel()) @ (
            self.log_derivatives()[:,None]*coeff
        )
        return result.reshape(u.shape+(3,))

    def tail_ohmic_power(self, coefficients):
        """Exact-angle external Joule power by 1D radial integration.

        The orthonormal Y_lm basis removes all cross terms from
        integral_{r>R}|J|^2/(2*sigma) dV. The remaining radial integral
        is evaluated adaptively and is independent of the finite-domain
        x/y magnetic cubature.
        """
        coeff = np.asarray(coefficients, complex)
        if coeff.shape != (len(self.modes), 3):
            raise ValueError("incorrect spherical current coefficient shape")
        g = self.gamma
        energy = 0.0
        for l in range(self.max_degree+1):
            k0 = spherical_kn(l, g*self.radius)
            norm = sum(
                float(np.vdot(coeff[i],coeff[i]).real)
                for i,(degree,_) in enumerate(self.modes) if degree == l
            )
            if norm == 0.0:
                continue
            integral, _ = quad(
                lambda r: r*r*abs(spherical_kn(l,g*r)/k0)**2,
                self.radius, np.inf, epsrel=1e-9,
            )
            energy += norm*integral
        return energy/(2*self.conductivity)
