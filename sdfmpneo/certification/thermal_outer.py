from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HomogeneousThermalExterior:
    """Proof assumptions for an infinite homogeneous seawater thermal exterior.

    Outside ``source_support_radius`` the medium must be homogeneous, isotropic,
    source free except for heat conducted outward from the enclosed system, and
    extend to infinity.  The artificial boundary is the sphere of
    ``outer_radius``.  The material constants are lower bounds valid throughout
    the exterior.
    """

    source_support_radius: float
    outer_radius: float
    thermal_conductivity_min: float
    volumetric_heat_capacity_min: float
    certified_geometry: bool
    provenance: str

    def __post_init__(self) -> None:
        values = np.array(
            [
                self.source_support_radius,
                self.outer_radius,
                self.thermal_conductivity_min,
                self.volumetric_heat_capacity_min,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("thermal exterior parameters must be finite and positive")
        if not self.source_support_radius < self.outer_radius:
            raise ValueError("source_support_radius must be smaller than outer_radius")
        if self.certified_geometry and not str(self.provenance).strip():
            raise ValueError("certified thermal exterior requires provenance")

    @property
    def clearance(self) -> float:
        return float(self.outer_radius - self.source_support_radius)


@dataclass(frozen=True)
class ThermalOuterDomainErrorCertificate:
    clearance: float
    initial_boundary_temperature_bound: float
    forced_boundary_temperature_bound: float
    boundary_temperature_error_bound: float
    state_mass_error_bound: float
    certified: bool
    provenance: str


def certify_thermal_outer_domain(
    exterior: HomogeneousThermalExterior,
    *,
    initial_absolute_excess_heat: float,
    heat_power_upper_bound: float,
    computational_total_heat_capacity: float,
) -> ThermalOuterDomainErrorCertificate:
    """Bound thermal truncation error on a spherical infinite-water exterior.

    Let ``d=R-R_s`` be the minimum distance from all initial/source support to the
    artificial boundary.  Positivity of the 3-D heat kernel gives, for any
    initial absolute excess heat ``E0``, the all-time pointwise bound

        theta_0 <= (3/(2*pi*e))^(3/2) E0 / ((rho c)_min d^3).

    For a non-negative or absolute heat-source power bounded by ``Pmax``, the
    transient Newton potential is bounded by its steady Green potential,

        theta_Q <= Pmax / (4*pi*k_min*d).

    Their sum bounds the true infinite-domain temperature on the artificial
    sphere for all time.  With the truncated problem imposing the ambient value
    there, the parabolic maximum principle bounds the interior temperature error
    by the same quantity.  The discrete/continuous thermal mass norm over the
    computational domain is then at most ``sqrt(C_total)*theta_boundary``.

    No outer-radius convergence fit is used.  Certification is enabled only when
    the homogeneous infinite-exterior assumptions themselves are certified.
    """

    E0 = float(initial_absolute_excess_heat)
    P = float(heat_power_upper_bound)
    C = float(computational_total_heat_capacity)
    if E0 < 0.0 or P < 0.0 or C <= 0.0:
        raise ValueError("heat bounds must be non-negative and total heat capacity positive")
    if not np.isfinite(E0 + P + C):
        raise ValueError("heat bounds must be finite")

    d = exterior.clearance
    initial = (
        (3.0 / (2.0 * np.pi * np.e)) ** 1.5
        * E0
        / (exterior.volumetric_heat_capacity_min * d**3)
    )
    forced = P / (4.0 * np.pi * exterior.thermal_conductivity_min * d)
    boundary = float(np.nextafter(initial + forced, np.inf))
    mass = float(np.nextafter(np.sqrt(C) * boundary, np.inf))
    return ThermalOuterDomainErrorCertificate(
        clearance=d,
        initial_boundary_temperature_bound=float(np.nextafter(initial, np.inf)),
        forced_boundary_temperature_bound=float(np.nextafter(forced, np.inf)),
        boundary_temperature_error_bound=boundary,
        state_mass_error_bound=mass,
        certified=bool(exterior.certified_geometry),
        provenance=str(exterior.provenance),
    )
