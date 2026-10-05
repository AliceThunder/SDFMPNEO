from __future__ import annotations

from math import gamma
import numpy as np
from scipy.special import ellipk, ellipe, erfc

MU0 = 4e-7 * np.pi


def superellipse_area(width: float, thickness: float, exponent: float) -> float:
    a, b = 0.5 * width, 0.5 * thickness
    return 4.0 * a * b * gamma(1.0 + 1.0 / exponent) ** 2 / gamma(1.0 + 2.0 / exponent)


def dc_resistance(length: float, conductivity: float, area: float) -> float:
    if length <= 0 or conductivity <= 0 or area <= 0:
        raise ValueError("length, conductivity, and area must be positive")
    return float(length / (conductivity * area))


def coaxial_circular_mutual_inductance(radius_a: float, radius_b: float, separation: float, mu: float = MU0) -> float:
    if radius_a <= 0 or radius_b <= 0 or separation < 0 or mu <= 0:
        raise ValueError("invalid coaxial-loop parameters")
    k2 = 4.0 * radius_a * radius_b / ((radius_a + radius_b) ** 2 + separation**2)
    if not (0.0 < k2 < 1.0):
        raise ValueError("coincident/singular coaxial loops are unsupported")
    k = np.sqrt(k2)
    return float(mu * np.sqrt(radius_a * radius_b) * (((2.0 - k2) * ellipk(k2) - 2.0 * ellipe(k2)) / k))


def neumann_mutual_inductance(points_a, points_b, mu: float = MU0) -> float:
    pa = np.asarray(points_a, dtype=float)
    pb = np.asarray(points_b, dtype=float)
    da = np.diff(pa, axis=0)
    db = np.diff(pb, axis=0)
    ma = 0.5 * (pa[:-1] + pa[1:])
    mb = 0.5 * (pb[:-1] + pb[1:])
    diff = ma[:, None, :] - mb[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    if np.any(dist <= 0.0):
        raise ValueError("Neumann benchmark requires disjoint curves")
    dot = da @ db.T
    return float(mu / (4.0 * np.pi) * np.sum(dot / dist))


def thermal_impulse_temperature(radius: float, time: float, energy: float, conductivity: float, density: float, heat_capacity: float) -> float:
    if time <= 0 or conductivity <= 0 or density <= 0 or heat_capacity <= 0:
        raise ValueError("invalid thermal impulse parameters")
    alpha = conductivity / (density * heat_capacity)
    return float(energy / (density * heat_capacity * (4.0 * np.pi * alpha * time) ** 1.5) * np.exp(-(radius**2) / (4.0 * alpha * time)))


def thermal_step_temperature(radius: float, time: float, power: float, conductivity: float, density: float, heat_capacity: float) -> float:
    if radius <= 0 or time <= 0 or conductivity <= 0 or density <= 0 or heat_capacity <= 0:
        raise ValueError("invalid thermal step parameters")
    alpha = conductivity / (density * heat_capacity)
    return float(power / (4.0 * np.pi * conductivity * radius) * erfc(radius / (2.0 * np.sqrt(alpha * time))))


def lumped_thermal_step(time: float, power: float, capacity: float, conductance: float) -> float:
    if time < 0 or capacity <= 0 or conductance <= 0:
        raise ValueError("invalid lumped thermal parameters")
    return float(power / conductance * (1.0 - np.exp(-conductance * time / capacity)))
