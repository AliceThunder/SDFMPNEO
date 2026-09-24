from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .scene import Scene


@dataclass(frozen=True)
class AnalyticBaselineResult:
    impedance: np.ndarray
    resistance: np.ndarray
    inductance: np.ndarray


def _polyline_elements(geometry, n_segments: int):
    poly = geometry.polyline(n_segments)
    dl = np.diff(poly.points, axis=0)
    mid = poly.midpoints
    return poly, mid, dl


def regularized_self_inductance(
    geometry,
    *,
    permeability: float,
    n_segments: int = 96,
    regularization_radius: float | None = None,
) -> float:
    if n_segments < 8:
        raise ValueError("n_segments must be >= 8")
    _, mid, dl = _polyline_elements(
        geometry,
        n_segments,
    )
    radius = (
        geometry.equivalent_radius
        if regularization_radius is None
        else float(regularization_radius)
    )
    if radius <= 0.0:
        raise ValueError(
            "regularization_radius must be positive"
        )
    diff = (
        mid[:, None, :]
        - mid[None, :, :]
    )
    dist2 = np.sum(
        diff * diff,
        axis=2,
    )
    kernel = 1.0 / np.sqrt(
        dist2 + radius**2
    )
    dot = dl @ dl.T
    value = (
        permeability
        / (4.0 * np.pi)
        * np.sum(dot * kernel)
    )
    return float(
        max(np.real(value), 0.0)
    )


def pair_mutual_inductance(
    geometry_a,
    geometry_b,
    *,
    permeability: float,
    n_segments: int = 96,
) -> float:
    if n_segments < 8:
        raise ValueError(
            "n_segments must be >= 8"
        )
    _, ma, da = _polyline_elements(
        geometry_a,
        n_segments,
    )
    _, mb, db = _polyline_elements(
        geometry_b,
        n_segments,
    )
    diff = (
        ma[:, None, :]
        - mb[None, :, :]
    )
    dist = np.linalg.norm(
        diff,
        axis=2,
    )
    if np.any(dist <= 1e-14):
        raise ValueError(
            "pair baseline requires disjoint conductor centerlines"
        )
    value = (
        permeability
        / (4.0 * np.pi)
        * np.sum(
            (da @ db.T) / dist
        )
    )
    return float(np.real(value))


def analytic_port_baseline(
    scene: Scene,
    frequency_hz: float,
    *,
    segments_per_coil: int = 96,
) -> AnalyticBaselineResult:
    if (
        not np.isfinite(frequency_hz)
        or frequency_hz < 0.0
    ):
        raise ValueError(
            "frequency_hz must be finite and nonnegative"
        )
    n = len(scene.coils)
    R = np.zeros(
        (n, n),
        dtype=float,
    )
    L = np.zeros(
        (n, n),
        dtype=float,
    )
    mu = scene.medium.permeability
    for i, coil in enumerate(
        scene.coils
    ):
        poly = coil.geometry.polyline(
            segments_per_coil
        )
        R[i, i] = (
            poly.total_length
            / (
                coil.material.conductivity
                * coil.geometry.cross_section_area
            )
        )
        L[i, i] = regularized_self_inductance(
            coil.geometry,
            permeability=mu,
            n_segments=segments_per_coil,
        )
    for i in range(n):
        for j in range(i):
            mutual = pair_mutual_inductance(
                scene.coils[i].geometry,
                scene.coils[j].geometry,
                permeability=mu,
                n_segments=segments_per_coil,
            )
            L[i, j] = mutual
            L[j, i] = mutual
    omega = 2.0 * np.pi * float(
        frequency_hz
    )
    Z = R.astype(complex) + 1j * omega * L
    return AnalyticBaselineResult(
        Z,
        R,
        L,
    )
