from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.polynomial.legendre import leggauss


@dataclass(frozen=True)
class SectionQuadrature:
    xy: np.ndarray
    weights: np.ndarray
    rho: np.ndarray
    theta: np.ndarray


def superellipse_section_quadrature(width: float, thickness: float, exponent: float, radial_order: int = 5, angular_order: int = 32) -> SectionQuadrature:
    if width <= 0 or thickness <= 0 or exponent < 2:
        raise ValueError("invalid superellipse section")
    if radial_order < 2 or angular_order < 8:
        raise ValueError("quadrature orders are too small")
    xg, wg = leggauss(radial_order)
    rho = 0.5 * (xg + 1.0)
    wr = 0.5 * wg
    theta = (np.arange(angular_order) + 0.5) * (2.0 * np.pi / angular_order)
    wt = np.full(angular_order, 2.0 * np.pi / angular_order)
    rr, tt = np.meshgrid(rho, theta, indexing="ij")
    wrr, wtt = np.meshgrid(wr, wt, indexing="ij")
    a, b = 0.5 * width, 0.5 * thickness
    denom = (np.abs(np.cos(tt)) / a) ** exponent + (np.abs(np.sin(tt)) / b) ** exponent
    rb = denom ** (-1.0 / exponent)
    x = rr * rb * np.cos(tt)
    y = rr * rb * np.sin(tt)
    jac = rr * rb**2
    weights = wrr * wtt * jac
    return SectionQuadrature(
        np.stack((x.ravel(), y.ravel()), axis=1),
        weights.ravel(), rr.ravel(), tt.ravel(),
    )


def _monomial_powers(degree: int):
    out = []
    for total in range(degree + 1):
        for px in range(total + 1):
            out.append((px, total - px))
    return out


@dataclass(frozen=True)
class SectionBasis:
    quadrature: SectionQuadrature
    values: np.ndarray
    moments: np.ndarray
    degree: int

    @property
    def n_modes(self) -> int:
        return self.values.shape[1]

    @property
    def area(self) -> float:
        return float(np.sum(self.quadrature.weights))


def polynomial_section_basis(width: float, thickness: float, exponent: float, degree: int = 1, radial_order: int = 5, angular_order: int = 32) -> SectionBasis:
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    q = superellipse_section_quadrature(width, thickness, exponent, radial_order, angular_order)
    x = q.xy[:, 0] / (0.5 * width)
    y = q.xy[:, 1] / (0.5 * thickness)
    raw = np.stack([x**px * y**py for px, py in _monomial_powers(degree)], axis=1)
    W = q.weights[:, None]
    gram = raw.T @ (W * raw)
    evals, evecs = np.linalg.eigh(gram)
    scale = max(float(np.max(evals)), 1.0)
    keep = evals > 1e-13 * scale
    if not np.any(keep):
        raise ValueError("cross-section basis is numerically empty")
    transform = evecs[:, keep] / np.sqrt(evals[keep])[None, :]
    values = raw @ transform
    moments = values.T @ q.weights
    order = np.argsort(-np.abs(moments))
    values = values[:, order]
    moments = moments[order]
    return SectionBasis(q, values, moments, degree)
