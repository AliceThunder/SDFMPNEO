"""True-physics and neural stability-region sampling."""
from __future__ import annotations

import numpy as np

from .validation import StabilityReport, energy_logarithmic_norm


def analyze_stability_callbacks(
    jacobian_factory,
    thermal_operators,
    states: np.ndarray,
    geometries: np.ndarray,
    operating: np.ndarray,
) -> StabilityReport:
    """Evaluate Gate 4 on arbitrary Jacobian callbacks, including real physics."""
    a = np.asarray(states, dtype=float)
    g = np.asarray(geometries, dtype=float)
    u = np.asarray(operating, dtype=float)
    if a.ndim != 2 or g.ndim != 2 or u.ndim != 2 or not (len(a) == len(g) == len(u)):
        raise ValueError("stability samples must be aligned matrices")
    values = []
    for ai, gi, ui in zip(a, g, u):
        jacobian = np.asarray(jacobian_factory(ai, gi, ui), dtype=float)
        mass = thermal_operators.operator(gi).mass
        values.append(energy_logarithmic_norm(jacobian, mass))
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values)) if values.size else float("-inf")
    return StabilityReport(
        sample_count=int(values.size),
        maximum_logarithmic_norm=maximum,
        percentile_99_logarithmic_norm=float(np.percentile(values, 99)) if values.size else float("-inf"),
        minimum_contraction_margin=float(-maximum),
        uniformly_contractive_on_samples=bool(values.size and maximum < 0.0),
    )


__all__ = ["analyze_stability_callbacks"]
