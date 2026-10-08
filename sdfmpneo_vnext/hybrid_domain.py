from __future__ import annotations

"""Hybrid geometry-domain API with explicit zero-package support."""

import numpy as np

from . import _hybrid_domain_core as _core
from ._hybrid_domain_core import *  # noqa: F401,F403


def validate_hybrid_geometry_domain(
    scene,
    frequency_hz: float,
    domain,
    *,
    validated_enclosures=None,
) -> None:
    """Fail closed while allowing a declared package-count lower bound of zero.

    The original validator remains the single implementation for every actual
    geometric/material bound. This adapter only maps the new ``[0, N]``
    topology declaration onto its already-supported exact-zero or ``[1, N]``
    cases, so no finite-package checks are weakened.
    """
    if domain is None or not isinstance(domain, dict):
        return _core.validate_hybrid_geometry_domain(
            scene,
            frequency_hz,
            domain,
            validated_enclosures=validated_enclosures,
        )
    if "n_packages_range" not in domain:
        return _core.validate_hybrid_geometry_domain(
            scene,
            frequency_hz,
            domain,
            validated_enclosures=validated_enclosures,
        )

    package_count = np.asarray(domain["n_packages_range"], dtype=int)
    if (
        package_count.shape != (2,)
        or package_count[0] < 0
        or package_count[1] < package_count[0]
    ):
        raise ValueError("invalid package-count geometry domain")

    lower = int(package_count[0])
    upper = int(package_count[1])
    actual = len(scene.packages)
    if actual < lower or actual > upper:
        raise ValueError("package count is outside the hybrid artifact geometry domain")
    if lower >= 1:
        return _core.validate_hybrid_geometry_domain(
            scene,
            frequency_hz,
            domain,
            validated_enclosures=validated_enclosures,
        )

    adjusted = dict(domain)
    if actual == 0:
        adjusted.pop("n_packages_range", None)
        adjusted["n_packages"] = 0
    else:
        adjusted["n_packages_range"] = [1, upper]
    return _core.validate_hybrid_geometry_domain(
        scene,
        frequency_hz,
        adjusted,
        validated_enclosures=validated_enclosures,
    )
