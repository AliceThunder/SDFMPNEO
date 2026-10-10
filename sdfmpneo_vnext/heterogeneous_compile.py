from __future__ import annotations

import numpy as np

from .graded_material import compile_graded_superquadric_regions
from .heterogeneous_medium import SmoothHeterogeneousMedium, SmoothMaterialAnchor
from .hybrid_domain import (
    package_domain_topology,
    validate_package_conductor_topology,
)
from .package_geometry import SuperquadricPackageGeometry
from .scene import (
    HomogeneousMedium,
    IsotropicMaterial,
    Scene,
    TensorElectricMaterial,
)


def _static_electric_tensors(material):
    """Return frequency-independent electric tensors for compile-time shells."""
    if isinstance(material, TensorElectricMaterial):
        return (
            np.asarray(material.relative_permittivity_tensor, dtype=float),
            np.asarray(material.conductivity_tensor, dtype=float),
        )
    if isinstance(material, (HomogeneousMedium, IsotropicMaterial)):
        return (
            float(material.relative_permittivity) * np.eye(3),
            float(material.conductivity) * np.eye(3),
        )
    raise TypeError(
        "smooth heterogeneous shell compilation currently supports static "
        "HomogeneousMedium, IsotropicMaterial, or TensorElectricMaterial "
        "responses"
    )


def _anchor_shell_profile(
    base_material,
    anchor: SmoothMaterialAnchor,
    *,
    cutoff_sigma: float,
):
    """Build a radial material callable in the anchor's local frame."""
    base_epsilon_world, base_sigma_world = _static_electric_tensors(base_material)
    target_epsilon_local, target_sigma_local = _static_electric_tensors(
        anchor.material
    )
    rotation = np.asarray(anchor.pose.rotation, dtype=float)
    base_epsilon_local = rotation.T @ base_epsilon_world @ rotation
    base_sigma_local = rotation.T @ base_sigma_world @ rotation
    permeability = float(base_material.relative_permeability)

    def material_at(normalized_radius: float):
        radius = float(normalized_radius)
        if not np.isfinite(radius) or radius < 0.0 or radius > 1.0:
            raise ValueError("normalized_radius must lie in [0,1]")
        influence = float(anchor.strength) * np.exp(
            -0.5 * (float(cutoff_sigma) * radius) ** 2
        )
        denominator = 1.0 + influence
        epsilon = (
            base_epsilon_local + influence * target_epsilon_local
        ) / denominator
        sigma = (
            base_sigma_local + influence * target_sigma_local
        ) / denominator
        return TensorElectricMaterial(
            relative_permittivity_tensor=epsilon,
            conductivity_tensor=sigma,
            relative_permeability=permeability,
        )

    return material_at


def compile_smooth_heterogeneous_scene(
    scene: Scene,
    *,
    shell_count: int = 8,
    cutoff_sigma: float = 4.0,
    name_prefix: str = "heterogeneous",
    clearance_fraction: float = 0.03,
    longitudinal_segments: int = 96,
    section_points: int = 24,
) -> Scene:
    """Compile smooth electric heterogeneity into nested mesh-free interfaces.

    Each Gaussian anchor is truncated at ``cutoff_sigma`` local standard scales
    and represented by homothetic ellipsoidal shells.  Multiple anchors may be
    used when their compiled regions are strictly disjoint or nested according
    to the existing material-topology contract.  Partial overlaps are rejected
    rather than silently approximated.

    The returned scene contains only the established homogeneous/tensor material
    objects and can therefore be consumed by the existing package-aware
    REFERENCE/CERTIFIED solvers.  Increasing ``shell_count`` refines the smooth
    profile without introducing a volume mesh.
    """
    if not isinstance(scene, Scene):
        raise TypeError("scene must be Scene")
    if not isinstance(scene.medium, SmoothHeterogeneousMedium):
        raise TypeError(
            "scene.medium must be SmoothHeterogeneousMedium for heterogeneous compilation"
        )
    if not isinstance(shell_count, (int, np.integer)) or shell_count < 2:
        raise ValueError("shell_count must be an integer >= 2")
    cutoff = float(cutoff_sigma)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff_sigma must be positive and finite")

    medium = scene.medium
    compiled_packages = list(scene.packages)
    for anchor_index, anchor in enumerate(medium.anchors):
        outer_geometry = SuperquadricPackageGeometry(
            half_extents=cutoff * np.asarray(anchor.length_scales, dtype=float),
            exponent_xy=2.0,
            exponent_z=2.0,
            pose=anchor.pose,
        )

        enclosed_coils = []
        for coil in scene.coils:
            classification = outer_geometry.classify_conductor(
                coil.geometry,
                longitudinal_segments=longitudinal_segments,
                section_points=section_points,
                tolerance=1e-10,
            )
            if classification == "inside":
                enclosed_coils.append(coil)

        profile = _anchor_shell_profile(
            medium.base_material,
            anchor,
            cutoff_sigma=cutoff,
        )
        compiled_packages.extend(
            compile_graded_superquadric_regions(
                outer_geometry,
                profile,
                shell_count=int(shell_count),
                name_prefix=f"{name_prefix}:{anchor_index:03d}",
                enclosed_coils=tuple(enclosed_coils),
                clearance_fraction=float(clearance_fraction),
                longitudinal_segments=int(longitudinal_segments),
                section_points=int(section_points),
            )
        )

    compiled = Scene(
        scene.coils,
        medium.base_material,
        tuple(compiled_packages),
    )
    package_domain_topology(compiled.packages)
    validate_package_conductor_topology(compiled)
    return compiled


__all__ = ["compile_smooth_heterogeneous_scene"]
