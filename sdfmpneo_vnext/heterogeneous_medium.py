from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .geometry import RigidPose
from .scene import (
    EPS0,
    MU0,
    PassiveIsotropicMaterial,
    TensorElectricMaterial,
)


def _validate_frequency(frequency_hz: float) -> float:
    frequency = float(frequency_hz)
    if not np.isfinite(frequency) or frequency < 0.0:
        raise ValueError("frequency_hz must be finite and nonnegative")
    return frequency


def _material_tensors(material, frequency_hz: float, *, rotation=None):
    """Return real relative-epsilon and effective-loss tensors in world axes."""
    frequency = _validate_frequency(frequency_hz)
    if isinstance(material, TensorElectricMaterial):
        epsilon = np.asarray(material.relative_permittivity_tensor, dtype=float)
        sigma = np.asarray(
            material.loss_conductivity_tensor(frequency),
            dtype=float,
        )
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=float)
            epsilon = rotation @ epsilon @ rotation.T
            sigma = rotation @ sigma @ rotation.T
        return epsilon, sigma

    relative = complex(material.relative_permittivity_at(frequency))
    epsilon = max(float(np.real(relative)), 1e-12) * np.eye(3)
    sigma = max(float(material.loss_conductivity(frequency)), 0.0) * np.eye(3)
    return epsilon, sigma


def _material_representatives(material):
    if isinstance(material, TensorElectricMaterial):
        epsilon = float(np.trace(material.relative_permittivity_tensor) / 3.0)
        conductivity = float(
            np.max(np.linalg.eigvalsh(material.conductivity_tensor))
        )
    else:
        epsilon = float(material.relative_permittivity)
        conductivity = float(material.conductivity)
    return epsilon, float(material.relative_permeability), conductivity


@dataclass(frozen=True)
class SmoothMaterialAnchor:
    """Compact analytic material anchor for a smooth heterogeneous background.

    The influence is an anisotropic Gaussian in the anchor's local SE(3) frame.
    ``material`` supplies the local principal electric tensors; tensor materials
    rotate with the anchor.  ``strength`` controls the convex-mixture weight but
    never compromises passivity.
    """

    material: PassiveIsotropicMaterial
    length_scales: np.ndarray
    pose: RigidPose = field(default_factory=RigidPose.identity)
    strength: float = 1.0

    def __post_init__(self):
        if not isinstance(self.material, PassiveIsotropicMaterial):
            raise TypeError(
                "heterogeneous material anchor must implement the passive "
                "electric frequency-response interface"
            )
        scales = np.asarray(self.length_scales, dtype=float)
        if (
            scales.shape != (3,)
            or np.any(~np.isfinite(scales))
            or np.any(scales <= 0.0)
        ):
            raise ValueError("length_scales must be a positive finite length-3 vector")
        if not isinstance(self.pose, RigidPose):
            raise TypeError("pose must be RigidPose")
        if not np.isfinite(self.strength) or self.strength <= 0.0:
            raise ValueError("strength must be positive and finite")
        object.__setattr__(self, "length_scales", scales.copy())
        object.__setattr__(self, "strength", float(self.strength))

    def influence(self, points) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        scalar = points.shape == (3,)
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
            raise ValueError("points must have shape (...,3) and be finite")
        local = (points - self.pose.translation[None, :]) @ self.pose.rotation
        normalized = local / self.length_scales[None, :]
        value = self.strength * np.exp(
            -0.5 * np.sum(normalized * normalized, axis=1)
        )
        return value[0] if scalar else value

    def transformed(self, common: RigidPose) -> "SmoothMaterialAnchor":
        if not isinstance(common, RigidPose):
            raise TypeError("common transform must be RigidPose")
        return SmoothMaterialAnchor(
            material=self.material,
            length_scales=self.length_scales,
            pose=common.compose(self.pose),
            strength=self.strength,
        )


@dataclass(frozen=True)
class SmoothHeterogeneousMedium:
    """Mesh-free smooth heterogeneous electric background.

    At every point the electric response is a convex mixture of one unbounded
    base material and any number of SE(3)-posed Gaussian material anchors.  The
    mixture is evaluated analytically, so no voxel/grid representation is
    introduced.  Convex mixing preserves positive-definite permittivity and
    positive-semidefinite loss conductivity whenever the source materials are
    passive.

    This first heterogeneous stage keeps magnetic permeability spatially
    uniform.  That preserves the existing MQS magnetic Green kernel exactly
    while the dedicated variable-epsilon/sigma REFERENCE backend is connected.
    Existing scalar electric material methods fail closed instead of silently
    replacing this field by one effective homogeneous medium.
    """

    base_material: PassiveIsotropicMaterial
    anchors: tuple[SmoothMaterialAnchor, ...] = ()
    relative_permittivity: float = field(init=False)
    relative_permeability: float = field(init=False)
    conductivity: float = field(init=False)
    spatially_varying_electric: bool = field(default=True, init=False)

    def __post_init__(self):
        if not isinstance(self.base_material, PassiveIsotropicMaterial):
            raise TypeError(
                "base_material must implement the passive electric "
                "frequency-response interface"
            )
        anchors = tuple(self.anchors)
        if not all(isinstance(anchor, SmoothMaterialAnchor) for anchor in anchors):
            raise TypeError("anchors must contain SmoothMaterialAnchor instances")
        epsilon, permeability, conductivity = _material_representatives(
            self.base_material
        )
        for anchor in anchors:
            if not np.isclose(
                float(anchor.material.relative_permeability),
                permeability,
                rtol=1e-12,
                atol=1e-14,
            ):
                raise ValueError(
                    "SmoothHeterogeneousMedium currently requires uniform "
                    "relative_permeability across base material and anchors"
                )
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(self, "relative_permittivity", epsilon)
        object.__setattr__(self, "relative_permeability", permeability)
        object.__setattr__(self, "conductivity", conductivity)

    def _weights(self, points):
        points = np.asarray(points, dtype=float)
        scalar = points.shape == (3,)
        points = np.atleast_2d(points)
        if points.ndim != 2 or points.shape[1] != 3 or np.any(~np.isfinite(points)):
            raise ValueError("points must have shape (...,3) and be finite")
        if self.anchors:
            influence = np.column_stack(
                [np.asarray(anchor.influence(points), dtype=float) for anchor in self.anchors]
            )
        else:
            influence = np.empty((len(points), 0), dtype=float)
        denominator = 1.0 + np.sum(influence, axis=1)
        return points, influence, denominator, scalar

    def material_tensors_at(self, points, frequency_hz: float):
        """Return ``(epsilon_r, sigma_loss, mu_r)`` at world-space points."""
        frequency = _validate_frequency(frequency_hz)
        points, influence, denominator, scalar = self._weights(points)
        base_epsilon, base_sigma = _material_tensors(
            self.base_material,
            frequency,
        )
        epsilon = np.broadcast_to(base_epsilon, (len(points), 3, 3)).copy()
        sigma = np.broadcast_to(base_sigma, (len(points), 3, 3)).copy()

        for index, anchor in enumerate(self.anchors):
            anchor_epsilon, anchor_sigma = _material_tensors(
                anchor.material,
                frequency,
                rotation=anchor.pose.rotation,
            )
            weight = influence[:, index]
            epsilon += weight[:, None, None] * anchor_epsilon[None, :, :]
            sigma += weight[:, None, None] * anchor_sigma[None, :, :]

        epsilon /= denominator[:, None, None]
        sigma /= denominator[:, None, None]
        mu = np.full(len(points), self.relative_permeability, dtype=float)
        if scalar:
            return epsilon[0], sigma[0], float(mu[0])
        return epsilon, sigma, mu

    def relative_permittivity_tensor_at(self, points, frequency_hz: float):
        return self.material_tensors_at(points, frequency_hz)[0]

    def loss_conductivity_tensor_at(self, points, frequency_hz: float):
        return self.material_tensors_at(points, frequency_hz)[1]

    def relative_permeability_at_points(self, points, frequency_hz: float = 0.0):
        return self.material_tensors_at(points, frequency_hz)[2]

    def electric_coefficient_tensors(
        self,
        points,
        frequency_hz: float,
        *,
        conduction_dc: bool = False,
    ) -> np.ndarray:
        frequency = _validate_frequency(frequency_hz)
        epsilon, sigma, _ = self.material_tensors_at(points, frequency)
        if conduction_dc:
            if frequency != 0.0:
                raise ValueError(
                    "conduction_dc coefficient is defined only at exact DC"
                )
            return np.asarray(sigma, dtype=complex)
        if frequency == 0.0:
            if np.max(np.abs(sigma)) > 0.0:
                raise ValueError(
                    "conductive heterogeneous medium requires the exact DC "
                    "conduction formulation"
                )
            return EPS0 * np.asarray(epsilon, dtype=complex)
        omega = 2.0 * np.pi * frequency
        return (
            EPS0 * np.asarray(epsilon, dtype=complex)
            - 1j * np.asarray(sigma, dtype=complex) / omega
        )

    def transformed(self, common: RigidPose) -> "SmoothHeterogeneousMedium":
        """Move all finite heterogeneity anchors by one common SE(3) transform."""
        if not isinstance(common, RigidPose):
            raise TypeError("common transform must be RigidPose")
        return SmoothHeterogeneousMedium(
            self.base_material,
            tuple(anchor.transformed(common) for anchor in self.anchors),
        )

    @property
    def permeability(self) -> float:
        return MU0 * self.relative_permeability

    def complex_permittivity(self, frequency_hz: float) -> complex:
        raise NotImplementedError(
            "heterogeneous medium has no single complex permittivity; use "
            "electric_coefficient_tensors"
        )

    def relative_permittivity_at(self, frequency_hz: float) -> complex:
        raise NotImplementedError(
            "heterogeneous medium has no single relative permittivity; use "
            "relative_permittivity_tensor_at"
        )

    def loss_conductivity(self, frequency_hz: float) -> float:
        raise NotImplementedError(
            "heterogeneous medium has no single loss conductivity; use "
            "loss_conductivity_tensor_at"
        )


__all__ = [
    "SmoothMaterialAnchor",
    "SmoothHeterogeneousMedium",
]
