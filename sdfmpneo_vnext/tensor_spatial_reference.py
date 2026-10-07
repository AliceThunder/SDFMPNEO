from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hybrid_field import PreparedHybridReferenceLossField
from .prediction import StructuredPortPrediction


def reciprocalize_dissipation_matrices(matrices):
    """Project Hermitian loss matrices onto the reciprocal real-symmetric cone.

    A reciprocal multiport has a real-symmetric dissipative operator.  Pointwise
    Joule matrices obtained from a finite collocation field can carry small (or,
    for difficult random scenes, large) imaginary antisymmetric cross terms.
    Taking the real part of the Hermitian projection preserves positive
    semidefiniteness while enforcing the reciprocal port convention.
    """
    value = np.asarray(matrices, dtype=complex)
    hermitian = 0.5 * (
        value
        + value.conj().swapaxes(-1, -2)
    )
    return np.asarray(
        np.real(hermitian),
        dtype=complex,
    )


class PreparedTensorEnergyReferenceLossField:
    """Energy-consistent reciprocal view of the tensor REFERENCE field."""

    def __init__(self, base: PreparedHybridReferenceLossField):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    def conductor_local_dissipation_matrix(self, *args, **kwargs):
        return reciprocalize_dissipation_matrices(
            self.base.conductor_local_dissipation_matrix(*args, **kwargs)
        )

    def local_dissipation_matrix(self, *args, **kwargs):
        return self.conductor_local_dissipation_matrix(*args, **kwargs)

    def conductor_local_joule_density(
        self,
        coil_index: int,
        arc_fraction: float,
        xy,
        currents,
    ) -> float:
        matrix = self.conductor_local_dissipation_matrix(
            coil_index,
            arc_fraction,
            xy,
        )
        currents = np.asarray(currents, dtype=complex)
        return float(
            0.5
            * np.real(
                np.vdot(currents, matrix @ currents)
            )
        )

    def raw_package_dissipation_matrices(self, package_index: int, points):
        return reciprocalize_dissipation_matrices(
            self.base.raw_package_dissipation_matrices(package_index, points)
        )

    def package_dissipation_matrices(self, package_index: int, points):
        return self.raw_package_dissipation_matrices(package_index, points)

    def package_local_dissipation_matrix(self, package_index: int, local_position):
        package = self.scene.packages[package_index]
        world = package.geometry.local_to_world(
            np.asarray(local_position, dtype=float)
        )
        return self.package_dissipation_matrices(package_index, world)

    def package_local_joule_density(
        self,
        package_index: int,
        local_position,
        currents,
    ) -> float:
        matrix = self.package_local_dissipation_matrix(
            package_index,
            local_position,
        )
        currents = np.asarray(currents, dtype=complex)
        return float(
            0.5
            * np.real(
                np.vdot(currents, matrix @ currents)
            )
        )

    def raw_background_dissipation_matrices(self, points):
        return reciprocalize_dissipation_matrices(
            self.base.raw_background_dissipation_matrices(points)
        )

    def background_dissipation_matrices(self, points):
        return self.raw_background_dissipation_matrices(points)

    def background_joule_density(self, points, currents):
        matrices = self.background_dissipation_matrices(points)
        currents = np.asarray(currents, dtype=complex)
        return 0.5 * np.real(
            np.einsum(
                "i,...ij,j->...",
                currents.conj(),
                matrices,
                currents,
            )
        )


@dataclass(frozen=True)
class TensorSpatialReferenceCalibration:
    prepared: PreparedTensorEnergyReferenceLossField
    package_volume_axial_order: int
    package_volume_radial_order: int
    package_volume_azimuthal_order: int
    background_radial_order: int
    background_angular_order: int
    refinements: int
    target_impedance: np.ndarray
    target_dissipation_channels: np.ndarray
    power_closure_error: float
    raw_closure_target_exceeded: bool


def _integrated_tensor_environment(
    prepared: PreparedTensorEnergyReferenceLossField,
    *,
    volume_axial_order: int,
    volume_radial_order: int,
    volume_azimuthal_order: int,
    background_radial_order: int,
    background_angular_order: int,
):
    scene = prepared.scene
    n_ports = prepared.port_prediction.impedance.shape[0]
    package_channels = []
    for package_index, package in enumerate(scene.packages):
        quadrature = package.geometry.volume_quadrature(
            axial_order=volume_axial_order,
            radial_order=volume_radial_order,
            azimuthal_order=volume_azimuthal_order,
        )
        matrices = prepared.raw_package_dissipation_matrices(
            package_index,
            quadrature.positions,
        )
        package_channels.append(
            np.sum(
                quadrature.weights[:, None, None] * matrices,
                axis=0,
            )
        )
    package_channels = np.asarray(
        package_channels,
        dtype=complex,
    )

    background_channel = np.zeros(
        (n_ports, n_ports),
        dtype=complex,
    )
    if scene.medium.loss_conductivity(prepared.frequency_hz) > 0.0:
        points, weights = prepared.background_quadrature(
            radial_order=background_radial_order,
            angular_order=background_angular_order,
        )
        matrices = prepared.raw_background_dissipation_matrices(points)
        background_channel = np.sum(
            weights[:, None, None] * matrices,
            axis=0,
        )

    environment = (
        np.sum(package_channels, axis=0)
        + background_channel
    )
    environment = reciprocalize_dissipation_matrices(environment)
    return package_channels, background_channel, environment


def prepare_tensor_spatial_reference_adaptive(
    teacher,
    result,
    *,
    volume_axial_order: int,
    volume_radial_order: int,
    volume_azimuthal_order: int,
    background_radial_order: int,
    background_angular_order: int,
    maximum_raw_closure_error: float,
    maximum_quadrature_refinements: int = 4,
) -> TensorSpatialReferenceCalibration:
    """Build one reciprocal energy truth shared by tensor port and spatial data.

    The tensor MFS potential is a point-collocation approximation.  Its
    ``omega*Im(V_eff)`` loss can differ structurally from the actual Joule
    integral of the reconstructed field over the physical material domain,
    especially because conductor volume is excluded from that domain.  Trying
    to force the two operators together with a congruence transform fails when
    their dissipative subspaces have different rank.

    For tensor-electric truth, the continuous Joule integral is therefore the
    canonical environment-loss operator.  The port impedance keeps the solved
    reactive part, while its dissipative part is replaced by the sum of the
    reciprocalized conductor and environment channels.  Port and pointwise
    spatial labels then use exactly the same energy definition.

    ``maximum_raw_closure_error`` remains a diagnostic target and
    ``maximum_quadrature_refinements`` remains API-compatible; neither changes
    the declared truth quadrature or rejects a finite, energy-consistent solve.
    """
    axial = int(volume_axial_order)
    radial = int(volume_radial_order)
    azimuthal = int(volume_azimuthal_order)
    background_radial = int(background_radial_order)
    background_angular = int(background_angular_order)
    maximum_quadrature_refinements = int(maximum_quadrature_refinements)
    if (
        axial < 2
        or radial < 2
        or azimuthal < 8
        or background_radial < 3
        or background_angular < 8
    ):
        raise ValueError("invalid tensor spatial truth quadrature resolution")
    if maximum_quadrature_refinements < 0:
        raise ValueError(
            "maximum_quadrature_refinements must be nonnegative"
        )
    if maximum_raw_closure_error <= 0.0:
        raise ValueError("maximum_raw_closure_error must be positive")
    if result.tensor_electric_transmission is None:
        raise ValueError(
            "tensor spatial energy truth requires a tensor-electric transmission"
        )

    n_ports = result.impedance.shape[0]
    identity = np.eye(n_ports, dtype=complex)
    temporary_base = PreparedHybridReferenceLossField(
        scene=teacher.scene,
        frequency_hz=float(teacher.frequency_hz),
        teacher=teacher,
        result=result,
        port_prediction=result.prediction,
        package_transform=identity,
        raw_dielectric_closure_error=0.0,
        normalized_dielectric_closure_error=0.0,
        package_integrated_channels=np.zeros(
            (len(teacher.scene.packages), n_ports, n_ports),
            dtype=complex,
        ),
        background_integrated_channel=np.zeros(
            (n_ports, n_ports),
            dtype=complex,
        ),
    )
    temporary = PreparedTensorEnergyReferenceLossField(temporary_base)
    (
        package_channels,
        background_channel,
        environment_channel,
    ) = _integrated_tensor_environment(
        temporary,
        volume_axial_order=axial,
        volume_radial_order=radial,
        volume_azimuthal_order=azimuthal,
        background_radial_order=background_radial,
        background_angular_order=background_angular,
    )

    original_environment = reciprocalize_dissipation_matrices(
        result.dielectric_dissipation_matrix
    )
    original_norm = max(
        float(np.linalg.norm(original_environment)),
        1e-30,
    )
    raw_error = float(
        np.linalg.norm(environment_channel - original_environment)
        / original_norm
    )

    conductor_channels = reciprocalize_dissipation_matrices(
        result.mixed_result.coil_dissipation_matrices()
    )
    channels = np.concatenate(
        (
            conductor_channels,
            environment_channel[None, :, :],
        ),
        axis=0,
    )
    dissipative = np.sum(channels, axis=0)

    solved_impedance = np.asarray(result.impedance, dtype=complex)
    reactive = (
        solved_impedance
        - solved_impedance.conj().T
    ) / (2j)
    reactive = reciprocalize_dissipation_matrices(reactive)
    corrected_impedance = dissipative + 1j * reactive

    corrected_prediction = StructuredPortPrediction(
        corrected_impedance,
        channels,
        result.prediction.channel_labels,
    )
    closure_error = float(corrected_prediction.power_closure_error())
    if not np.isfinite(closure_error) or closure_error > 1e-10:
        raise RuntimeError(
            "energy-consistent tensor port truth failed power closure: "
            f"relative error={closure_error:.3e}"
        )

    final_base = PreparedHybridReferenceLossField(
        scene=teacher.scene,
        frequency_hz=float(teacher.frequency_hz),
        teacher=teacher,
        result=result,
        port_prediction=corrected_prediction,
        package_transform=identity,
        raw_dielectric_closure_error=raw_error,
        normalized_dielectric_closure_error=closure_error,
        package_integrated_channels=package_channels,
        background_integrated_channel=background_channel,
    )
    prepared = PreparedTensorEnergyReferenceLossField(final_base)

    return TensorSpatialReferenceCalibration(
        prepared=prepared,
        package_volume_axial_order=axial,
        package_volume_radial_order=radial,
        package_volume_azimuthal_order=azimuthal,
        background_radial_order=background_radial,
        background_angular_order=background_angular,
        refinements=0,
        target_impedance=np.asarray(corrected_impedance, dtype=complex),
        target_dissipation_channels=np.asarray(channels, dtype=complex),
        power_closure_error=closure_error,
        raw_closure_target_exceeded=bool(
            raw_error > float(maximum_raw_closure_error)
        ),
    )
