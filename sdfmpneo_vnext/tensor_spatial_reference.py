from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace

import numpy as np

from .exterior_quadrature import conductor_volume_mask
from .hybrid_domain import package_domain_topology
from .hybrid_field import PreparedHybridReferenceLossField
from .prediction import StructuredPortPrediction
from .scene import TensorElectricMaterial


def _hermitian(matrices):
    value = np.asarray(matrices, dtype=complex)
    return 0.5 * (value + value.conj().swapaxes(-1, -2))


def reciprocalize_dissipation_matrices(matrices):
    """Project an integrated Hermitian loss operator to reciprocal resistance."""
    return np.asarray(np.real(_hermitian(matrices)), dtype=complex)


def _stable_cholesky(matrix):
    matrix = _hermitian(matrix)
    try:
        return np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        diagonal = np.real(np.diag(matrix))
        scale = max(
            float(np.max(np.abs(diagonal))) if diagonal.size else 0.0,
            float(np.linalg.norm(matrix)) / max(matrix.shape[0], 1),
            np.finfo(float).tiny,
        )
        eye = np.eye(matrix.shape[0], dtype=complex)
        for relative in (1e-14, 1e-12, 1e-10, 1e-8):
            try:
                return np.linalg.cholesky(
                    matrix + relative * scale * eye
                )
            except np.linalg.LinAlgError:
                continue
    raise RuntimeError(
        "tensor energy loss operator is not positive semidefinite enough for "
        "the reciprocal energy projection"
    )


def _energy_congruence(raw_total, reciprocal_target):
    """Map raw total Joule energy to reciprocal resistance with one transform."""
    raw_factor = _stable_cholesky(raw_total)
    target_factor = _stable_cholesky(reciprocal_target)
    inverse_raw = np.linalg.solve(
        raw_factor,
        np.eye(raw_factor.shape[0], dtype=complex),
    )
    return target_factor @ inverse_raw


def _apply_energy_transform(matrices, transform):
    value = np.asarray(matrices, dtype=complex)
    corrected = transform @ value @ transform.conj().T
    return _hermitian(corrected)


def _material_loss_tensor(material, frequency_hz: float, *, rotation=None):
    if isinstance(material, TensorElectricMaterial):
        tensor = np.asarray(
            material.loss_conductivity_tensor(frequency_hz),
            dtype=float,
        )
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=float)
            tensor = rotation @ tensor @ rotation.T
        return tensor
    return (
        float(material.loss_conductivity(frequency_hz))
        * np.eye(3, dtype=float)
    )


class PreparedTensorEnergyReferenceLossField:
    """Energy-consistent reciprocal view of the tensor REFERENCE field.

    Raw pointwise Joule matrices remain Hermitian PSD, including their local
    phase-sensitive cross-port terms.  A single global congruence is applied to
    every conductor/package/background matrix so the integrated total equals a
    reciprocal real-symmetric port resistance without independently distorting
    each spatial point or loss mechanism.
    """

    def __init__(self, base: PreparedHybridReferenceLossField):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def energy_transform(self):
        return np.asarray(self.base.package_transform, dtype=complex)

    def transform_dissipation_matrices(self, matrices):
        return _apply_energy_transform(matrices, self.energy_transform)

    def raw_conductor_local_dissipation_matrix(self, *args, **kwargs):
        return _hermitian(
            self.base.conductor_local_dissipation_matrix(*args, **kwargs)
        )

    def conductor_local_dissipation_matrix(self, *args, **kwargs):
        return self.transform_dissipation_matrices(
            self.raw_conductor_local_dissipation_matrix(*args, **kwargs)
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
        return float(0.5 * np.real(np.vdot(currents, matrix @ currents)))

    def raw_package_dissipation_matrices(self, package_index: int, points):
        if not 0 <= package_index < len(self.scene.packages):
            raise IndexError("package_index out of range")
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        package = self.scene.packages[package_index]
        topology = package_domain_topology(self.scene.packages)
        region = np.asarray(
            topology.deepest_containing(
                self.scene.packages,
                points,
                tolerance=2e-12,
            ),
            dtype=int,
        )
        inside = region == package_index
        if np.any(inside):
            inside &= ~np.asarray(
                conductor_volume_mask(
                    self.scene,
                    points,
                    segments=self.teacher.conductor_teacher._mqs._segments,
                ),
                dtype=bool,
            )

        n_ports = self.port_prediction.impedance.shape[0]
        out = np.zeros((len(points), n_ports, n_ports), dtype=complex)
        if np.any(inside):
            conductivity = _material_loss_tensor(
                package.material,
                self.frequency_hz,
                rotation=package.geometry.pose.rotation,
            )
            if np.linalg.norm(conductivity) > 0.0:
                transfer = self.electric_field_transfer(points[inside])
                if transfer.ndim == 2:
                    transfer = transfer[None, :, :]
                matrices = np.einsum(
                    "qdi,de,qej->qij",
                    transfer.conj(),
                    conductivity,
                    transfer,
                )
                out[inside] = _hermitian(matrices)
        return out[0] if scalar else out

    def package_dissipation_matrices(self, package_index: int, points):
        return self.transform_dissipation_matrices(
            self.raw_package_dissipation_matrices(package_index, points)
        )

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
        return float(0.5 * np.real(np.vdot(currents, matrix @ currents)))

    def raw_background_dissipation_matrices(self, points):
        points = np.asarray(points, dtype=float)
        scalar = points.ndim == 1
        points = np.atleast_2d(points)
        n_ports = self.port_prediction.impedance.shape[0]
        out = np.zeros((len(points), n_ports, n_ports), dtype=complex)
        conductivity = _material_loss_tensor(
            self.scene.medium,
            self.frequency_hz,
        )
        if np.linalg.norm(conductivity) <= 0.0:
            return out[0] if scalar else out

        exterior = np.asarray(
            self.base._background_domain_mask(points),
            dtype=bool,
        )
        if np.any(exterior):
            transfer = self.electric_field_transfer(points[exterior])
            if transfer.ndim == 2:
                transfer = transfer[None, :, :]
            matrices = np.einsum(
                "qdi,de,qej->qij",
                transfer.conj(),
                conductivity,
                transfer,
            )
            out[exterior] = _hermitian(matrices)
        return out[0] if scalar else out

    def background_dissipation_matrices(self, points):
        return self.transform_dissipation_matrices(
            self.raw_background_dissipation_matrices(points)
        )

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
    ).reshape((-1, n_ports, n_ports))

    background_channel = np.zeros((n_ports, n_ports), dtype=complex)
    background_loss = _material_loss_tensor(
        scene.medium,
        prepared.frequency_hz,
    )
    if np.linalg.norm(background_loss) > 0.0:
        points, weights = prepared.background_quadrature(
            radial_order=background_radial_order,
            angular_order=background_angular_order,
        )
        matrices = prepared.raw_background_dissipation_matrices(points)
        background_channel = np.sum(
            weights[:, None, None] * matrices,
            axis=0,
        )

    environment = _hermitian(
        np.sum(package_channels, axis=0) + background_channel
    )
    return package_channels, _hermitian(background_channel), environment


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

    The raw pointwise loss matrices are integrated first without discarding
    phase-sensitive cross-port terms.  One common congruence is then applied to
    every loss mechanism and every spatial point so the total dissipative port
    operator is reciprocal and power-closing.  This mirrors the hard passive
    normalization used by the FAST port decoder while preserving much more of
    the reconstructed local Joule field than pointwise real projection.

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
            "maximum_spatial_quadrature_refinements must be nonnegative"
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
        raw_package_channels,
        raw_background_channel,
        raw_environment_channel,
    ) = _integrated_tensor_environment(
        temporary,
        volume_axial_order=axial,
        volume_radial_order=radial,
        volume_azimuthal_order=azimuthal,
        background_radial_order=background_radial,
        background_angular_order=background_angular,
    )

    original_environment = _hermitian(result.dielectric_dissipation_matrix)
    original_norm = max(float(np.linalg.norm(original_environment)), 1e-30)
    raw_error = float(
        np.linalg.norm(raw_environment_channel - original_environment)
        / original_norm
    )

    raw_conductor_channels = _hermitian(
        result.mixed_result.coil_dissipation_matrices()
    )
    raw_channels = np.concatenate(
        (
            raw_conductor_channels,
            raw_environment_channel[None, :, :],
        ),
        axis=0,
    )
    raw_total = _hermitian(np.sum(raw_channels, axis=0))
    target_resistance = reciprocalize_dissipation_matrices(raw_total)
    transform = _energy_congruence(raw_total, target_resistance)
    corrected_channels = _apply_energy_transform(raw_channels, transform)
    corrected_total = _hermitian(np.sum(corrected_channels, axis=0))
    energy_closure = float(
        np.linalg.norm(corrected_total - target_resistance)
        / max(float(np.linalg.norm(target_resistance)), 1e-30)
    )
    if not np.isfinite(energy_closure) or energy_closure > 1e-8:
        raise RuntimeError(
            "tensor reciprocal energy projection failed total power closure: "
            f"relative error={energy_closure:.3e}"
        )

    solved_impedance = np.asarray(result.impedance, dtype=complex)
    reactive = reciprocalize_dissipation_matrices(
        (solved_impedance - solved_impedance.conj().T) / (2j)
    )
    corrected_impedance = target_resistance + 1j * reactive

    corrected_prediction = StructuredPortPrediction(
        corrected_impedance,
        corrected_channels,
        result.prediction.channel_labels,
    )
    closure_error = float(corrected_prediction.power_closure_error())
    if not np.isfinite(closure_error) or closure_error > 1e-8:
        raise RuntimeError(
            "energy-consistent tensor port truth failed power closure: "
            f"relative error={closure_error:.3e}"
        )

    corrected_environment = corrected_channels[-1]
    corrected_package_channels = _apply_energy_transform(
        raw_package_channels,
        transform,
    )
    corrected_background_channel = _apply_energy_transform(
        raw_background_channel,
        transform,
    )

    corrected_result = result
    if is_dataclass(result) and is_dataclass(result.mixed_result):
        corrected_mixed_result = replace(
            result.mixed_result,
            impedance=np.asarray(corrected_impedance, dtype=complex),
        )
        corrected_result = replace(
            result,
            mixed_result=corrected_mixed_result,
            prediction=corrected_prediction,
            dielectric_dissipation_matrix=np.asarray(
                corrected_environment,
                dtype=complex,
            ),
        )

    final_base = PreparedHybridReferenceLossField(
        scene=teacher.scene,
        frequency_hz=float(teacher.frequency_hz),
        teacher=teacher,
        result=corrected_result,
        port_prediction=corrected_prediction,
        package_transform=transform,
        raw_dielectric_closure_error=raw_error,
        normalized_dielectric_closure_error=max(
            closure_error,
            energy_closure,
        ),
        package_integrated_channels=corrected_package_channels,
        background_integrated_channel=corrected_background_channel,
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
        target_dissipation_channels=np.asarray(corrected_channels, dtype=complex),
        power_closure_error=max(closure_error, energy_closure),
        raw_closure_target_exceeded=bool(
            raw_error > float(maximum_raw_closure_error)
        ),
    )
