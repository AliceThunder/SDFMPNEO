from __future__ import annotations

import numpy as np

from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
    DielectricCoupledReferenceArtifact,
)
from .hybrid_field import (
    PreparedHybridReferenceLossField,
    _hermitian_psd_sqrt,
)
from .scene import (
    Scene,
    TensorElectricMaterial,
)


class PreparedTensorElectricReferenceLossField:
    """Tensor-aware spatial-loss wrapper over the hybrid REFERENCE field."""

    def __init__(
        self,
        base: PreparedHybridReferenceLossField,
    ):
        self.base = base
        self.scene = base.scene
        self.frequency_hz = base.frequency_hz
        self.teacher = base.teacher
        self.result = base.result
        self.port_prediction = base.port_prediction
        self.package_transform = base.package_transform
        self.raw_dielectric_closure_error = (
            base.raw_dielectric_closure_error
        )
        self.normalized_dielectric_closure_error = (
            base.normalized_dielectric_closure_error
        )
        self.package_integrated_channels = (
            base.package_integrated_channels
        )
        self.background_integrated_channel = (
            base.background_integrated_channel
        )

    @property
    def normalization_closure_error(
        self,
    ) -> float:
        return self.base.normalization_closure_error

    @property
    def n_conductor_channels(
        self,
    ) -> int:
        return self.base.n_conductor_channels

    @property
    def dielectric_channel_index(
        self,
    ) -> int:
        return self.base.dielectric_channel_index

    @property
    def environment_channel_index(
        self,
    ) -> int:
        return self.base.environment_channel_index

    @property
    def background_channel_index(
        self,
    ) -> int | None:
        return self.base.background_channel_index

    @property
    def environment_transform(
        self,
    ) -> np.ndarray:
        return self.base.environment_transform

    def conductor_local_dissipation_matrix(
        self,
        *args,
        **kwargs,
    ):
        return self.base.conductor_local_dissipation_matrix(
            *args,
            **kwargs,
        )

    def local_dissipation_matrix(
        self,
        *args,
        **kwargs,
    ):
        return self.base.local_dissipation_matrix(
            *args,
            **kwargs,
        )

    def conductor_local_joule_density(
        self,
        *args,
        **kwargs,
    ):
        return self.base.conductor_local_joule_density(
            *args,
            **kwargs,
        )

    def electric_field_transfer(
        self,
        points,
    ) -> np.ndarray:
        return self.base.electric_field_transfer(
            points
        )

    def background_quadrature(
        self,
        **kwargs,
    ):
        return self.base.background_quadrature(
            **kwargs,
        )

    def raw_background_dissipation_matrices(
        self,
        points,
    ) -> np.ndarray:
        points = np.asarray(
            points,
            dtype=float,
        )
        scalar = points.ndim == 1
        points = np.atleast_2d(
            points
        )
        n_ports = int(
            self.port_prediction.impedance.shape[
                0
            ]
        )
        out = np.zeros(
            (
                len(
                    points
                ),
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )
        transmission = (
            self.result.tensor_electric_transmission
        )
        if transmission is None:
            return self.base.raw_background_dissipation_matrices(
                points[0]
                if scalar
                else points
            )
        if (
            self.scene.medium.loss_conductivity(
                self.frequency_hz
            )
            <= 0.0
        ):
            return out[0] if scalar else out
        exterior = np.asarray(
            self.base._background_domain_mask(
                points
            ),
            dtype=bool,
        )
        if np.any(
            exterior
        ):
            transfer = self.electric_field_transfer(
                points[
                    exterior
                ]
            )
            if transfer.ndim == 2:
                transfer = transfer[
                    None,
                    :,
                    :,
                ]
            conductivity = (
                transmission.conductivity_tensor_at(
                    points[
                        exterior
                    ]
                )
            )
            matrices = np.einsum(
                "qdi,qde,qej->qij",
                transfer.conj(),
                conductivity,
                transfer,
            )
            out[
                exterior
            ] = 0.5 * (
                matrices
                + matrices.conj().transpose(
                    0,
                    2,
                    1,
                )
            )
        return out[0] if scalar else out

    def background_dissipation_matrices(
        self,
        points,
    ) -> np.ndarray:
        raw = self.raw_background_dissipation_matrices(
            points
        )
        scalar = raw.ndim == 2
        if scalar:
            raw = raw[
                None,
                :,
                :,
            ]
        transform = self.environment_transform
        corrected = (
            transform[
                None,
                :,
                :,
            ]
            @ raw
            @ transform.conj().T[
                None,
                :,
                :,
            ]
        )
        corrected = 0.5 * (
            corrected
            + corrected.conj().transpose(
                0,
                2,
                1,
            )
        )
        return corrected[0] if scalar else corrected

    def background_joule_density(
        self,
        points,
        currents,
    ):
        matrices = self.background_dissipation_matrices(
            points
        )
        currents = np.asarray(
            currents,
            dtype=complex,
        )
        if currents.shape != (
            self.port_prediction.impedance.shape[
                0
            ],
        ):
            raise ValueError(
                "currents have wrong shape"
            )
        return 0.5 * np.real(
            np.einsum(
                "i,...ij,j->...",
                currents.conj(),
                matrices,
                currents,
            )
        )

    def raw_package_dissipation_matrices(
        self,
        *args,
        **kwargs,
    ):
        return self.base.raw_package_dissipation_matrices(
            *args,
            **kwargs,
        )

    def package_dissipation_matrices(
        self,
        *args,
        **kwargs,
    ):
        return self.base.package_dissipation_matrices(
            *args,
            **kwargs,
        )

    def package_local_dissipation_matrix(
        self,
        *args,
        **kwargs,
    ):
        return self.base.package_local_dissipation_matrix(
            *args,
            **kwargs,
        )

    def package_local_joule_density(
        self,
        *args,
        **kwargs,
    ):
        return self.base.package_local_joule_density(
            *args,
            **kwargs,
        )


def prepare_tensor_reference_loss_field(
    teacher,
    result,
    *,
    volume_axial_order: int = 8,
    volume_radial_order: int = 6,
    volume_azimuthal_order: int = 24,
    background_radial_order: int = 12,
    background_angular_order: int = 48,
    maximum_raw_closure_error: float = 0.25,
    normalized_closure_tolerance: float = 1e-6,
) -> PreparedTensorElectricReferenceLossField:
    if maximum_raw_closure_error <= 0.0:
        raise ValueError(
            "maximum_raw_closure_error must be positive"
        )
    if normalized_closure_tolerance <= 0.0:
        raise ValueError(
            "normalized_closure_tolerance must be positive"
        )
    scene = teacher.scene
    n_ports = int(
        result.impedance.shape[
            0
        ]
    )
    identity = np.eye(
        n_ports,
        dtype=complex,
    )
    temporary_base = PreparedHybridReferenceLossField(
        scene=scene,
        frequency_hz=float(
            teacher.frequency_hz
        ),
        teacher=teacher,
        result=result,
        port_prediction=result.prediction,
        package_transform=identity,
        raw_dielectric_closure_error=0.0,
        normalized_dielectric_closure_error=0.0,
        package_integrated_channels=np.zeros(
            (
                len(
                    scene.packages
                ),
                n_ports,
                n_ports,
            ),
            dtype=complex,
        ),
        background_integrated_channel=np.zeros(
            (
                n_ports,
                n_ports,
            ),
            dtype=complex,
        ),
    )
    temporary = PreparedTensorElectricReferenceLossField(
        temporary_base
    )

    raw_package_channels = []
    for package_index, package in enumerate(
        scene.packages
    ):
        quadrature = package.geometry.volume_quadrature(
            axial_order=(
                volume_axial_order
            ),
            radial_order=(
                volume_radial_order
            ),
            azimuthal_order=(
                volume_azimuthal_order
            ),
        )
        raw = temporary.raw_package_dissipation_matrices(
            package_index,
            quadrature.positions,
        )
        integrated = np.sum(
            quadrature.weights[
                :,
                None,
                None,
            ]
            * raw,
            axis=0,
        )
        raw_package_channels.append(
            0.5 * (
                integrated
                + integrated.conj().T
            )
        )
    if raw_package_channels:
        raw_package_channels = np.asarray(
            raw_package_channels,
            dtype=complex,
        )
    else:
        raw_package_channels = np.zeros(
            (
                0,
                n_ports,
                n_ports,
            ),
            dtype=complex,
        )

    raw_background_channel = np.zeros(
        (
            n_ports,
            n_ports,
        ),
        dtype=complex,
    )
    if (
        scene.medium.loss_conductivity(
            teacher.frequency_hz
        )
        > 0.0
    ):
        points, weights = temporary.background_quadrature(
            radial_order=(
                background_radial_order
            ),
            angular_order=(
                background_angular_order
            ),
        )
        raw_background = temporary.raw_background_dissipation_matrices(
            points
        )
        raw_background_channel = np.sum(
            weights[
                :,
                None,
                None,
            ]
            * raw_background,
            axis=0,
        )
        raw_background_channel = 0.5 * (
            raw_background_channel
            + raw_background_channel.conj().T
        )

    raw_total = (
        np.sum(
            raw_package_channels,
            axis=0,
        )
        + raw_background_channel
    )
    target = 0.5 * (
        result.dielectric_dissipation_matrix
        + result.dielectric_dissipation_matrix.conj().T
    )
    target_norm = max(
        float(
            np.linalg.norm(
                target
            )
        ),
        1e-30,
    )
    raw_error = float(
        np.linalg.norm(
            raw_total
            - target
        )
        / target_norm
    )

    if np.linalg.norm(target) <= 1e-18:
        if np.linalg.norm(raw_total) > 1e-12:
            raise RuntimeError(
                "tensor spatial field predicts environment loss for a "
                "lossless port-level channel"
            )
        transform = identity
        corrected_package_channels = np.zeros_like(
            raw_package_channels
        )
        corrected_background_channel = np.zeros_like(
            raw_background_channel
        )
        normalized_error = 0.0
    else:
        if raw_error > maximum_raw_closure_error:
            raise RuntimeError(
                "raw tensor electric-environment field integration does not "
                "close the port-level loss channel: "
                f"relative error={raw_error:.3e}"
            )
        scale = max(
            float(
                np.trace(
                    target
                ).real
                / max(
                    n_ports,
                    1,
                )
            ),
            target_norm / max(n_ports, 1),
            1e-30,
        )
        regularization = 1e-12 * scale
        transform = (
            _hermitian_psd_sqrt(
                target
                + regularization
                * identity,
                inverse=False,
            )
            @ _hermitian_psd_sqrt(
                raw_total
                + regularization
                * identity,
                inverse=True,
            )
        )
        if len(raw_package_channels):
            corrected_package_channels = np.asarray(
                [
                    0.5 * (
                        transform
                        @ channel
                        @ transform.conj().T
                        + (
                            transform
                            @ channel
                            @ transform.conj().T
                        ).conj().T
                    )
                    for channel in raw_package_channels
                ],
                dtype=complex,
            )
        else:
            corrected_package_channels = np.zeros_like(
                raw_package_channels
            )
        corrected_background_channel = 0.5 * (
            transform
            @ raw_background_channel
            @ transform.conj().T
            + (
                transform
                @ raw_background_channel
                @ transform.conj().T
            ).conj().T
        )
        corrected_total = (
            np.sum(
                corrected_package_channels,
                axis=0,
            )
            + corrected_background_channel
        )
        normalized_error = float(
            np.linalg.norm(
                corrected_total
                - target
            )
            / target_norm
        )
        if normalized_error > normalized_closure_tolerance:
            raise RuntimeError(
                "normalized tensor electric-environment spatial field failed "
                "power closure: "
                f"relative error={normalized_error:.3e}"
            )

    final_base = PreparedHybridReferenceLossField(
        scene=scene,
        frequency_hz=float(
            teacher.frequency_hz
        ),
        teacher=teacher,
        result=result,
        port_prediction=result.prediction,
        package_transform=transform,
        raw_dielectric_closure_error=raw_error,
        normalized_dielectric_closure_error=normalized_error,
        package_integrated_channels=corrected_package_channels,
        background_integrated_channel=corrected_background_channel,
    )
    return PreparedTensorElectricReferenceLossField(
        final_base
    )


class TensorElectricReferenceArtifact(
    DielectricCoupledReferenceArtifact
):
    """REFERENCE artifact adding tensor-aware continuous loss reconstruction."""

    @staticmethod
    def supports_scene(
        scene: Scene,
    ) -> bool:
        return bool(
            isinstance(
                scene.medium,
                TensorElectricMaterial,
            )
            or any(
                isinstance(
                    package.material,
                    TensorElectricMaterial,
                )
                for package in scene.packages
            )
        )

    def prepare_spatial(
        self,
        scene: Scene,
        frequency_hz: float,
        *,
        volume_axial_order: int = 8,
        volume_radial_order: int = 6,
        volume_azimuthal_order: int = 24,
        background_radial_order: int = 12,
        background_angular_order: int = 48,
        maximum_raw_closure_error: float = 0.25,
        normalized_closure_tolerance: float = 1e-6,
    ):
        if not self.supports_scene(
            scene
        ):
            return super().prepare_spatial(
                scene,
                frequency_hz,
                volume_axial_order=(
                    volume_axial_order
                ),
                volume_radial_order=(
                    volume_radial_order
                ),
                volume_azimuthal_order=(
                    volume_azimuthal_order
                ),
                background_radial_order=(
                    background_radial_order
                ),
                background_angular_order=(
                    background_angular_order
                ),
                maximum_raw_closure_error=(
                    maximum_raw_closure_error
                ),
                normalized_closure_tolerance=(
                    normalized_closure_tolerance
                ),
            )
        teacher = DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            self.config,
            surface_vertical_order=(
                self.surface_vertical_order
            ),
            surface_azimuthal_order=(
                self.surface_azimuthal_order
            ),
            magnetic_volume_axial_order=(
                volume_axial_order
            ),
            magnetic_volume_radial_order=(
                volume_radial_order
            ),
            magnetic_volume_azimuthal_order=(
                volume_azimuthal_order
            ),
            maximum_raw_magnetic_reciprocity_defect=(
                self.maximum_raw_magnetic_reciprocity_defect
            ),
        )
        result = teacher.solve()
        return prepare_tensor_reference_loss_field(
            teacher,
            result,
            volume_axial_order=(
                volume_axial_order
            ),
            volume_radial_order=(
                volume_radial_order
            ),
            volume_azimuthal_order=(
                volume_azimuthal_order
            ),
            background_radial_order=(
                background_radial_order
            ),
            background_angular_order=(
                background_angular_order
            ),
            maximum_raw_closure_error=(
                maximum_raw_closure_error
            ),
            normalized_closure_tolerance=(
                normalized_closure_tolerance
            ),
        )
