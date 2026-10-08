from __future__ import annotations

from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
    DielectricCoupledReferenceArtifact,
)
from .scene import Scene, TensorElectricMaterial
from .tensor_spatial_reference import (
    PreparedTensorEnergyReferenceLossField,
    prepare_tensor_spatial_reference_adaptive,
)


class PreparedTensorElectricReferenceLossField(
    PreparedTensorEnergyReferenceLossField
):
    """Backward-compatible name for the canonical tensor energy field."""


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
    """Prepare tensor spatial truth from the canonical continuous-energy model.

    Tensor port dissipation and pointwise spatial dissipation must be derived
    from the same E^H sigma E energy operator.  This function intentionally
    delegates to the training/reference energy helper rather than maintaining a
    second Im(V_eff)-to-spatial congruence calibration path.
    """
    if normalized_closure_tolerance <= 0.0:
        raise ValueError("normalized_closure_tolerance must be positive")

    calibration = prepare_tensor_spatial_reference_adaptive(
        teacher,
        result,
        volume_axial_order=int(volume_axial_order),
        volume_radial_order=int(volume_radial_order),
        volume_azimuthal_order=int(volume_azimuthal_order),
        background_radial_order=int(background_radial_order),
        background_angular_order=int(background_angular_order),
        maximum_raw_closure_error=float(maximum_raw_closure_error),
        maximum_quadrature_refinements=0,
    )
    prepared = PreparedTensorElectricReferenceLossField(
        calibration.prepared.base
    )
    closure = float(prepared.normalization_closure_error)
    if closure > float(normalized_closure_tolerance):
        raise RuntimeError(
            "tensor energy REFERENCE field failed normalized power closure: "
            f"relative error={closure:.3e}"
        )
    return prepared


class TensorElectricReferenceArtifact(
    DielectricCoupledReferenceArtifact
):
    """REFERENCE artifact using one energy-consistent tensor electric truth."""

    def __init__(
        self,
        *,
        config=None,
        surface_vertical_order: int = 16,
        surface_azimuthal_order: int = 32,
        magnetic_volume_axial_order: int = 8,
        magnetic_volume_radial_order: int = 6,
        magnetic_volume_azimuthal_order: int = 24,
        maximum_raw_magnetic_reciprocity_defect: float = 0.08,
        energy_volume_axial_order: int = 8,
        energy_volume_radial_order: int = 6,
        energy_volume_azimuthal_order: int = 24,
        energy_background_radial_order: int = 12,
        energy_background_angular_order: int = 48,
        maximum_raw_energy_closure_error: float = 0.25,
        normalized_energy_closure_tolerance: float = 1e-6,
    ):
        super().__init__(
            config=config,
            surface_vertical_order=surface_vertical_order,
            surface_azimuthal_order=surface_azimuthal_order,
            magnetic_volume_axial_order=magnetic_volume_axial_order,
            magnetic_volume_radial_order=magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order=magnetic_volume_azimuthal_order,
            maximum_raw_magnetic_reciprocity_defect=(
                maximum_raw_magnetic_reciprocity_defect
            ),
        )
        self.energy_volume_axial_order = int(energy_volume_axial_order)
        self.energy_volume_radial_order = int(energy_volume_radial_order)
        self.energy_volume_azimuthal_order = int(energy_volume_azimuthal_order)
        self.energy_background_radial_order = int(
            energy_background_radial_order
        )
        self.energy_background_angular_order = int(
            energy_background_angular_order
        )
        self.maximum_raw_energy_closure_error = float(
            maximum_raw_energy_closure_error
        )
        self.normalized_energy_closure_tolerance = float(
            normalized_energy_closure_tolerance
        )
        if (
            self.energy_volume_axial_order < 2
            or self.energy_volume_radial_order < 2
            or self.energy_volume_azimuthal_order < 8
            or self.energy_background_radial_order < 3
            or self.energy_background_angular_order < 8
        ):
            raise ValueError("invalid tensor REFERENCE energy quadrature")
        if (
            self.maximum_raw_energy_closure_error <= 0.0
            or self.normalized_energy_closure_tolerance <= 0.0
        ):
            raise ValueError("invalid tensor REFERENCE energy tolerances")

    @staticmethod
    def supports_scene(
        scene: Scene,
    ) -> bool:
        return bool(
            isinstance(scene.medium, TensorElectricMaterial)
            or any(
                isinstance(package.material, TensorElectricMaterial)
                for package in scene.packages
            )
        )

    def _raw_teacher(
        self,
        scene: Scene,
        frequency_hz: float,
    ) -> DielectricCoupledMixedTeacher:
        return DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            self.config,
            surface_vertical_order=self.surface_vertical_order,
            surface_azimuthal_order=self.surface_azimuthal_order,
            magnetic_volume_axial_order=self.magnetic_volume_axial_order,
            magnetic_volume_radial_order=self.magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order=self.magnetic_volume_azimuthal_order,
            maximum_raw_magnetic_reciprocity_defect=(
                self.maximum_raw_magnetic_reciprocity_defect
            ),
        )

    def solve(
        self,
        scene: Scene,
        frequency_hz: float,
    ):
        if not self.supports_scene(scene):
            return super().solve(scene, frequency_hz)
        teacher = self._raw_teacher(scene, frequency_hz)
        raw_result = teacher.solve()
        calibration = prepare_tensor_spatial_reference_adaptive(
            teacher,
            raw_result,
            volume_axial_order=self.energy_volume_axial_order,
            volume_radial_order=self.energy_volume_radial_order,
            volume_azimuthal_order=self.energy_volume_azimuthal_order,
            background_radial_order=self.energy_background_radial_order,
            background_angular_order=self.energy_background_angular_order,
            maximum_raw_closure_error=(
                self.maximum_raw_energy_closure_error
            ),
            maximum_quadrature_refinements=0,
        )
        return calibration.prepared.result

    def prepare_spatial(
        self,
        scene: Scene,
        frequency_hz: float,
        *,
        volume_axial_order: int | None = None,
        volume_radial_order: int | None = None,
        volume_azimuthal_order: int | None = None,
        background_radial_order: int | None = None,
        background_angular_order: int | None = None,
        maximum_raw_closure_error: float | None = None,
        normalized_closure_tolerance: float | None = None,
    ):
        tensor_scene = self.supports_scene(scene)
        if tensor_scene:
            volume_axial_order = (
                self.energy_volume_axial_order
                if volume_axial_order is None
                else int(volume_axial_order)
            )
            volume_radial_order = (
                self.energy_volume_radial_order
                if volume_radial_order is None
                else int(volume_radial_order)
            )
            volume_azimuthal_order = (
                self.energy_volume_azimuthal_order
                if volume_azimuthal_order is None
                else int(volume_azimuthal_order)
            )
            background_radial_order = (
                self.energy_background_radial_order
                if background_radial_order is None
                else int(background_radial_order)
            )
            background_angular_order = (
                self.energy_background_angular_order
                if background_angular_order is None
                else int(background_angular_order)
            )
            maximum_raw_closure_error = (
                self.maximum_raw_energy_closure_error
                if maximum_raw_closure_error is None
                else float(maximum_raw_closure_error)
            )
            normalized_closure_tolerance = (
                self.normalized_energy_closure_tolerance
                if normalized_closure_tolerance is None
                else float(normalized_closure_tolerance)
            )
        else:
            volume_axial_order = 8 if volume_axial_order is None else int(volume_axial_order)
            volume_radial_order = 6 if volume_radial_order is None else int(volume_radial_order)
            volume_azimuthal_order = 24 if volume_azimuthal_order is None else int(volume_azimuthal_order)
            background_radial_order = 12 if background_radial_order is None else int(background_radial_order)
            background_angular_order = 48 if background_angular_order is None else int(background_angular_order)
            maximum_raw_closure_error = (
                0.25
                if maximum_raw_closure_error is None
                else float(maximum_raw_closure_error)
            )
            normalized_closure_tolerance = (
                1e-6
                if normalized_closure_tolerance is None
                else float(normalized_closure_tolerance)
            )
            return super().prepare_spatial(
                scene,
                frequency_hz,
                volume_axial_order=volume_axial_order,
                volume_radial_order=volume_radial_order,
                volume_azimuthal_order=volume_azimuthal_order,
                background_radial_order=background_radial_order,
                background_angular_order=background_angular_order,
                maximum_raw_closure_error=maximum_raw_closure_error,
                normalized_closure_tolerance=normalized_closure_tolerance,
            )

        teacher = self._raw_teacher(scene, frequency_hz)
        result = teacher.solve()
        return prepare_tensor_reference_loss_field(
            teacher,
            result,
            volume_axial_order=volume_axial_order,
            volume_radial_order=volume_radial_order,
            volume_azimuthal_order=volume_azimuthal_order,
            background_radial_order=background_radial_order,
            background_angular_order=background_angular_order,
            maximum_raw_closure_error=maximum_raw_closure_error,
            normalized_closure_tolerance=normalized_closure_tolerance,
        )
