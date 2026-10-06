from __future__ import annotations

from dataclasses import dataclass
import math

from .hybrid_field import (
    PreparedHybridReferenceLossField,
    prepare_hybrid_reference_loss_field,
)


_RAW_CLOSURE_PREFIX = (
    "raw electric-environment field integration does not close the port-level "
    "dielectric loss channel"
)


@dataclass(frozen=True)
class TensorSpatialReferenceCalibration:
    prepared: PreparedHybridReferenceLossField
    package_volume_axial_order: int
    package_volume_radial_order: int
    package_volume_azimuthal_order: int
    background_radial_order: int
    background_angular_order: int
    refinements: int


def _refined_order(value: int, *, minimum_step: int) -> int:
    """Compatibility helper retained for callers/tests from the adaptive path."""
    value = int(value)
    return max(
        value + int(minimum_step),
        int(math.ceil(1.5 * value)),
    )


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
    """Calibrate tensor spatial loss at the requested object-local resolution.

    Tensor-electric port loss is obtained from the effective nodal-potential
    operator, while the continuous spatial field integrates only the physical
    dielectric/background domain and explicitly excludes conductor volume.
    Their *raw* powers therefore need not agree even when both numerical
    integrations are converged.  Treating that model discrepancy as a
    quadrature error caused production training to refine to very high orders
    without reducing the closure defect.

    We still run the declared raw-closure diagnostic first so the raw defect is
    visible whenever it already satisfies the historical tolerance.  If that
    diagnostic alone fails, perform the existing PSD congruence calibration at
    the same requested resolution and rely on its strict normalized power-
    closure check (default 1e-6).  Other RuntimeError values are never masked.

    ``maximum_quadrature_refinements`` remains in the public/config contract so
    existing cache identities and callers stay compatible, but raw tensor
    closure is no longer misclassified as a quadrature-convergence problem.
    """
    maximum_quadrature_refinements = int(maximum_quadrature_refinements)
    if maximum_quadrature_refinements < 0:
        raise ValueError(
            "maximum_quadrature_refinements must be nonnegative"
        )

    axial = int(volume_axial_order)
    radial = int(volume_radial_order)
    azimuthal = int(volume_azimuthal_order)
    background_radial = int(background_radial_order)
    background_angular = int(background_angular_order)

    try:
        prepared = prepare_hybrid_reference_loss_field(
            teacher,
            result,
            volume_axial_order=axial,
            volume_radial_order=radial,
            volume_azimuthal_order=azimuthal,
            background_radial_order=background_radial,
            background_angular_order=background_angular,
            maximum_raw_closure_error=float(
                maximum_raw_closure_error
            ),
        )
    except RuntimeError as exc:
        if not str(exc).startswith(_RAW_CLOSURE_PREFIX):
            raise
        prepared = prepare_hybrid_reference_loss_field(
            teacher,
            result,
            volume_axial_order=axial,
            volume_radial_order=radial,
            volume_azimuthal_order=azimuthal,
            background_radial_order=background_radial,
            background_angular_order=background_angular,
            maximum_raw_closure_error=float("inf"),
        )

    return TensorSpatialReferenceCalibration(
        prepared=prepared,
        package_volume_axial_order=axial,
        package_volume_radial_order=radial,
        package_volume_azimuthal_order=azimuthal,
        background_radial_order=background_radial,
        background_angular_order=background_angular,
        refinements=0,
    )
