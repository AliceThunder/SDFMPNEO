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
    """Calibrate tensor spatial loss with object-local adaptive quadrature.

    The requested orders are the first calibration level.  A coarse package
    product rule can miss the narrow, high-field neighborhood around finite
    conductors, even though pointwise electric-field reconstruction is valid.
    When that happens, refine only the REFERENCE closure integration.  Training
    sample density remains independently controlled by the caller.
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

    last_closure_error = None
    for refinement in range(maximum_quadrature_refinements + 1):
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
            return TensorSpatialReferenceCalibration(
                prepared=prepared,
                package_volume_axial_order=axial,
                package_volume_radial_order=radial,
                package_volume_azimuthal_order=azimuthal,
                background_radial_order=background_radial,
                background_angular_order=background_angular,
                refinements=refinement,
            )
        except RuntimeError as exc:
            message = str(exc)
            if not message.startswith(_RAW_CLOSURE_PREFIX):
                raise
            last_closure_error = exc
            if refinement >= maximum_quadrature_refinements:
                break

            axial = _refined_order(axial, minimum_step=2)
            radial = _refined_order(radial, minimum_step=2)
            azimuthal = _refined_order(azimuthal, minimum_step=8)
            background_radial = _refined_order(
                background_radial,
                minimum_step=2,
            )
            background_angular = _refined_order(
                background_angular,
                minimum_step=8,
            )

    raise RuntimeError(
        "tensor spatial REFERENCE closure remained outside tolerance after "
        f"{maximum_quadrature_refinements} adaptive quadrature refinements; "
        f"final package orders=({axial},{radial},{azimuthal}), "
        f"background orders=({background_radial},{background_angular})"
    ) from last_closure_error
