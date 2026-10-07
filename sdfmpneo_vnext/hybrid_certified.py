from __future__ import annotations

import numpy as np

from .certification import PortCertificate
from .certified import CertifiedPortResult
from .em import MQSConfig
from .hybrid_dielectric import DielectricCoupledMixedTeacher
from .hybrid_convergence import hybrid_reference_convergence
from .scene import Scene
from .tensor_spatial_reference import prepare_tensor_spatial_reference_adaptive


def certify_dielectric_ports(
    scene: Scene,
    frequency_hz: float,
    artifact,
    *,
    config: MQSConfig | None = None,
    convergence_report=None,
    graded_convergence_report=None,
    auto_convergence: bool = False,
    convergence_tolerance: float = 2e-3,
    convergence_surface_residual_tolerance: float | None = None,
    convergence_magnetic_surface_residual_tolerance: float | None = None,
    surface_vertical_order: int = 16,
    surface_azimuthal_order: int = 32,
    magnetic_volume_axial_order: int = 8,
    magnetic_volume_radial_order: int = 6,
    magnetic_volume_azimuthal_order: int = 24,
    energy_volume_axial_order: int = 8,
    energy_volume_radial_order: int = 6,
    energy_volume_azimuthal_order: int = 24,
    energy_background_radial_order: int = 12,
    energy_background_angular_order: int = 48,
    maximum_raw_energy_closure_error: float = 0.25,
    algebraic_tolerance: float = 1e-9,
    surface_tolerance: float = 1e-9,
    reciprocity_tolerance: float = 1e-8,
    magnetic_reciprocity_tolerance: float = 0.08,
    power_tolerance: float = 1e-6,
    passivity_tolerance: float = 1e-10,
    fast_domain_correction_limit: float = 0.20,
) -> CertifiedPortResult:
    """REFERENCE-backed certification for heterogeneous/tensor electric scenes.

    Tensor-electric scenes use the same continuous E^H sigma E energy truth as
    tensor FAST training, tensor REFERENCE queries, and convergence checks.
    Scalar heterogeneous scenes retain the original dense interface reference.
    """
    if not hasattr(artifact, "predict_structured"):
        raise TypeError("artifact must expose predict_structured")
    if convergence_tolerance <= 0.0:
        raise ValueError("convergence_tolerance must be positive")
    if (
        convergence_surface_residual_tolerance is not None
        and convergence_surface_residual_tolerance <= 0.0
    ):
        raise ValueError(
            "convergence_surface_residual_tolerance must be positive"
        )
    if (
        convergence_magnetic_surface_residual_tolerance is not None
        and convergence_magnetic_surface_residual_tolerance <= 0.0
    ):
        raise ValueError(
            "convergence_magnetic_surface_residual_tolerance must be positive"
        )
    if (
        algebraic_tolerance <= 0.0
        or surface_tolerance <= 0.0
        or reciprocity_tolerance < 0.0
        or magnetic_reciprocity_tolerance < 0.0
        or power_tolerance < 0.0
        or passivity_tolerance < 0.0
        or fast_domain_correction_limit < 0.0
        or maximum_raw_energy_closure_error <= 0.0
    ):
        raise ValueError("invalid dielectric certification tolerances")
    if (
        energy_volume_axial_order < 2
        or energy_volume_radial_order < 2
        or energy_volume_azimuthal_order < 8
        or energy_background_radial_order < 3
        or energy_background_angular_order < 8
    ):
        raise ValueError("invalid tensor certification energy quadrature")

    resolved_config = config or MQSConfig()
    if convergence_report is None and auto_convergence:
        convergence_report = hybrid_reference_convergence(
            scene,
            frequency_hz,
            resolved_config,
            surface_vertical_order=surface_vertical_order,
            surface_azimuthal_order=surface_azimuthal_order,
            magnetic_volume_axial_order=magnetic_volume_axial_order,
            magnetic_volume_radial_order=magnetic_volume_radial_order,
            magnetic_volume_azimuthal_order=magnetic_volume_azimuthal_order,
            energy_volume_axial_order=energy_volume_axial_order,
            energy_volume_radial_order=energy_volume_radial_order,
            energy_volume_azimuthal_order=energy_volume_azimuthal_order,
            energy_background_radial_order=energy_background_radial_order,
            energy_background_angular_order=energy_background_angular_order,
            tolerance=convergence_tolerance,
            surface_residual_tolerance=(
                surface_tolerance
                if convergence_surface_residual_tolerance is None
                else convergence_surface_residual_tolerance
            ),
            magnetic_surface_residual_tolerance=(
                surface_tolerance
                if convergence_magnetic_surface_residual_tolerance is None
                else convergence_magnetic_surface_residual_tolerance
            ),
        )

    fast_prediction = None
    fast_impedance = None
    fast_domain_reason = None
    try:
        fast_prediction = artifact.predict_structured(scene, frequency_hz)
        fast_impedance = np.asarray(fast_prediction.impedance, dtype=complex)
    except (ValueError, NotImplementedError) as exc:
        fast_domain_reason = str(exc)

    teacher = DielectricCoupledMixedTeacher(
        scene,
        frequency_hz,
        resolved_config,
        surface_vertical_order=surface_vertical_order,
        surface_azimuthal_order=surface_azimuthal_order,
        magnetic_volume_axial_order=magnetic_volume_axial_order,
        magnetic_volume_radial_order=magnetic_volume_radial_order,
        magnetic_volume_azimuthal_order=magnetic_volume_azimuthal_order,
        maximum_raw_magnetic_reciprocity_defect=(
            magnetic_reciprocity_tolerance
        ),
    )
    result = teacher.solve()
    if result.tensor_electric_transmission is not None:
        calibration = prepare_tensor_spatial_reference_adaptive(
            teacher,
            result,
            volume_axial_order=int(energy_volume_axial_order),
            volume_radial_order=int(energy_volume_radial_order),
            volume_azimuthal_order=int(energy_volume_azimuthal_order),
            background_radial_order=int(energy_background_radial_order),
            background_angular_order=int(energy_background_angular_order),
            maximum_raw_closure_error=float(
                maximum_raw_energy_closure_error
            ),
            maximum_quadrature_refinements=0,
        )
        result = calibration.prepared.result

    impedance = np.asarray(result.impedance, dtype=complex)
    if fast_impedance is not None and fast_impedance.shape != impedance.shape:
        raise ValueError("FAST artifact returned the wrong port-matrix shape")

    scale = max(float(np.linalg.norm(impedance)), 1e-30)
    reciprocity = float(np.linalg.norm(impedance - impedance.T) / scale)
    dissipation = 0.5 * (impedance + impedance.conj().T)
    minimum_dissipation = float(np.min(np.linalg.eigvalsh(dissipation)))
    power_closure = float(result.prediction.power_closure_error())
    physical_residual = float(
        max(
            result.mixed_result.normalized_residual,
            result.surface_residual,
            result.magnetic_surface_residual,
        )
    )
    algebraic_certified = bool(
        result.mixed_result.normalized_residual <= algebraic_tolerance
        and result.surface_residual <= surface_tolerance
        and result.magnetic_surface_residual <= surface_tolerance
        and result.raw_magnetic_reciprocity_defect
        <= magnetic_reciprocity_tolerance
        and reciprocity <= reciprocity_tolerance
        and minimum_dissipation >= -passivity_tolerance
        and power_closure <= power_tolerance
    )
    port_certificate = PortCertificate(
        reciprocity_defect=reciprocity,
        minimum_dissipation_eigenvalue=minimum_dissipation,
        power_closure_error=power_closure,
        algebraic_residual=physical_residual,
        certified=algebraic_certified,
    )

    if fast_impedance is None:
        correction = float("inf")
        fast_domain_valid = False
    else:
        correction = float(np.linalg.norm(impedance - fast_impedance) / scale)
        fast_domain_valid = bool(correction <= fast_domain_correction_limit)
    reference_discretization_certified = bool(
        convergence_report is not None
        and getattr(convergence_report, "converged", False)
    )
    graded_discretization_certified = bool(
        graded_convergence_report is None
        or getattr(graded_convergence_report, "converged", False)
    )
    discretization_certified = bool(
        reference_discretization_certified
        and graded_discretization_certified
    )

    if not algebraic_certified:
        status = "UNCERTIFIED"
    elif not discretization_certified:
        status = "DISCRETE_CERTIFIED"
    elif fast_domain_valid:
        status = "CERTIFIED"
    else:
        status = "CORRECTED_OUT_OF_FAST_DOMAIN"

    return CertifiedPortResult(
        status=status,
        impedance=impedance,
        result=result,
        port_certificate=port_certificate,
        initial_residual=physical_residual,
        final_residual=physical_residual,
        correction_iterations=tuple(0 for _ in range(impedance.shape[0])),
        algebraic_certified=algebraic_certified,
        discretization_certified=discretization_certified,
        used_reference_fallback=True,
        operator_backend="dense_electric_magnetic_interface_reference",
        relative_observable_correction=correction,
        fast_domain_valid=fast_domain_valid,
        fast_domain_reason=fast_domain_reason,
    )
