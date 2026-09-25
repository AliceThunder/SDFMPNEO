from __future__ import annotations

import numpy as np

from .certification import PortCertificate
from .certified import CertifiedPortResult
from .em import MQSConfig
from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
)
from .scene import Scene


def certify_dielectric_ports(
    scene: Scene,
    frequency_hz: float,
    artifact,
    *,
    config: MQSConfig | None = None,
    convergence_report=None,
    surface_vertical_order: int = 16,
    surface_azimuthal_order: int = 32,
    algebraic_tolerance: float = 1e-9,
    surface_tolerance: float = 1e-9,
    reciprocity_tolerance: float = 1e-8,
    power_tolerance: float = 1e-6,
    passivity_tolerance: float = 1e-10,
    fast_domain_correction_limit: float = 0.20,
) -> CertifiedPortResult:
    """REFERENCE-backed certification for dielectric package scenes.

    The result explicitly reports used_reference_fallback=True because the
    coupled dielectric SIE is currently solved as the full dense physical
    system. A later matrix-free SIE correction can replace this backend without
    changing the public certification contract.
    """
    if not scene.packages:
        raise ValueError(
            "dielectric certification requires at least one package"
        )
    if not hasattr(
        artifact,
        "predict_structured",
    ):
        raise TypeError(
            "artifact must expose predict_structured"
        )
    if (
        algebraic_tolerance <= 0.0
        or surface_tolerance <= 0.0
        or reciprocity_tolerance < 0.0
        or power_tolerance < 0.0
        or passivity_tolerance < 0.0
        or fast_domain_correction_limit < 0.0
    ):
        raise ValueError(
            "invalid dielectric certification tolerances"
        )

    fast_prediction = (
        artifact.predict_structured(
            scene,
            frequency_hz,
        )
    )
    fast_impedance = np.asarray(
        fast_prediction.impedance,
        dtype=complex,
    )

    teacher = (
        DielectricCoupledMixedTeacher(
            scene,
            frequency_hz,
            config
            or MQSConfig(),
            surface_vertical_order=(
                surface_vertical_order
            ),
            surface_azimuthal_order=(
                surface_azimuthal_order
            ),
        )
    )
    result = teacher.solve()
    impedance = np.asarray(
        result.impedance,
        dtype=complex,
    )
    if (
        fast_impedance.shape
        != impedance.shape
    ):
        raise ValueError(
            "FAST artifact returned the wrong port-matrix shape"
        )

    scale = max(
        float(
            np.linalg.norm(
                impedance
            )
        ),
        1e-30,
    )
    reciprocity = float(
        np.linalg.norm(
            impedance
            - impedance.T
        )
        / scale
    )
    dissipation = 0.5 * (
        impedance
        + impedance.conj().T
    )
    minimum_dissipation = float(
        np.min(
            np.linalg.eigvalsh(
                dissipation
            )
        )
    )
    power_closure = float(
        result.prediction.power_closure_error()
    )
    physical_residual = float(
        max(
            result.mixed_result.normalized_residual,
            result.surface_residual,
        )
    )
    algebraic_certified = bool(
        result.mixed_result.normalized_residual
        <= algebraic_tolerance
        and result.surface_residual
        <= surface_tolerance
        and reciprocity
        <= reciprocity_tolerance
        and minimum_dissipation
        >= -passivity_tolerance
        and power_closure
        <= power_tolerance
    )
    port_certificate = PortCertificate(
        reciprocity_defect=(
            reciprocity
        ),
        minimum_dissipation_eigenvalue=(
            minimum_dissipation
        ),
        power_closure_error=(
            power_closure
        ),
        algebraic_residual=(
            physical_residual
        ),
        certified=(
            algebraic_certified
        ),
    )

    correction = float(
        np.linalg.norm(
            impedance
            - fast_impedance
        )
        / scale
    )
    fast_domain_valid = bool(
        correction
        <= fast_domain_correction_limit
    )
    discretization_certified = bool(
        convergence_report
        is not None
        and getattr(
            convergence_report,
            "converged",
            False,
        )
    )

    if not algebraic_certified:
        status = "UNCERTIFIED"
    elif not discretization_certified:
        status = (
            "DISCRETE_CERTIFIED"
        )
    elif fast_domain_valid:
        status = "CERTIFIED"
    else:
        status = (
            "CORRECTED_OUT_OF_FAST_DOMAIN"
        )

    return CertifiedPortResult(
        status=status,
        impedance=impedance,
        result=result,
        port_certificate=(
            port_certificate
        ),
        initial_residual=(
            physical_residual
        ),
        final_residual=(
            physical_residual
        ),
        correction_iterations=tuple(
            0
            for _ in range(
                impedance.shape[
                    0
                ]
            )
        ),
        algebraic_certified=(
            algebraic_certified
        ),
        discretization_certified=(
            discretization_certified
        ),
        used_reference_fallback=True,
        operator_backend=(
            "dense_dielectric_reference"
        ),
        relative_observable_correction=(
            correction
        ),
        fast_domain_valid=(
            fast_domain_valid
        ),
    )
