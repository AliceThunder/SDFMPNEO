"""SDF-MPNEO vNext mesh-free-first MVP.

This package is intentionally independent from the legacy fixed-grid runtime.
"""

from .geometry import (
    RigidPose,
    SuperellipseSpiral,
    PolylineConductor,
    bishop_segment_frames,
    haar_rotation,
)
from .scene import (
    ConductorMaterial,
    HomogeneousMedium,
    CoilObject,
    Scene,
)
from .basis import (
    SectionBasis,
    SectionQuadrature,
    polynomial_section_basis,
    superellipse_section_quadrature,
)
from .em import (
    DenseMQSTeacher,
    MQSConfig,
    MQSResult,
)
from .mixed import (
    DenseMixedConductorTeacher,
    MixedResult,
)
from .thermal import StableThermalModel
from .loss import ConductorLossField
from .certification import PortCertificate, certify_port_result
from .convergence import ConvergenceStep, ConvergenceReport, impedance_convergence
from .benchmarks import (
    MU0,
    coaxial_circular_mutual_inductance,
    dc_resistance,
    lumped_thermal_step,
    neumann_mutual_inductance,
    superellipse_area,
    thermal_impulse_temperature,
    thermal_step_temperature,
)

__all__ = [
    "RigidPose",
    "SuperellipseSpiral",
    "PolylineConductor",
    "bishop_segment_frames",
    "haar_rotation",
    "ConductorMaterial",
    "HomogeneousMedium",
    "CoilObject",
    "Scene",
    "SectionBasis",
    "SectionQuadrature",
    "polynomial_section_basis",
    "superellipse_section_quadrature",
    "DenseMQSTeacher",
    "MQSConfig",
    "MQSResult",
    "DenseMixedConductorTeacher",
    "MixedResult",
    "StableThermalModel",
    "ConductorLossField",
    "PortCertificate",
    "certify_port_result",
    "ConvergenceStep",
    "ConvergenceReport",
    "impedance_convergence",
    "MU0",
    "coaxial_circular_mutual_inductance",
    "dc_resistance",
    "lumped_thermal_step",
    "neumann_mutual_inductance",
    "superellipse_area",
    "thermal_impulse_temperature",
    "thermal_step_temperature",
]
