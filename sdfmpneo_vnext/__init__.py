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
from .electrothermal import (
    CoilThermalProperties,
    ElectroThermalStep,
    CurrentControlledEnvelope,
    VoltageControlledEnvelope,
    build_lumped_coil_thermal_model,
)
from .prediction import StructuredPortPrediction
from .fast import (
    FastCurrentControlledEnvelope,
    FastVoltageControlledEnvelope,
)
from .analytic_baseline import (
    AnalyticBaselineArtifact,
    AnalyticBaselineResult,
    analytic_port_baseline,
    pair_mutual_inductance,
    regularized_self_inductance,
)
from .features import EncodedScene, encode_scene_invariant
from .training_data import TeacherSample
from .sampling import MVPSceneSamplerConfig, sample_two_coil_mvp_scene
from .dataset import (
    DATASET_SCHEMA,
    SPLITS,
    DatasetRecord,
    ImmutableTeacherDataset,
    deterministic_split,
)
from .serialization import scene_from_dict, scene_to_dict
from .evaluation import SurrogateAudit, audit_surrogate
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
    "CoilThermalProperties",
    "ElectroThermalStep",
    "CurrentControlledEnvelope",
    "VoltageControlledEnvelope",
    "build_lumped_coil_thermal_model",
    "StructuredPortPrediction",
    "FastCurrentControlledEnvelope",
    "FastVoltageControlledEnvelope",
    "AnalyticBaselineArtifact",
    "AnalyticBaselineResult",
    "analytic_port_baseline",
    "pair_mutual_inductance",
    "regularized_self_inductance",
    "EncodedScene",
    "encode_scene_invariant",
    "TeacherSample",
    "MVPSceneSamplerConfig",
    "sample_two_coil_mvp_scene",
    "DATASET_SCHEMA",
    "SPLITS",
    "DatasetRecord",
    "ImmutableTeacherDataset",
    "deterministic_split",
    "scene_from_dict",
    "scene_to_dict",
    "SurrogateAudit",
    "audit_surrogate",
    "MU0",
    "coaxial_circular_mutual_inductance",
    "dc_resistance",
    "lumped_thermal_step",
    "neumann_mutual_inductance",
    "superellipse_area",
    "thermal_impulse_temperature",
    "thermal_step_temperature",
]
