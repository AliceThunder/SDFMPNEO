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
from .matrix_free import (
    MatrixFreeMetadata,
    MatrixFreeMQSOperator,
    MatrixFreeMixedMetadata,
    MatrixFreeMixedOperator,
)
from .thermal import StableThermalModel
from .thermal_field import (
    HomogeneousThermalMedium,
    ThermalSourceQuadrature,
    PreparedThermalGreenField,
    ContinuousThermalGreenArtifact,
    build_thermal_source_quadrature,
)
from .loss import ConductorLossField
from .field import PreparedUniformLossField, UniformLossFieldDecoder
from .reference import MixedReferenceArtifact, PreparedReferenceLossField
from .system import MeshfreeVNextSystem, SystemCapabilities
from .certification import PortCertificate, certify_port_result
from .certified import (
    CertifiedPortResult,
    certify_mqs_ports,
    certify_mixed_ports,
)
from .convergence import (
    ConvergenceStep,
    ConvergenceReport,
    impedance_convergence,
    mixed_impedance_convergence,
)
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
from .training_data import SpatialLossSamples, TeacherSample
from .sampling import MVPSceneSamplerConfig, sample_two_coil_mvp_scene
from .dataset import (
    DATASET_SCHEMA,
    SPLITS,
    DatasetRecord,
    ImmutableTeacherDataset,
    deterministic_split,
    migrate_dataset_v3_to_v4,
)
from .serialization import scene_from_dict, scene_to_dict
from .evaluation import SurrogateAudit, audit_surrogate
from .inference import run_system_inference
from .spatial_evaluation import SpatialSurrogateAudit, audit_spatial_surrogate
from .active_learning import (
    ActiveLearningCandidate,
    ActiveLearningRound,
    run_active_learning_round,
    scene_regime_vector,
    score_active_learning_candidates,
    select_diverse_candidates,
)
from .bundle import (
    BUNDLE_SCHEMA,
    LoadedVNextBundle,
    load_bundle,
    publish_bundle,
)
from .uncertainty import (
    FastErrorCalibrator,
    CalibratedFastPrediction,
    calibrated_fast_predict,
    fast_error_indicator,
    fit_fast_error_calibrator,
)
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
    "MatrixFreeMetadata",
    "MatrixFreeMQSOperator",
    "MatrixFreeMixedMetadata",
    "MatrixFreeMixedOperator",
    "StableThermalModel",
    "HomogeneousThermalMedium",
    "ThermalSourceQuadrature",
    "PreparedThermalGreenField",
    "ContinuousThermalGreenArtifact",
    "build_thermal_source_quadrature",
    "ConductorLossField",
    "PreparedUniformLossField",
    "UniformLossFieldDecoder",
    "MixedReferenceArtifact",
    "PreparedReferenceLossField",
    "MeshfreeVNextSystem",
    "SystemCapabilities",
    "PortCertificate",
    "certify_port_result",
    "CertifiedPortResult",
    "certify_mqs_ports",
    "certify_mixed_ports",
    "CertifiedReleaseAudit",
    "audit_certified_release",
    "certify_mixed_ports",
    "ConvergenceStep",
    "ConvergenceReport",
    "impedance_convergence",
    "mixed_impedance_convergence",
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
    "SpatialLossSamples",
    "TeacherSample",
    "MVPSceneSamplerConfig",
    "sample_two_coil_mvp_scene",
    "DATASET_SCHEMA",
    "SPLITS",
    "DatasetRecord",
    "ImmutableTeacherDataset",
    "deterministic_split",
    "migrate_dataset_v3_to_v4",
    "scene_from_dict",
    "scene_to_dict",
    "SurrogateAudit",
    "audit_surrogate",
    "run_system_inference",
    "SpatialSurrogateAudit",
    "audit_spatial_surrogate",
    "ActiveLearningCandidate",
    "ActiveLearningRound",
    "run_active_learning_round",
    "scene_regime_vector",
    "score_active_learning_candidates",
    "select_diverse_candidates",
    "BUNDLE_SCHEMA",
    "LoadedVNextBundle",
    "load_bundle",
    "publish_bundle",
    "FastErrorCalibrator",
    "CalibratedFastPrediction",
    "calibrated_fast_predict",
    "fast_error_indicator",
    "fit_fast_error_calibrator",
    "MU0",
    "coaxial_circular_mutual_inductance",
    "dc_resistance",
    "lumped_thermal_step",
    "neumann_mutual_inductance",
    "superellipse_area",
    "thermal_impulse_temperature",
    "thermal_step_temperature",
]
