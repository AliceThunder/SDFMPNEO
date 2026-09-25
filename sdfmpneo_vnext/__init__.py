"""SDF-MPNEO vNext mesh-free-first electrothermal runtime.

The vNext package is intentionally independent from the legacy fixed-grid
runtime. Optional PyTorch-backed neural artifacts are imported lazily by their
own modules so the physical REFERENCE/CERTIFIED stack remains NumPy/SciPy only.
"""

from .geometry import (
    RigidPose,
    SuperellipseSpiral,
    PolylineConductor,
    bishop_segment_frames,
    haar_rotation,
)
from .package_geometry import (
    SuperquadricPackageGeometry,
    SuperquadricSurfaceQuadrature,
    SuperquadricVolumeQuadrature,
)
from .scene import (
    ConductorMaterial,
    HomogeneousMedium,
    IsotropicMaterial,
    DebyeMaterial,
    PackageObject,
    CoilObject,
    Scene,
)
from .basis import (
    SectionBasis,
    SectionQuadrature,
    adaptive_section_basis,
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
from .field import (
    PreparedUniformLossField,
    UniformLossFieldDecoder,
)
from .reference import (
    MixedReferenceArtifact,
    PreparedReferenceLossField,
)
from .system import (
    MeshfreeVNextSystem,
    SystemCapabilities,
    mvp_system_capabilities,
)
from .certification import (
    PortCertificate,
    certify_port_result,
)
from .certified import (
    CertifiedPortResult,
    certify_mqs_ports,
    certify_mixed_ports,
)
from .certified_evaluation import (
    CertifiedReleaseAudit,
    audit_certified_release,
)
from .convergence import (
    ConvergenceStep,
    ConvergenceReport,
    ReferenceConvergenceDirection,
    ReferenceConvergenceReport,
    impedance_convergence,
    mixed_impedance_convergence,
    mixed_reference_convergence,
)
from .electrothermal import (
    CoilThermalProperties,
    ElectroThermalStep,
    CurrentControlledEnvelope,
    VoltageControlledEnvelope,
    build_lumped_coil_thermal_model,
)
from .channel_thermal import (
    ThermalNodeProperties,
    ChannelElectroThermalStep,
    ChannelResolvedCurrentEnvelope,
    ChannelResolvedVoltageEnvelope,
    build_lumped_channel_thermal_model,
)
from .prediction import StructuredPortPrediction
from .dielectric_surface import (
    DielectricSurfaceResult,
    DielectricSurfaceSolver,
)
from .hybrid_dielectric import (
    DielectricCoupledMixedTeacher,
    DielectricCoupledResult,
    DielectricCoupledReferenceArtifact,
)
from .hybrid_field import (
    PreparedHybridReferenceLossField,
    prepare_hybrid_reference_loss_field,
)
from .hybrid_certified import certify_dielectric_ports
from .hybrid_convergence import (
    HybridReferenceConvergenceDirection,
    HybridReferenceConvergenceReport,
    hybrid_reference_convergence,
)
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
from .features import (
    EncodedScene,
    encode_scene_invariant,
)
from .hybrid_features import (
    EncodedHybridScene,
    encode_hybrid_scene_invariant,
)
from .training_data import (
    SpatialLossSamples,
    TeacherSample,
)
from .hybrid_training_data import (
    HYBRID_REFERENCE_BACKEND,
    HybridTeacherSample,
    PackageSpatialLossSamples,
)
from .hybrid_dataset import (
    HYBRID_DATASET_SCHEMA,
    HybridDatasetRecord,
    ImmutableHybridTeacherDataset,
)
from .sampling import (
    MVPSceneSamplerConfig,
    HybridSceneSamplerConfig,
    sample_two_coil_mvp_scene,
    sample_hybrid_package_scene,
)
from .dataset import (
    DATASET_SCHEMA,
    SPLITS,
    DatasetRecord,
    ImmutableTeacherDataset,
    deterministic_split,
    migrate_dataset_v3_to_v4,
)
from .serialization import (
    scene_from_dict,
    scene_to_dict,
)
from .evaluation import (
    SurrogateAudit,
    audit_surrogate,
)
from .inference import run_system_inference
from .spatial_evaluation import (
    SpatialSurrogateAudit,
    audit_spatial_surrogate,
)
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
    "SuperquadricPackageGeometry",
    "SuperquadricSurfaceQuadrature",
    "SuperquadricVolumeQuadrature",
    "SuperellipseSpiral",
    "PolylineConductor",
    "bishop_segment_frames",
    "haar_rotation",
    "ConductorMaterial",
    "HomogeneousMedium",
    "IsotropicMaterial",
    "DebyeMaterial",
    "PackageObject",
    "CoilObject",
    "Scene",
    "SectionBasis",
    "SectionQuadrature",
    "adaptive_section_basis",
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
    "PreparedHybridReferenceLossField",
    "prepare_hybrid_reference_loss_field",
    "MixedReferenceArtifact",
    "PreparedReferenceLossField",
    "MeshfreeVNextSystem",
    "SystemCapabilities",
    "mvp_system_capabilities",
    "PortCertificate",
    "certify_port_result",
    "CertifiedPortResult",
    "certify_mqs_ports",
    "certify_dielectric_ports",
    "HybridReferenceConvergenceDirection",
    "HybridReferenceConvergenceReport",
    "hybrid_reference_convergence",
    "certify_mixed_ports",
    "CertifiedReleaseAudit",
    "audit_certified_release",
    "ConvergenceStep",
    "ConvergenceReport",
    "ReferenceConvergenceDirection",
    "ReferenceConvergenceReport",
    "impedance_convergence",
    "mixed_impedance_convergence",
    "mixed_reference_convergence",
    "CoilThermalProperties",
    "ElectroThermalStep",
    "CurrentControlledEnvelope",
    "VoltageControlledEnvelope",
    "build_lumped_coil_thermal_model",
    "ThermalNodeProperties",
    "ChannelElectroThermalStep",
    "ChannelResolvedCurrentEnvelope",
    "ChannelResolvedVoltageEnvelope",
    "build_lumped_channel_thermal_model",
    "StructuredPortPrediction",
    "DielectricSurfaceResult",
    "DielectricSurfaceSolver",
    "DielectricCoupledMixedTeacher",
    "DielectricCoupledResult",
    "DielectricCoupledReferenceArtifact",
    "FastCurrentControlledEnvelope",
    "FastVoltageControlledEnvelope",
    "AnalyticBaselineArtifact",
    "AnalyticBaselineResult",
    "analytic_port_baseline",
    "pair_mutual_inductance",
    "regularized_self_inductance",
    "EncodedScene",
    "encode_scene_invariant",
    "EncodedHybridScene",
    "encode_hybrid_scene_invariant",
    "SpatialLossSamples",
    "TeacherSample",
    "HYBRID_REFERENCE_BACKEND",
    "HybridTeacherSample",
    "PackageSpatialLossSamples",
    "HYBRID_DATASET_SCHEMA",
    "HybridDatasetRecord",
    "ImmutableHybridTeacherDataset",
    "MVPSceneSamplerConfig",
    "HybridSceneSamplerConfig",
    "sample_two_coil_mvp_scene",
    "sample_hybrid_package_scene",
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
