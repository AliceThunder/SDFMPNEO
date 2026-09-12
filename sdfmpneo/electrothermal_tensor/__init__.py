"""Structure-preserving quadratic-current neural electrothermal ROM."""

from .adapters import (
    fixed_research_tensor_factory,
    fixed_research_thermal_family,
    geometry_research_tensor_factory,
    geometry_research_thermal_family,
)
from .dataset import (
    QuadraticJouleDataset,
    SnapshotManifest,
    frozen_split_indices,
    generate_snapshot_dataset,
    latin_hypercube_box,
)
from .generator import generate_snapshots_resumable
from .integrators import (
    GeneralizedETD2Stepper,
    IntegrationResult,
    integrate_etd2,
    integrate_imex_euler,
    integrate_reference,
)
from .model import (
    NeuralROMPrediction,
    NeuralROMSteadyState,
    StructurePreservingNeuralElectroThermalROM,
)
from .network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .physical_layer import (
    decode_heat_source_batch_numpy,
    decode_heat_source_numpy,
    decode_heat_source_torch,
)
from .pipeline import PipelineResult, build_fixed_neural_rom, build_geometry_neural_rom
from .pod import TensorPOD, fit_dataset_pod, fit_tensor_pod
from .quadratic_joule import (
    augmented_operating_vector,
    quadratic_heat_source,
    quadratic_heat_source_batch,
    quadratic_joule_tensor,
)
from .surrogate import NeuralTensorSurrogate
from .symmetric import (
    quadratic_feature,
    quadratic_from_svec,
    smat,
    svec,
    symmetric_packed_size,
    tensor_smat,
    tensor_svec,
)
from .trainer import NeuralTrainingConfig, NeuralTrainingReport, train_tensor_surrogate
from .validation import (
    StabilityReport,
    SurrogateValidationReport,
    VectorFieldValidationReport,
    analyze_stability,
    energy_logarithmic_norm,
    tensor_to_heat_error_bound,
    trajectory_error_bound,
    validate_surrogate_on_dataset,
    validate_vector_field,
)
from .vector_field import (
    CallableThermalOperatorFamily,
    FixedThermalOperatorFamily,
    NeuralElectroThermalVectorField,
    ReducedThermalOperator,
)

__all__ = [
    "CallableThermalOperatorFamily",
    "FeatureNormalizer",
    "FixedThermalOperatorFamily",
    "GeneralizedETD2Stepper",
    "IntegrationResult",
    "NeuralElectroThermalVectorField",
    "NeuralROMPrediction",
    "NeuralROMSteadyState",
    "NeuralTensorSurrogate",
    "NeuralTrainingConfig",
    "NeuralTrainingReport",
    "PipelineResult",
    "QuadraticJouleDataset",
    "ReducedThermalOperator",
    "ResidualMLPConfig",
    "SnapshotManifest",
    "StabilityReport",
    "StructurePreservingNeuralElectroThermalROM",
    "SurrogateValidationReport",
    "TensorPOD",
    "VectorFieldValidationReport",
    "analyze_stability",
    "augmented_operating_vector",
    "build_fixed_neural_rom",
    "build_geometry_neural_rom",
    "build_residual_mlp",
    "decode_heat_source_batch_numpy",
    "decode_heat_source_numpy",
    "decode_heat_source_torch",
    "energy_logarithmic_norm",
    "fit_dataset_pod",
    "fit_tensor_pod",
    "fixed_research_tensor_factory",
    "fixed_research_thermal_family",
    "frozen_split_indices",
    "generate_snapshot_dataset",
    "generate_snapshots_resumable",
    "geometry_research_tensor_factory",
    "geometry_research_thermal_family",
    "integrate_etd2",
    "integrate_imex_euler",
    "integrate_reference",
    "latin_hypercube_box",
    "quadratic_feature",
    "quadratic_from_svec",
    "quadratic_heat_source",
    "quadratic_heat_source_batch",
    "quadratic_joule_tensor",
    "smat",
    "svec",
    "symmetric_packed_size",
    "tensor_smat",
    "tensor_svec",
    "tensor_to_heat_error_bound",
    "train_tensor_surrogate",
    "trajectory_error_bound",
    "validate_surrogate_on_dataset",
    "validate_vector_field",
]
