"""Structure-preserving quadratic-current neural electrothermal ROM."""

from .adapters import (
    fixed_research_tensor_factory,
    fixed_research_thermal_family,
    fixed_research_thermal_rhs_forcing,
    geometry_research_embedded_thermal_family,
    geometry_research_tensor_factory,
    geometry_research_thermal_rhs_forcing,
)
from .batch import BatchedNeuralROMPrediction, predict_batch_fixed_etd2
from .dataset import QuadraticJouleDataset, SnapshotManifest, frozen_split_indices, latin_hypercube_box
from .disk_dataset import DiskQuadraticJouleDataset
from .generator import generate_snapshots_resumable
from .geometry_thermal import AffineGeometryThermalOperatorFamily
from .integrators import (
    GeneralizedThermalSpectrum,
    IntegrationResult,
    integrate_etd2,
    integrate_etd2_adaptive,
    integrate_imex_euler,
)
from .model import NeuralROMPrediction, NeuralROMSteadyState, StructurePreservingNeuralElectroThermalROM
from .network import FeatureNormalizer, ResidualMLPConfig, build_residual_mlp
from .physical_layer import decode_heat_source_batch_numpy, decode_heat_source_numpy, decode_heat_source_torch
from .pipeline import PipelineResult, build_fixed_neural_rom, build_geometry_neural_rom, retrain_neural_rom
from .pod import TensorPOD, fit_dataset_pod, fit_tensor_pod
from .quadratic_joule import augmented_operating_vector, quadratic_heat_source, quadratic_heat_source_batch, quadratic_joule_tensor
from .surrogate import NeuralTensorSurrogate, PreparedOperatingQuadraticLayer
from .symmetric import quadratic_feature, smat, svec, symmetric_packed_size, tensor_smat, tensor_svec
from .trainer import NeuralTrainingConfig, NeuralTrainingReport, train_tensor_surrogate
from .vector_field import FixedThermalOperatorFamily, NeuralElectroThermalVectorField, ReducedThermalOperator

__all__ = [
    "AffineGeometryThermalOperatorFamily",
    "BatchedNeuralROMPrediction",
    "DiskQuadraticJouleDataset",
    "FeatureNormalizer",
    "FixedThermalOperatorFamily",
    "GeneralizedThermalSpectrum",
    "IntegrationResult",
    "NeuralElectroThermalVectorField",
    "NeuralROMPrediction",
    "NeuralROMSteadyState",
    "NeuralTensorSurrogate",
    "NeuralTrainingConfig",
    "NeuralTrainingReport",
    "PipelineResult",
    "PreparedOperatingQuadraticLayer",
    "QuadraticJouleDataset",
    "ReducedThermalOperator",
    "ResidualMLPConfig",
    "SnapshotManifest",
    "StructurePreservingNeuralElectroThermalROM",
    "TensorPOD",
    "augmented_operating_vector",
    "build_fixed_neural_rom",
    "build_geometry_neural_rom",
    "build_residual_mlp",
    "decode_heat_source_batch_numpy",
    "decode_heat_source_numpy",
    "decode_heat_source_torch",
    "fit_dataset_pod",
    "fit_tensor_pod",
    "fixed_research_tensor_factory",
    "fixed_research_thermal_family",
    "fixed_research_thermal_rhs_forcing",
    "frozen_split_indices",
    "generate_snapshots_resumable",
    "geometry_research_embedded_thermal_family",
    "geometry_research_tensor_factory",
    "geometry_research_thermal_rhs_forcing",
    "integrate_etd2",
    "integrate_etd2_adaptive",
    "integrate_imex_euler",
    "latin_hypercube_box",
    "predict_batch_fixed_etd2",
    "quadratic_feature",
    "quadratic_heat_source",
    "quadratic_heat_source_batch",
    "quadratic_joule_tensor",
    "retrain_neural_rom",
    "smat",
    "svec",
    "symmetric_packed_size",
    "tensor_smat",
    "tensor_svec",
    "train_tensor_surrogate",
]
