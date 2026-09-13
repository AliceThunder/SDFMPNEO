"""SDF-MPNEO unified multiscale full-space neural electrothermal solver."""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_dataset import MaxwellResidualDataset, generate_residual_dataset
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from .unified_maxwell import MaxwellSolveReport, NeuralMaxwellAccelerator
from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_neural_operator import (
    EdgeMultiscaleConfig,
    OperatorGraph,
    build_edge_residual_operator,
    edge_multiscale_group_ids,
    edge_static_features,
    operator_feature_statistics,
    residual_features,
)
from .unified_thermal import ThermalBasisReport, build_thermal_basis
from .unified_trainer import MaxwellTrainingConfig, MaxwellTrainingReport, train_maxwell_accelerator

__version__ = "0.13.0"

__all__ = [
    "ARCHITECTURE", "BackgroundContext", "CoilGeometry", "EdgeMultiscaleConfig",
    "FixedMultiscaleBackground", "MaxwellResidualDataset", "MaxwellSolveReport",
    "MaxwellTrainingConfig", "MaxwellTrainingReport", "NeuralMaxwellAccelerator",
    "OperatorGraph", "PackageGeometry", "Pose", "ThermalBasisReport",
    "UnifiedNeuralElectroThermalModel", "UnifiedPrediction", "UnifiedSteadyState",
    "UnifiedUWPTGeometry", "build_edge_residual_operator", "build_thermal_basis",
    "edge_multiscale_group_ids", "edge_static_features", "generate_residual_dataset",
    "operator_feature_statistics", "residual_features", "sample_geometry", "stretched_axis",
    "train_maxwell_accelerator",
]
