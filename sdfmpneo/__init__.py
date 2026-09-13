"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

The production API is the static geometry -> EM tensors -> thermal ROM path.
Legacy full-space neural-Maxwell symbols remain importable for old research
scripts, but are intentionally not part of the production ``__all__`` surface.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_tensor_surrogate import (
    DecodedTensors,
    TensorDataset,
    UnifiedTensorSurrogate,
    decode_physical_tensors,
    encode_geometry,
    pack_tensors,
    solve_truth_tensors,
)
from .unified_tensor_training import TensorTrainingReport, train_matrix_tensor_surrogate
from .unified_thermal import ThermalBasisReport, build_thermal_basis

# Backward-compatible research imports.  run.py and the production API do not
# use these paths anymore.
from .unified_dataset import MaxwellResidualDataset, generate_residual_dataset
from .unified_maxwell import MaxwellSolveReport, NeuralMaxwellAccelerator
from .unified_neural_operator import (
    EdgeMultiscaleConfig,
    OperatorGraph,
    build_edge_residual_operator,
    edge_multiscale_group_ids,
    edge_static_features,
    operator_feature_statistics,
    residual_features,
)
from .unified_trainer import MaxwellTrainingConfig, MaxwellTrainingReport, train_maxwell_accelerator

__version__ = "0.13.0"

__all__ = [
    "ARCHITECTURE",
    "BackgroundContext",
    "CoilGeometry",
    "DecodedTensors",
    "FixedMultiscaleBackground",
    "PackageGeometry",
    "Pose",
    "TensorDataset",
    "TensorTrainingReport",
    "ThermalBasisReport",
    "UnifiedNeuralElectroThermalModel",
    "UnifiedPrediction",
    "UnifiedSteadyState",
    "UnifiedTensorSurrogate",
    "UnifiedUWPTGeometry",
    "build_thermal_basis",
    "decode_physical_tensors",
    "encode_geometry",
    "pack_tensors",
    "sample_geometry",
    "solve_truth_tensors",
    "stretched_axis",
    "train_matrix_tensor_surrogate",
]
