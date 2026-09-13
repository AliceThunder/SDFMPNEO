"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

The package production API is the single static
``geometry -> EM tensors -> explicit current/circuit physics -> thermal ROM``
architecture.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_open_boundary import OpenBoundaryBackground
from .unified_tensor_surrogate import (
    DecodedTensors,
    TensorDataset,
    UnifiedTensorSurrogate,
    decode_physical_tensors,
    encode_geometry,
    pack_tensors,
    solve_port_truth_tensors,
    solve_truth_tensors,
)
from .unified_tensor_training import TensorTrainingReport, train_matrix_tensor_surrogate
from .unified_thermal import ThermalBasisReport, build_thermal_basis

__version__ = "0.13.0"

__all__ = [
    "ARCHITECTURE",
    "BackgroundContext",
    "CoilGeometry",
    "DecodedTensors",
    "FixedMultiscaleBackground",
    "OpenBoundaryBackground",
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
    "solve_port_truth_tensors",
    "solve_truth_tensors",
    "stretched_axis",
    "train_matrix_tensor_surrogate",
]