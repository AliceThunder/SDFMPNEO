"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

Production API:
``geometry -> deterministic Phi(g),Mr(g),Kr(g) + neural EM tensors -> explicit
current/circuit physics -> true thermal ROM``.
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
from .unified_thermal import (
    GeometryAwareThermalLibrary,
    ThermalBasisReport,
    audit_geometry_aware_thermal_trajectories,
    build_geometry_aware_thermal_library,
)

__version__ = "0.14.0"

__all__ = [
    "ARCHITECTURE",
    "BackgroundContext",
    "CoilGeometry",
    "DecodedTensors",
    "FixedMultiscaleBackground",
    "GeometryAwareThermalLibrary",
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
    "audit_geometry_aware_thermal_trajectories",
    "build_geometry_aware_thermal_library",
    "decode_physical_tensors",
    "encode_geometry",
    "pack_tensors",
    "sample_geometry",
    "solve_port_truth_tensors",
    "solve_truth_tensors",
    "stretched_axis",
    "train_matrix_tensor_surrogate",
]
