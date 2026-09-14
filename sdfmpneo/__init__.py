"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

Production API:
``geometry -> deterministic Phi(g),Mr(g),Kr(g) + neural EM tensors -> explicit
current/circuit physics -> true thermal ROM``.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_open_boundary import OpenBoundaryBackground
from .unified_transverse_source import install as _install_transverse_source

_install_transverse_source(OpenBoundaryBackground)

from . import unified_self_correction as _self_correction
from .unified_certified_local_solve import install as _install_certified_local_solve

_install_certified_local_solve(_self_correction)

from . import unified_corrected_truth_preflight as _corrected_preflight
from .unified_preflight_diagnosis_patch import install as _install_preflight_diagnosis

_install_preflight_diagnosis(_corrected_preflight)

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

__version__ = "0.15.0"

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
