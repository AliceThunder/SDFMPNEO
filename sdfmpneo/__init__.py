"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

Production API:
``geometry -> deterministic Phi(g),Mr(g),Kr(g) + neural EM tensors -> explicit
current/circuit physics -> true thermal ROM``.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from . import unified_model as _unified_model

# Physics truth version 15 restores the theory-defined open two-terminal source
# and replaces transverse-source projection with a compatible scalar-gradient
# Maxwell block. Old truth tensors/certificates must not be mixed with this one.
_unified_model.FORMAT_VERSION = 15

from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_open_boundary import OpenBoundaryBackground
from .unified_maxwell_operator_metadata import install as _install_maxwell_operator_metadata

# Preserve the physical open-path source.  The longitudinal terminal response is
# solved by the compatible gradient block; it is not projected out of the RHS.
_install_maxwell_operator_metadata(OpenBoundaryBackground)

from . import unified_self_correction as _self_correction
from . import unified_certified_local_solve as _certified_local_solve
from .unified_fast_local_krylov import install as _install_fast_local_krylov
from .unified_local_solve_cache import install as _install_local_solve_cache

# Linear-algebra acceleration changes neither the physical operator nor any Gate.
# Exact memoization is installed only after the certified solver exists.
_install_fast_local_krylov(_certified_local_solve)
_certified_local_solve.install(_self_correction)
_install_local_solve_cache(_self_correction)

# Install the same certified large-system policy on global truth/Gate solves.
# This must happen before corrected preflight imports _solve_fields by value.
from . import unified_tensor_surrogate as _tensor_surrogate
from . import unified_physics_gate as _physics_gate
from .unified_fast_global_maxwell import install as _install_fast_global_maxwell

_install_fast_global_maxwell(_physics_gate, _tensor_surrogate, _certified_local_solve)

from . import unified_truth_preflight as _truth_preflight
from .unified_source_preflight_patch import install as _install_source_preflight

_install_source_preflight(_truth_preflight)

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

__version__ = "0.17.0"

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
