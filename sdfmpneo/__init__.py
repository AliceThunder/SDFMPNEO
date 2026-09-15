"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

Production API:
``geometry -> deterministic Phi(g),Mr(g),Kr(g) + neural EM tensors -> explicit
current/circuit physics -> true thermal ROM``.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from . import unified_model as _unified_model

# Physics truth version 18 keeps the full open two-terminal source, resolves its
# physical source/package support below the production EM cell scale, and applies
# the complete local fine-minus-coarse self defect (v3). Old truth tensors and
# certificates must not be mixed with this one.
_unified_model.FORMAT_VERSION = 18
_SELF_CORRECTION_MODEL_V3 = "canonical_local_full_fine_minus_coarse_self_defect_v3"

from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_open_boundary import OpenBoundaryBackground
from .unified_resolved_package_fraction import install as _install_resolved_package_fraction
from .unified_terminal_contact_source import install as _install_terminal_contact_source
from .unified_maxwell_operator_metadata import install as _install_maxwell_operator_metadata

# Package and source geometry are physical.  Numerical subcell/quadrature
# resolution may increase with refinement but must not change the represented
# OBB, conductor rectangle, terminal contact or total ampere-turns.
_install_resolved_package_fraction(OpenBoundaryBackground)
_install_terminal_contact_source(OpenBoundaryBackground)
_install_maxwell_operator_metadata(OpenBoundaryBackground)

from . import unified_self_correction as _self_correction
from . import unified_certified_local_solve as _certified_local_solve
from . import unified_two_level_local_krylov as _two_level_local_krylov
from .unified_fast_local_krylov import install as _install_fast_local_krylov
from .unified_hcurl_warm_start import install as _install_hcurl_warm_start
from .unified_two_level_local_krylov import install as _install_two_level_local_krylov
from .unified_two_level_residual_replacement import install as _install_two_level_residual_replacement
from .unified_localized_self_solve import install as _install_localized_self_solve
from .unified_stable_localized_self import install as _install_stable_localized_self
from .unified_local_solve_cache import install as _install_local_solve_cache
from .unified_full_self_defect import install as _install_full_self_defect

# Linear-algebra acceleration changes neither the physical operator nor any Gate.
_install_fast_local_krylov(_certified_local_solve)
_install_hcurl_warm_start(_certified_local_solve)
_install_two_level_local_krylov(_certified_local_solve)
_install_two_level_residual_replacement(_two_level_local_krylov, _certified_local_solve)
_certified_local_solve.install(_self_correction)
# Solve the compatible longitudinal/transverse components directly so the tiny
# transverse field is never recovered by subtracting two huge full fields.
_install_localized_self_solve(_self_correction, _certified_local_solve)
_install_stable_localized_self(_self_correction)
# Exact memoization is installed after the final local truth implementation.
_install_local_solve_cache(_self_correction)
# v3 changes only the object added back to the global diagonal: the complete
# local fine-minus-coarse defect. Absolute local terminal response is never
# substituted for the global one.
_install_full_self_defect(_self_correction)

# Install the v3 audit before corrected preflight imports the audit function by
# value. It certifies the same complete defect used by production truth.
from . import unified_self_correction_audit as _self_correction_audit
from .unified_full_self_defect_audit import install as _install_full_self_defect_audit
_install_full_self_defect_audit(_self_correction_audit, _self_correction)

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

_corrected_preflight._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL_V3
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

# v18 changes source/material subcell integration and the production self-defect
# semantics. Invalidate all earlier physical caches/release metadata.
from . import unified_runtime as _unified_runtime
_unified_runtime._CACHE_FORMAT = 20
_unified_runtime._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL_V3

# These modules are imported by unified_runtime. Their functions read the
# module-level model tag at call time, so synchronize the release/Gate metadata.
from . import unified_corrected_physics_gate as _corrected_physics_gate
_corrected_physics_gate._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL_V3
