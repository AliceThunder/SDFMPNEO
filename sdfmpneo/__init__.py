"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

Production API:
``geometry -> deterministic Phi(g),Mr(g),Kr(g) + neural EM tensors -> explicit
current/circuit physics -> true thermal ROM``.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from . import unified_model as _unified_model

# Physics truth version 25 keeps the full open two-terminal source, resolves its
# physical source/package support below the production EM cell scale, declares
# terminal charge continuity independently from incidental edge deposition, and
# resolves the remaining longitudinal self near field by nested terminal-contact
# refinement.  The global scalar solution still owns the nonlocal return path and
# supplies every artificial-patch Dirichlet trace; only cells intersecting the
# physical feed/return contact support receive sub-mm scalar refinement.
_unified_model.FORMAT_VERSION = 25
_SELF_CORRECTION_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"

from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_open_boundary import OpenBoundaryBackground
from .unified_resolved_package_fraction import install as _install_resolved_package_fraction
from .unified_terminal_contact_source import install as _install_terminal_contact_source
from .unified_charge_regularized_source import install as _install_charge_regularized_source
from .unified_maxwell_operator_metadata import install as _install_maxwell_operator_metadata

# Package and source geometry are physical. Numerical quadrature may become more
# resolved with mesh refinement but must not change the represented OBB,
# conductor rectangle, contact length or ampere-turns.  The charge-lift layer is
# installed after physical source deposition so it can replace only the discrete
# longitudinal continuity while preserving source curl exactly.
_install_resolved_package_fraction(OpenBoundaryBackground)
_install_terminal_contact_source(OpenBoundaryBackground)
_install_charge_regularized_source(OpenBoundaryBackground)
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

_corrected_preflight._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL
_install_preflight_diagnosis(_corrected_preflight)

# The canonical local box certifies/refines only the localizable transverse/cross
# self defect.  The nonlocal longitudinal problem remains global, while its
# unresolved source near field is corrected on a Cartesian scalar patch whose
# boundary potential is inherited from that global solution.
from . import unified_corrected_truth as _corrected_truth
from . import unified_global_longitudinal_reference as _global_longitudinal_reference
from .unified_longitudinal_patch_consistency import install as _install_longitudinal_patch_consistency
from .unified_terminal_longitudinal_refinement import install as _install_terminal_longitudinal_refinement
from .unified_scalar_charge_patch import install as _install_scalar_charge_patch
from .unified_global_longitudinal_reference import install as _install_global_longitudinal_reference

# Start from full-geometry v2 patch semantics: every scalar patch retains the
# complete global geometry/material problem and selects only the target-port RHS.
_global_longitudinal_reference._MODEL = "global_boundary_conditioned_longitudinal_nearfield_defect_v2"
# Certify the coarse patch as the exact parent-scalar restriction, then replace
# whole-package 3/2.25-mm refinement with nested terminal-contact-scale nodes.
# The refined scalar-only states consume q_target directly so they do not build
# a redundant edge-space G.T G charge-lift factor in addition to G.T D G.
_install_longitudinal_patch_consistency(_global_longitudinal_reference)
_install_terminal_longitudinal_refinement(_global_longitudinal_reference)
_install_scalar_charge_patch(_global_longitudinal_reference)
_install_global_longitudinal_reference(_corrected_preflight, _corrected_truth)

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

# v25 changes the longitudinal reference discretization from whole-package
# millimetre refinement to nested terminal-contact-scale refinement.  Invalidate
# every earlier physical cache/release so no v24 defect is reused.
from . import unified_runtime as _unified_runtime
_unified_runtime._CACHE_FORMAT = 27
_unified_runtime._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL

# Resolve deterministic scalar-patch settings every time the production
# background is constructed, including physical-cache hits that skip preflight.
# The runtime signature is computed before build_background(), so this does not
# alter cache-key semantics beyond the explicit cache-format bump above.
_original_runtime_build_background = _unified_runtime.build_background

def _build_background_with_longitudinal_reference(settings, *args, **kwargs):
    background = _original_runtime_build_background(settings, *args, **kwargs)
    _global_longitudinal_reference._resolve_settings(settings, background)
    return background

_unified_runtime.build_background = _build_background_with_longitudinal_reference

# Post-basis mesh/thermal Gates must consume the exact same corrected truth as
# preflight and tensor-label generation.
from . import unified_corrected_physics_gate as _corrected_physics_gate
from .unified_global_longitudinal_physics_gate import install as _install_global_longitudinal_physics_gate

_corrected_physics_gate._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL
_install_global_longitudinal_physics_gate(
    _corrected_physics_gate, _global_longitudinal_reference
)
