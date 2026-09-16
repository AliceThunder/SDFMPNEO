"""SDF-MPNEO geometry-to-tensor electrothermal ROM.

Production API:
``geometry -> deterministic Phi(g),Mr(g),Kr(g) + neural EM tensors -> explicit
current/circuit physics -> true thermal ROM``.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from . import unified_model as _unified_model

# Physics truth version 31 keeps the full open two-terminal source, resolves its
# physical source/package support below the production EM cell scale, declares
# terminal charge continuity independently from incidental edge deposition, and
# splits longitudinal self correction into independently certified reactive and
# dissipative near-field pieces. The reactive term uses the inherited-boundary
# full-port terminal patch. Dissipative self loss now keeps the complete balanced
# full-port q_target in every solve, refines one terminal at a time, and contracts
# Joule energy only over disjoint parent-cell-aligned terminal windows. This
# avoids inadmissible isolated net-charge scalar solves while leaving smooth
# feed/return cross-field physics global. The falsified whole-domain 9->6.75-mm
# dissipative reference remains disabled in production.
_unified_model.FORMAT_VERSION = 31
_SELF_CORRECTION_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"

from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_open_boundary import OpenBoundaryBackground
from .unified_resolved_package_fraction import install as _install_resolved_package_fraction
from .unified_terminal_contact_source import install as _install_terminal_contact_source
from .unified_charge_regularized_source import install as _install_charge_regularized_source
from .unified_maxwell_operator_metadata import install as _install_maxwell_operator_metadata

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

_install_fast_local_krylov(_certified_local_solve)
_install_hcurl_warm_start(_certified_local_solve)
_install_two_level_local_krylov(_certified_local_solve)
_install_two_level_residual_replacement(_two_level_local_krylov, _certified_local_solve)
_certified_local_solve.install(_self_correction)
_install_localized_self_solve(_self_correction, _certified_local_solve)
_install_stable_localized_self(_self_correction)
_install_local_solve_cache(_self_correction)

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

from . import unified_corrected_truth as _corrected_truth
from . import unified_global_longitudinal_reference as _global_longitudinal_reference
from .unified_longitudinal_patch_consistency import install as _install_longitudinal_patch_consistency
from .unified_terminal_longitudinal_refinement import install as _install_terminal_longitudinal_refinement
from .unified_scalar_charge_patch import install as _install_scalar_charge_patch
from .unified_reactive_longitudinal_reference import install as _install_reactive_longitudinal_reference
from . import unified_global_dissipative_reference as _global_dissipative_reference_impl
from .unified_global_dissipative_reference import install as _install_global_dissipative_reference
from .unified_resolved_dissipative_reference import install as _install_resolved_dissipative_reference
from .unified_terminal_dissipative_defect import install as _install_terminal_dissipative_defect
from .unified_longitudinal_preflight_schedule import install as _install_longitudinal_preflight_schedule
from .unified_global_longitudinal_reference import install as _install_global_longitudinal_reference

_global_longitudinal_reference._MODEL = "global_boundary_conditioned_longitudinal_nearfield_defect_v2"
_install_longitudinal_patch_consistency(_global_longitudinal_reference)
_install_terminal_longitudinal_refinement(_global_longitudinal_reference)
_install_scalar_charge_patch(_global_longitudinal_reference)
# Full-port terminal patches independently certify only the reactive longitudinal
# defect. Their absolute local Dvol remains diagnostic-only.
_install_reactive_longitudinal_reference(_global_longitudinal_reference)
# Keep the historical whole-domain dissipative implementation available for
# diagnostics/regression, including exact sigma/epsilon reference support. The
# balanced terminal-local installer immediately disables that uniform production
# reference and installs the source-scale dissipative truth instead.
_install_global_dissipative_reference(_global_longitudinal_reference)
_install_resolved_dissipative_reference(
    _global_longitudinal_reference,
    _global_dissipative_reference_impl,
)
_install_terminal_dissipative_defect(
    _global_longitudinal_reference,
    _global_dissipative_reference_impl,
)
_install_global_longitudinal_reference(_corrected_preflight, _corrected_truth)
_install_longitudinal_preflight_schedule(
    _corrected_preflight, _global_longitudinal_reference
)

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

# v31 replaces the transient v30 net-charge-component idea with the admissible
# balanced full-port / terminal-local energy-defect truth. Old v29/v30 artifacts
# cannot be mixed with this definition.
from . import unified_runtime as _unified_runtime
_unified_runtime._CACHE_FORMAT = 33
_unified_runtime._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL

_original_runtime_build_background = _unified_runtime.build_background

def _build_background_with_longitudinal_reference(settings, *args, **kwargs):
    background = _original_runtime_build_background(settings, *args, **kwargs)
    _global_longitudinal_reference._resolve_settings(settings, background)
    return background

_unified_runtime.build_background = _build_background_with_longitudinal_reference

from . import unified_corrected_physics_gate as _corrected_physics_gate
from .unified_global_longitudinal_physics_gate import install as _install_global_longitudinal_physics_gate

_corrected_physics_gate._SELF_CORRECTION_MODEL = _SELF_CORRECTION_MODEL
_install_global_longitudinal_physics_gate(
    _corrected_physics_gate, _global_longitudinal_reference
)
