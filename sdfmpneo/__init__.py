"""SDF-MPNEO geometry-to-spatial-Joule electrothermal ROM.

Production API:
``geometry -> neural Z/D/cellwise Joule tensors -> geometry-local thermal ROM
from true M(g),K(g) -> explicit current/circuit physics -> temperature``.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from . import unified_model as _unified_model

# Physics truth version 53 keeps the certified v34 EM/source semantics and
# replaces the failed cross-geometry thermal state atlas with a thermal-rank-free
# spatial Joule surrogate.  v53 additionally carries the spatially resolved
# canonical local self defect into parent-grid truth and hard-gates the physical
# PSD/sum-to-D projection.  The network predicts corrected Z_field, D_vol and one
# Hermitian Joule tensor per background cell.  Decoder projection enforces cell
# PSD and exact sum-to-D; each query geometry then assembles its true M(g), K(g)
# and builds a small local rational-Krylov thermal ROM.  No Maxwell solve is used
# during prediction.
#
# EM/source semantics remain unchanged: spiral geometry rejects adjacent-turn
# overlap; the corrected global truth uses the same finite-support terminal source,
# open-boundary power form, canonical local fine-minus-coarse self defect and
# unchanged true-residual certificates.
_unified_model.FORMAT_VERSION = 53
_SELF_CORRECTION_MODEL = "canonical_local_transverse_fine_minus_coarse_self_defect_v2"

from .unified_model import ARCHITECTURE, UnifiedNeuralElectroThermalModel, UnifiedPrediction, UnifiedSteadyState
from .unified_open_boundary import OpenBoundaryBackground
from .unified_resolved_package_fraction import install as _install_resolved_package_fraction
from . import unified_terminal_contact_source as _terminal_contact_source
from .unified_terminal_contact_source import install as _install_terminal_contact_source
from . import unified_charge_regularized_source as _charge_regularized_source
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
from . import unified_terminal_dissipative_defect as _terminal_dissipative_defect
from .unified_terminal_dissipative_defect import install as _install_terminal_dissipative_defect
from .unified_fast_terminal_dissipative_scalar import install as _install_fast_terminal_dissipative_scalar
from .unified_terminal_dissipative_quadrature_fix import install as _install_terminal_dissipative_quadrature_fix
from .unified_fast_reactive_scalar import install as _install_fast_reactive_scalar
from .unified_longitudinal_state_cache import install as _install_longitudinal_state_cache
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
# diagnostics/regression. The balanced terminal-local installer disables that
# uniform production reference and installs the source-scale dissipative truth.
_install_global_dissipative_reference(_global_longitudinal_reference)
_install_resolved_dissipative_reference(
    _global_longitudinal_reference,
    _global_dissipative_reference_impl,
)
_install_terminal_dissipative_defect(
    _global_longitudinal_reference,
    _global_dissipative_reference_impl,
)
# Refined dissipative patches use production-parent material coefficients rather
# than the experimentally falsified geometry-resolved sigma/epsilon reassembly.
# Large scalar systems still use true-residual-certified iterative solvers. The
# material suffix is physics semantics and therefore remains in the model name;
# the solver implementation itself does not alter the truth label.
_material_suffix = "production_parent_piecewise_constant_complex_mass_v1"
_longitudinal_physics_model = f"{_global_longitudinal_reference._MODEL}+{_material_suffix}"
_terminal_dissipative_physics_model = f"{_terminal_dissipative_defect._MODEL}+{_material_suffix}"
_install_fast_terminal_dissipative_scalar(_global_longitudinal_reference)
_global_longitudinal_reference._MODEL = _longitudinal_physics_model
_terminal_dissipative_defect._MODEL = _terminal_dissipative_physics_model
# Lock reference/validation to the same physical source quadrature. This installer
# also applies the v33 single-terminal component lock as its outer source adapter.
_install_terminal_dissipative_quadrature_fix(
    _terminal_contact_source,
    _charge_regularized_source,
    _terminal_dissipative_defect,
    _global_longitudinal_reference,
)
# Reactive terminal refinement keeps its existing discretization but uses the
# same certified fast scalar linear algebra for large systems.
_install_fast_reactive_scalar(_global_longitudinal_reference)
# Reuse compact phi=None reference states already computed by the early audit
# when later preflight correction asks for the identical reference.
_install_longitudinal_state_cache(
    _global_longitudinal_reference,
    _terminal_dissipative_defect,
)
_install_global_longitudinal_reference(_corrected_preflight, _corrected_truth)
_install_longitudinal_preflight_schedule(
    _corrected_preflight, _global_longitudinal_reference
)

from .unified_tensor_surrogate import (
    DecodedTensors,
    SpatialDecodedTensors,
    SpatialTensorDataset,
    TensorDataset,
    UnifiedSpatialTensorSurrogate,
    UnifiedTensorSurrogate,
    decode_physical_tensors,
    decode_spatial_tensors,
    encode_geometry,
    pack_spatial_tensors,
    pack_tensors,
    solve_port_truth_tensors,
    solve_truth_tensors,
)
from .unified_tensor_training import (
    TensorTrainingReport,
    train_matrix_tensor_surrogate,
    train_spatial_tensor_surrogate,
)
from .unified_online_thermal import (
    OnlineThermalReport,
    audit_online_thermal_trajectories,
    build_online_thermal_context,
)
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
    "SpatialDecodedTensors",
    "SpatialTensorDataset",
    "FixedMultiscaleBackground",
    "GeometryAwareThermalLibrary",
    "OpenBoundaryBackground",
    "PackageGeometry",
    "Pose",
    "TensorDataset",
    "TensorTrainingReport",
    "OnlineThermalReport",
    "ThermalBasisReport",
    "UnifiedNeuralElectroThermalModel",
    "UnifiedPrediction",
    "UnifiedSteadyState",
    "UnifiedTensorSurrogate",
    "UnifiedSpatialTensorSurrogate",
    "UnifiedUWPTGeometry",
    "audit_geometry_aware_thermal_trajectories",
    "build_geometry_aware_thermal_library",
    "decode_physical_tensors",
    "decode_spatial_tensors",
    "encode_geometry",
    "pack_tensors",
    "pack_spatial_tensors",
    "sample_geometry",
    "solve_port_truth_tensors",
    "solve_truth_tensors",
    "stretched_axis",
    "train_matrix_tensor_surrogate",
    "train_spatial_tensor_surrogate",
    "build_online_thermal_context",
    "audit_online_thermal_trajectories",
]

# v53 production removes the falsified cross-geometry thermal state atlas.  The
# surrogate predicts corrected Z/D plus a PSD cellwise Joule tensor field; each
# query geometry constructs a small thermal ROM directly from its true M(g),K(g).
# The certified EM preflight/port-field cache keeps its physical signature and
# remains reusable across this thermal architecture change.
from . import unified_runtime as _unified_runtime
_unified_runtime._CACHE_FORMAT = 56
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

# Mesh refinement must compare the same production subgrid truth.  Freeze the
# certified terminal dissipative correction from the production background while
# the 12->9 mm pre/post-basis mesh Gates run; reactive/transverse corrections are
# still recomputed on each Maxwell grid.
from .unified_mesh_gate_terminal_dissipation import install as _install_mesh_gate_terminal_dissipation
_install_mesh_gate_terminal_dissipation(
    _corrected_preflight,
    _corrected_physics_gate,
    _global_longitudinal_reference,
)
