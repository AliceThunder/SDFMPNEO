from .adaptive_block import AdaptiveAggregateEnergyPreconditioner
from .block_riesz import (
    BlockActionFactory,
    PhysicalBlockEnergyPreconditioner,
    SparseLUExactBlockPreconditioner,
    make_physical_block_pcg_riesz_factory,
)
from .certified_riesz import (
    CertifiedEnergyPreconditioner,
    CertifiedPCGRieszAction,
    CertifiedRieszActionResult,
    DiagonalGershgorinEnergyPreconditioner,
    RieszNormDecision,
)
from .compatible import CompatibleAphiDiscretization
from .constitutive import (
    AffineConductivity,
    CompositeCellConductivity,
    ConductivityLaw,
    ConductivityRegion,
    ConstantConductivity,
    ReciprocalLinearResistivity,
)
from .curl_block_riesz import (
    CurlAuxiliaryPhysicalBlockPreconditioner,
    make_curl_auxiliary_physical_pcg_riesz_factory,
)
from .diagnostics import RegionLossProjector, build_region_loss_projector
from .energy_solver import (
    PhysicalEnergySparseApsiSolver,
    apsi_physical_energy_metric,
    solve_physical_energy_certified_apsi,
)
from .face_block_riesz import (
    FaceAuxiliaryPhysicalBlockPreconditioner,
    make_face_auxiliary_physical_pcg_riesz_factory,
)
from .fast_reduced import SparseEnergyReducedEMModel, SparseEnergyResidualGreedyEMReducer
from .grid3d import (
    RectilinearComplex3D,
    SpatialAphiAssembly,
    build_compatible_aphi_from_cells,
    face_loop_source,
)
from .hierarchical_energy import HierarchicalEnergyPreconditioner
from .magnetic_auxiliary import (
    MagneticCurlSubsetEnergyPreconditioner,
    build_gauge_restricted_magnetic_curl_factor,
)
from .magnetic_face_auxiliary import MagneticFaceCirculationEnergyPreconditioner
from .morse_block_riesz import (
    MorseAuxiliaryPhysicalBlockPreconditioner,
    make_morse_auxiliary_physical_pcg_riesz_factory,
)
from .morse_face_auxiliary import MorseFaceCirculationEnergyPreconditioner
from .nonlinear import NonlinearSpatialAphiProblem
from .pair_block import CoupledPairEnergyPreconditioner
from .ports import (
    CertifiedEnergyReducedMultiPortResult,
    CertifiedEnergySparseMultiPortResult,
    CertifiedSparseMultiPortResult,
    ImpressedCurrentPortSet,
    MultiPortImpedanceResult,
)
from .terminal_ports import SolidTerminalPortSet
from .reciprocal_series import ReciprocalSeriesCertificate, certified_reciprocal_polynomials
from .reduced import (
    ParametricEMProblem,
    ReducedEMModel,
    ResidualGreedyEMReducer,
    RieszFactor,
)
from .riesz_action import (
    CertifiedRieszAction,
    RieszActionFactory,
    SparseLUReferenceRieszAction,
)
from .sparse_reduced import (
    SparseEnergyReductionCertificate,
    SparseEnergyResidualCertificate,
)
from .sparse_solver import (
    ApsiBlockTriangularPreconditioner,
    ApsiEnergyMetric,
    CertifiedEnergySparseApsiSolver,
    CertifiedSparseApsiSolver,
    SparseEnergyLinearSolveCertificate,
    SparseLinearSolveCertificate,
    solve_certified_sparse_apsi,
    solve_energy_certified_sparse_apsi,
)
from .tetra import (
    TetrahedralApsiDiscretization,
    build_tetrahedral_apsi_from_thermal_modes,
    tetra_face_loop_source,
)
from .tetra_diagnostics import build_tetrahedral_region_loss_projector
from .tetra_nonlinear import (
    NonlinearTetrahedralApsiProblem,
    TetrahedralConstitutiveCertificate,
)
from .tetra_nonlinear_diagnostics import NonlinearTetrahedralRegionLossEvaluator

__all__ = [
    "AdaptiveAggregateEnergyPreconditioner",
    "BlockActionFactory",
    "PhysicalBlockEnergyPreconditioner",
    "SparseLUExactBlockPreconditioner",
    "make_physical_block_pcg_riesz_factory",
    "CurlAuxiliaryPhysicalBlockPreconditioner",
    "make_curl_auxiliary_physical_pcg_riesz_factory",
    "FaceAuxiliaryPhysicalBlockPreconditioner",
    "make_face_auxiliary_physical_pcg_riesz_factory",
    "MorseAuxiliaryPhysicalBlockPreconditioner",
    "make_morse_auxiliary_physical_pcg_riesz_factory",
    "CertifiedEnergyPreconditioner",
    "CertifiedPCGRieszAction",
    "CertifiedRieszActionResult",
    "DiagonalGershgorinEnergyPreconditioner",
    "CoupledPairEnergyPreconditioner",
    "HierarchicalEnergyPreconditioner",
    "MagneticCurlSubsetEnergyPreconditioner",
    "MagneticFaceCirculationEnergyPreconditioner",
    "MorseFaceCirculationEnergyPreconditioner",
    "build_gauge_restricted_magnetic_curl_factor",
    "RieszNormDecision",
    "CertifiedRieszAction",
    "RieszActionFactory",
    "SparseLUReferenceRieszAction",
    "CompatibleAphiDiscretization",
    "ConductivityLaw",
    "ConstantConductivity",
    "AffineConductivity",
    "ReciprocalLinearResistivity",
    "ConductivityRegion",
    "CompositeCellConductivity",
    "RegionLossProjector",
    "build_region_loss_projector",
    "RectilinearComplex3D",
    "SpatialAphiAssembly",
    "build_compatible_aphi_from_cells",
    "face_loop_source",
    "NonlinearSpatialAphiProblem",
    "ImpressedCurrentPortSet",
    "SolidTerminalPortSet",
    "MultiPortImpedanceResult",
    "CertifiedSparseMultiPortResult",
    "CertifiedEnergySparseMultiPortResult",
    "CertifiedEnergyReducedMultiPortResult",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "RieszFactor",
    "SparseEnergyReducedEMModel",
    "SparseEnergyReductionCertificate",
    "SparseEnergyResidualCertificate",
    "SparseEnergyResidualGreedyEMReducer",
    "ApsiBlockTriangularPreconditioner",
    "ApsiEnergyMetric",
    "CertifiedSparseApsiSolver",
    "CertifiedEnergySparseApsiSolver",
    "SparseLinearSolveCertificate",
    "SparseEnergyLinearSolveCertificate",
    "solve_certified_sparse_apsi",
    "solve_energy_certified_sparse_apsi",
    "PhysicalEnergySparseApsiSolver",
    "apsi_physical_energy_metric",
    "solve_physical_energy_certified_apsi",
    "TetrahedralApsiDiscretization",
    "build_tetrahedral_apsi_from_thermal_modes",
    "tetra_face_loop_source",
    "build_tetrahedral_region_loss_projector",
    "ReciprocalSeriesCertificate",
    "certified_reciprocal_polynomials",
    "NonlinearTetrahedralApsiProblem",
    "TetrahedralConstitutiveCertificate",
    "NonlinearTetrahedralRegionLossEvaluator",
]
