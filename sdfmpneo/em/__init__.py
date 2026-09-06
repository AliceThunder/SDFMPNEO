from .compatible import CompatibleAphiDiscretization
from .constitutive import (
    AffineConductivity,
    CompositeCellConductivity,
    ConductivityLaw,
    ConductivityRegion,
    ConstantConductivity,
    ReciprocalLinearResistivity,
)
from .diagnostics import RegionLossProjector, build_region_loss_projector
from .energy_solver import (
    PhysicalEnergySparseApsiSolver,
    apsi_physical_energy_metric,
    solve_physical_energy_certified_apsi,
)
from .grid3d import (
    RectilinearComplex3D,
    SpatialAphiAssembly,
    build_compatible_aphi_from_cells,
    face_loop_source,
)
from .nonlinear import NonlinearSpatialAphiProblem
from .ports import (
    CertifiedEnergyReducedMultiPortResult,
    CertifiedEnergySparseMultiPortResult,
    CertifiedSparseMultiPortResult,
    ImpressedCurrentPortSet,
    MultiPortImpedanceResult,
)
from .reciprocal_series import ReciprocalSeriesCertificate, certified_reciprocal_polynomials
from .reduced import (
    ParametricEMProblem,
    ReducedEMModel,
    ResidualGreedyEMReducer,
    RieszFactor,
)
from .sparse_reduced import (
    SparseEnergyReducedEMModel,
    SparseEnergyReductionCertificate,
    SparseEnergyResidualCertificate,
    SparseEnergyResidualGreedyEMReducer,
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
