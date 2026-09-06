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
from .grid3d import (
    RectilinearComplex3D,
    SpatialAphiAssembly,
    build_compatible_aphi_from_cells,
    face_loop_source,
)
from .nonlinear import NonlinearSpatialAphiProblem
from .ports import ImpressedCurrentPortSet, MultiPortImpedanceResult
from .reduced import (
    ParametricEMProblem,
    ReducedEMModel,
    ResidualGreedyEMReducer,
    RieszFactor,
)
from .tetra import (
    TetrahedralApsiDiscretization,
    build_tetrahedral_apsi_from_thermal_modes,
    tetra_face_loop_source,
)
from .tetra_diagnostics import build_tetrahedral_region_loss_projector

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
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "RieszFactor",
    "TetrahedralApsiDiscretization",
    "build_tetrahedral_apsi_from_thermal_modes",
    "tetra_face_loop_source",
    "build_tetrahedral_region_loss_projector",
]
