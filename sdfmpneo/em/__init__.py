from .compatible import CompatibleAphiDiscretization
from .diagnostics import RegionLossProjector, build_region_loss_projector
from .grid3d import (
    RectilinearComplex3D,
    SpatialAphiAssembly,
    build_compatible_aphi_from_cells,
    face_loop_source,
)
from .reduced import (
    ParametricEMProblem,
    ReducedEMModel,
    ResidualGreedyEMReducer,
    RieszFactor,
)

__all__ = [
    "CompatibleAphiDiscretization",
    "RegionLossProjector",
    "build_region_loss_projector",
    "RectilinearComplex3D",
    "SpatialAphiAssembly",
    "build_compatible_aphi_from_cells",
    "face_loop_source",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "RieszFactor",
]
