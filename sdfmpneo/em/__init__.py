from .compatible import CompatibleAphiDiscretization
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
    "RectilinearComplex3D",
    "SpatialAphiAssembly",
    "build_compatible_aphi_from_cells",
    "face_loop_source",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "RieszFactor",
]
