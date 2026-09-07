"""SDF-MPNEO executable core."""

from .analytic import (
    AnalyticEvolutionGraph,
    AnalyticSeries,
    CertifiedAnalyticEvolutionOperator,
    GeometryConditionedAnalyticEvolutionOperator,
    MultiChartGeometryAnalyticEvolutionOperator,
    MultiChartGeometryAnalyticPrediction,
    ParametricAnalyticEvolutionGraph,
    ParametricAnalyticSeries,
)
from .electrothermal import CertifiedElectroThermalVectorField
from .em import (
    ParametricEMProblem,
    ReducedEMModel,
    ResidualGreedyEMReducer,
    SolidTerminalPortSet,
    SparseEnergyReducedEMModel,
    SparseEnergyResidualGreedyEMReducer,
)
from .geometry_family import CertifiedGeometryElectroThermalFamily
from .geometry_multichart import (
    CertifiedMultiChartGeometryFamily,
    MultiChartGeometrySelection,
)
from .model import (
    ExecutableSDFMPNEOModel,
    OnlinePrediction,
    ParametricExecutableSDFMPNEOModel,
    ParametricOnlinePrediction,
)
from .tetra_core import TetrahedralElectroThermalCore
from .thermal import ThermalSpectralModel
from .training import GeometryElectroThermalResidual, SolutionDataFreeClosedLoopTrainer

__version__ = "0.9.0"

__all__ = [
    "AnalyticEvolutionGraph",
    "AnalyticSeries",
    "ParametricAnalyticEvolutionGraph",
    "ParametricAnalyticSeries",
    "CertifiedAnalyticEvolutionOperator",
    "GeometryConditionedAnalyticEvolutionOperator",
    "MultiChartGeometryAnalyticPrediction",
    "MultiChartGeometryAnalyticEvolutionOperator",
    "CertifiedElectroThermalVectorField",
    "CertifiedGeometryElectroThermalFamily",
    "CertifiedMultiChartGeometryFamily",
    "MultiChartGeometrySelection",
    "GeometryElectroThermalResidual",
    "SolutionDataFreeClosedLoopTrainer",
    "ThermalSpectralModel",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "SparseEnergyReducedEMModel",
    "SparseEnergyResidualGreedyEMReducer",
    "SolidTerminalPortSet",
    "ExecutableSDFMPNEOModel",
    "OnlinePrediction",
    "ParametricExecutableSDFMPNEOModel",
    "ParametricOnlinePrediction",
    "TetrahedralElectroThermalCore",
]
