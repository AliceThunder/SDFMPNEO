"""SDF-MPNEO public API."""

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
from .analytic.fixed_response_network import FixedAnalyticResponseNetwork
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
from .geometry_multichart import CertifiedMultiChartGeometryFamily, MultiChartGeometrySelection
from .model import (
    ExecutableSDFMPNEOModel,
    OnlinePrediction,
    ParametricExecutableSDFMPNEOModel,
    ParametricOnlinePrediction,
)
from .rollout import (
    SegmentedRolloutResult,
    SteadyStateSolveResult,
    rollout_fixed_network,
    solve_physical_steady_state,
)
from .tetra_core import TetrahedralElectroThermalCore
from .thermal import ThermalSpectralModel
from .training import AffineOperatingRHSMap, ElectroThermalResidual, ParametricElectroThermalResidual
from .training.research import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network, train_research_network
from .research import ResearchElectroThermalModel, demo_research_model, model_from_config
from .geometry_research import GeometryResearchModel, geometry_model_from_config

__version__ = "0.10.0"

__all__ = [
    "AnalyticEvolutionGraph", "AnalyticSeries", "ParametricAnalyticEvolutionGraph",
    "ParametricAnalyticSeries", "FixedAnalyticResponseNetwork",
    "CertifiedAnalyticEvolutionOperator", "GeometryConditionedAnalyticEvolutionOperator",
    "MultiChartGeometryAnalyticPrediction", "MultiChartGeometryAnalyticEvolutionOperator",
    "CertifiedElectroThermalVectorField", "CertifiedGeometryElectroThermalFamily",
    "CertifiedMultiChartGeometryFamily", "MultiChartGeometrySelection",
    "ThermalSpectralModel", "ParametricEMProblem", "ReducedEMModel",
    "ResidualGreedyEMReducer", "SparseEnergyReducedEMModel",
    "SparseEnergyResidualGreedyEMReducer", "SolidTerminalPortSet",
    "ExecutableSDFMPNEOModel", "OnlinePrediction", "ParametricExecutableSDFMPNEOModel",
    "ParametricOnlinePrediction", "TetrahedralElectroThermalCore",
    "AffineOperatingRHSMap", "ElectroThermalResidual", "ParametricElectroThermalResidual",
    "ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network",
    "train_research_network", "ResearchElectroThermalModel", "demo_research_model",
    "model_from_config", "GeometryResearchModel", "geometry_model_from_config",
    "SegmentedRolloutResult", "SteadyStateSolveResult", "rollout_fixed_network",
    "solve_physical_steady_state",
]
