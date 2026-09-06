"""SDF-MPNEO executable core."""

from .analytic import (
    AnalyticEvolutionGraph,
    AnalyticSeries,
    ParametricAnalyticEvolutionGraph,
    ParametricAnalyticSeries,
)
from .em import ParametricEMProblem, ReducedEMModel, ResidualGreedyEMReducer
from .model import (
    ExecutableSDFMPNEOModel,
    OnlinePrediction,
    ParametricExecutableSDFMPNEOModel,
    ParametricOnlinePrediction,
)
from .thermal import ThermalSpectralModel

__version__ = "0.3.0"

__all__ = [
    "AnalyticEvolutionGraph",
    "AnalyticSeries",
    "ParametricAnalyticEvolutionGraph",
    "ParametricAnalyticSeries",
    "ThermalSpectralModel",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "ExecutableSDFMPNEOModel",
    "OnlinePrediction",
    "ParametricExecutableSDFMPNEOModel",
    "ParametricOnlinePrediction",
]
