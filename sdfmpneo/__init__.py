"""SDF-MPNEO executable core."""

from .analytic import AnalyticEvolutionGraph, AnalyticSeries
from .em import ParametricEMProblem, ReducedEMModel, ResidualGreedyEMReducer
from .model import ExecutableSDFMPNEOModel, OnlinePrediction
from .thermal import ThermalSpectralModel

__version__ = "0.2.0"

__all__ = [
    "AnalyticEvolutionGraph",
    "AnalyticSeries",
    "ThermalSpectralModel",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
    "ExecutableSDFMPNEOModel",
    "OnlinePrediction",
]
