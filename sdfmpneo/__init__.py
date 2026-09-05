"""SDF-MPNEO minimal executable core."""

from .analytic import AnalyticEvolutionGraph, AnalyticSeries
from .thermal import ThermalSpectralModel
from .em import ParametricEMProblem, ReducedEMModel, ResidualGreedyEMReducer

__version__ = "0.1.0"

__all__ = [
    "AnalyticEvolutionGraph",
    "AnalyticSeries",
    "ThermalSpectralModel",
    "ParametricEMProblem",
    "ReducedEMModel",
    "ResidualGreedyEMReducer",
]
