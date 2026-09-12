from .parametric_residual import AffineOperatingRHSMap, ParametricElectroThermalResidual, ParametricResidualSample
from .residual import ElectroThermalResidual, ResidualSample
from .research import ResearchTrainingConfig, ResearchTrainingReport, make_fixed_network, train_research_network

__all__ = [
    "ElectroThermalResidual", "ResidualSample", "AffineOperatingRHSMap",
    "ParametricElectroThermalResidual", "ParametricResidualSample",
    "ResearchTrainingConfig", "ResearchTrainingReport", "make_fixed_network",
    "train_research_network",
]
