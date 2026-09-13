"""SDF-MPNEO unified residual-corrected neural electrothermal solver.

The public package has one model semantics: analytic/implicit geometry on a
fixed multiscale background, automatically ranked Maxwell/thermal physical
spaces, a neural Maxwell initial guess, true Maxwell residual correction,
physical Joule heating, and hard thermal dynamics.
"""

from .unified_background import BackgroundContext, FixedMultiscaleBackground, stretched_axis
from .unified_basis import BasisReport, build_residual_basis
from .unified_dataset import MaxwellOperatorDataset, generate_operator_dataset, operator_encoding
from .unified_geometry import CoilGeometry, PackageGeometry, Pose, UnifiedUWPTGeometry, sample_geometry
from .unified_maxwell import MaxwellSolveReport, NeuralMaxwellAccelerator
from .unified_model import (
    ARCHITECTURE,
    UnifiedNeuralElectroThermalModel,
    UnifiedPrediction,
    UnifiedSteadyState,
)
from .unified_thermal import ThermalBasisReport, build_thermal_basis
from .unified_trainer import (
    MaxwellTrainingConfig,
    MaxwellTrainingReport,
    train_maxwell_accelerator,
)

__version__ = "0.11.0"

__all__ = [
    "ARCHITECTURE",
    "BackgroundContext",
    "BasisReport",
    "CoilGeometry",
    "FixedMultiscaleBackground",
    "MaxwellOperatorDataset",
    "MaxwellSolveReport",
    "MaxwellTrainingConfig",
    "MaxwellTrainingReport",
    "NeuralMaxwellAccelerator",
    "PackageGeometry",
    "Pose",
    "ThermalBasisReport",
    "UnifiedNeuralElectroThermalModel",
    "UnifiedPrediction",
    "UnifiedSteadyState",
    "UnifiedUWPTGeometry",
    "build_residual_basis",
    "build_thermal_basis",
    "generate_operator_dataset",
    "operator_encoding",
    "sample_geometry",
    "stretched_axis",
    "train_maxwell_accelerator",
]
